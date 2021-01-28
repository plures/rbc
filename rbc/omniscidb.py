import ast
import inspect
import os
import re
import warnings
import configparser
import numpy
from collections import defaultdict, namedtuple
from .remotejit import RemoteJIT
from .thrift.utils import resolve_includes
from . import omnisci_backend
from .omnisci_backend import (
    OmnisciOutputColumnType, OmnisciColumnType,
    OmnisciCompilerPipeline, OmnisciCursorType,
    BufferMeta)
from .targetinfo import TargetInfo
from .irtools import compile_to_LLVM
from .errors import ForbiddenNameError, OmnisciServerError
from .utils import parse_version
from . import typesystem


def get_literal_return(func, verbose=False):
    """Return a literal value from the last function return statement or
    None.
    """
    def _convert(node):
        if isinstance(node, (int, float, complex, str, tuple, list, dict)):
            return node
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Num):
            return _convert(node.n)
        if isinstance(node, ast.Name):
            if node.id in func.__code__.co_names:
                return func.__globals__[node.id]
            args = func.__code__.co_varnames[:func.__code__.co_argcount]
            args_with_default = args[len(args) - len(func.__defaults__):]
            if node.id in args_with_default:
                return func.__defaults__[
                    list(args_with_default).index(node.id)]
            raise NotImplementedError(f'literal value from name `{node.id}`')
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Add):
                return _convert(node.left) + _convert(node.right)
            if isinstance(node.op, ast.Sub):
                return _convert(node.left) - _convert(node.right)
            if isinstance(node.op, ast.Mult):
                return _convert(node.left) * _convert(node.right)
            if isinstance(node.op, ast.Div):
                return _convert(node.left) / _convert(node.right)
            if isinstance(node.op, ast.FloorDiv):
                return _convert(node.left) // _convert(node.right)
            if isinstance(node.op, ast.Mod):
                return _convert(node.left) % _convert(node.right)
            if isinstance(node.op, ast.Pow):
                return _convert(node.left) ** _convert(node.right)
            raise NotImplementedError(
                f'literal value from binary op `{node.op}`')
        raise NotImplementedError(f'literal value from `{node}`')
    source = ''
    intent = None
    for line in inspect.getsourcelines(func)[0]:
        if line.lstrip().startswith('@'):
            continue
        if intent is None:
            intent = len(line) - len(line.lstrip())
        source += line[intent:]

    tree = ast.parse(source)
    last_node = tree.body[0].body[-1]
    if isinstance(last_node, ast.Return):
        try:
            return _convert(last_node.value)
        except Exception as msg:
            if verbose:
                print(f'get_literal_return: {msg}')
                print(source)


def is_available(_cache={}):
    """Return version tuple and None if OmnisciDB server is accessible or
    recent enough. Otherwise return None and the reason about
    unavailability.
    """
    if not _cache:
        config = get_client_config()
        omnisci = RemoteOmnisci(**config)
        try:
            version = omnisci.version
        except Exception as msg:
            _cache['reason'] = 'failed to get OmniSci version: %s' % (msg)
        else:
            print(' OmnisciDB version', version)
            if version[:2] >= (4, 6):
                _cache['version'] = version
            else:
                _cache['reason'] = (
                    'expected OmniSci version 4.6 or greater, got %s'
                    % (version,))
    return _cache.get('version', ()), _cache.get('reason', '')


def get_client_config(**config):
    """Retrieve the omnisci client configuration parameters from a client
    home directory.

    Note that here the client configurations parameters are those that
    are used to configure the client software such as rbc or pymapd.
    This is different from omnisci instance configuration described in
    https://docs.omnisci.com/latest/4_configuration.html that is used
    for configuring the omniscidb server software.

    In Linux clients, the omnisci client configuration is read from:
    :code:`$HOME/.config/omnisci/client.conf`

    In Windows clients, the configuration is read from
    :code:`%UserProfile/.config/omnisci/client.conf` or
    :code:`%AllUsersProfile/.config/omnisci/client.conf`

    When :code:`OMNISCI_CLIENT_CONF` environment variable is defined then
    the configuration is read from the file specified in this
    variable.

    The configuration file must use configuration language similar to
    one used in MS Windows INI files. For omnisci client
    configuration, the file may contain, for instance::

      [user]
      name: <OmniSciDB user name, defaults to admin>
      password: <OmniSciDB user password>

      [server]
      host: <OmniSciDB server host name or IP, defaults to localhost>
      port: <OmniSciDB server port, defaults to 6274>

    Parameters
    ----------
    config : dict
      Specify configuration parameters that override the parameters
      from configuration file.

    Returns
    -------
    config : dict
      A dictionary of `user`, `password`, `host`, `port`, `dbname` and
      other RemoteJIT options.

    """
    _config = dict(user='admin', password='HyperInteractive',
                   host='localhost', port=6274, dbname='omnisci')
    _config.update(**config)
    config = _config

    conf_file = os.environ.get('OMNISCI_CLIENT_CONF', None)
    if conf_file is not None and not os.path.isfile(conf_file):
        print(f'rbc.omnisci.get_client_config:'  # noqa: F541
              ' OMNISCI_CLIENT_CONF={conf_file!r}'
              ' is not a file, ignoring.')
        conf_file = None
    if conf_file is None:
        conf_file_base = os.path.join('.config', 'omnisci', 'client.conf')
        for prefix_env in ['UserProfile', 'AllUsersProfile', 'HOME']:
            prefix = os.environ.get(prefix_env, None)
            if prefix is not None:
                fn = os.path.join(prefix, conf_file_base)
                if os.path.isfile(fn):
                    conf_file = fn
                    break
    if conf_file is None:
        return config

    conf = configparser.ConfigParser()
    conf.read(conf_file)

    if 'user' in conf:
        user = conf['user']
        if 'name' in user:
            config['user'] = user['name']
        if 'password' in user:
            config['password'] = user['password']

    if 'server' in conf:
        server = conf['server']
        if 'host' in server:
            config['host'] = server['host']
        if 'port' in server:
            config['port'] = int(server['port'])

    if 'rbc' in conf:
        rbc = conf['rbc']
        for k in ['debug', 'use_host_target']:
            if k in rbc:
                config[k] = rbc.getboolean(k)

    return config


class RemoteOmnisci(RemoteJIT):

    """Usage:

    .. highlight:: python
    .. code-block:: python

        omnisci = RemoteOmnisci(host=..., port=...)

        @omnisci('int(int, int)')
        def add(a, b):
            return a + b

        omnisci.register()

    Use pymapd, for instance, to make a SQL query `select add(c1,
    c2) from table`
    """
    multiplexed = False
    mangle_prefix = ''

    def __init__(self,
                 user='admin',
                 password='HyperInteractive',
                 host='localhost',
                 port=6274,
                 dbname='omnisci',
                 **options):
        self.user = user
        self.password = password
        self.dbname = dbname

        thrift_filename = os.path.join(os.path.dirname(__file__),
                                       'omnisci.thrift')
        content = resolve_includes(
            open(thrift_filename).read(),
            [os.path.dirname(thrift_filename)])
        for p in ['common.', 'extension_functions.']:
            content = content.replace(p, '')
        self.thrift_content = content

        RemoteJIT.__init__(self, host=host, port=port, **options)

        self._version = None
        self._thrift_client = None
        self._session_id = None
        self._targets = None
        self.thrift_typemap = defaultdict(dict)
        self._init_thrift_typemap()
        self.has_cuda = None
        self._null_values = dict()

        # An user-defined device-LLVM IR mapping.
        self.user_defined_llvm_ir = {}

    def _init_thrift_typemap(self):
        """Initialize thrift type map using client thrift configuration.
        """
        typemap = self.thrift_typemap
        for typename, typ in self.thrift_client.thrift.__dict__.items():
            if hasattr(typ, '_NAMES_TO_VALUES'):
                for name, value in typ._NAMES_TO_VALUES.items():
                    typemap[typename][name] = value

    @property
    def version(self):
        if self._version is None:
            version = self.thrift_call('get_version')
            return parse_version(version)
        return self._version

    @property
    def session_id(self):
        if self._session_id is None:
            user = self.user
            pw = self.password
            dbname = self.dbname
            self._session_id = self.thrift_call('connect', user, pw, dbname)
        return self._session_id

    @property
    def thrift_client(self):
        if self._thrift_client is None:
            self._thrift_client = self.client
        return self._thrift_client

    def thrift_call(self, name, *args, **kwargs):
        client = kwargs.get('client')
        if client is None:
            client = self.thrift_client

        if name == 'register_runtime_udf' and self.version[:2] >= (4, 9):
            warnings.warn('Using register_runtime_udf is deprecated, '
                          'use register_runtime_extension_functions instead.')
            session_id, signatures, device_ir_map = args
            udfs = []
            sig_re = re.compile(r"\A(?P<name>[\w\d_]+)\s+"
                                r"'(?P<rtype>[\w\d_]+)[(](?P<atypes>.*)[)]'\Z")
            thrift = client.thrift
            ext_arguments_map = self._get_ext_arguments_map()
            for signature in signatures.splitlines():
                m = sig_re.match(signature)
                if m is None:
                    raise RuntimeError(
                        'Failed to parse signature %r' % (signature))
                name = m.group('name')
                rtype = ext_arguments_map[m.group('rtype')]
                atypes = [ext_arguments_map[a.strip()]
                          for a in m.group('atypes').split(',')]
                udfs.append(thrift.TUserDefinedFunction(
                    name, atypes, rtype))
            return self.thrift_call('register_runtime_extension_functions',
                                    session_id, udfs, [], device_ir_map)

        if self.debug:
            msg = 'thrift_call %s%s' % (name, args)
            if len(msg) > 200:
                msg = msg[:180] + '...' + msg[-15:]
            print(msg)
        try:
            return client(Omnisci={name: args})['Omnisci'][name]
        except client.thrift.TMapDException as msg:
            m = re.match(r'.*CalciteContextException.*('
                         r'No match found for function signature.*'
                         r'|Cannot apply.*)',
                         msg.error_msg)
            if m:
                raise OmnisciServerError(f'[Calcite] {m.group(1)}')
            m = re.match(r'.*Exception: (.*)', msg.error_msg)
            if m:
                raise OmnisciServerError(f'{m.group(1)}')
            # TODO: catch more known server failures here.
            raise

    def get_tables(self):
        """Return a list of table names stored in the OmnisciDB server.
        """
        return self.thrift_call('get_tables', self.session_id)

    def get_table_details(self, table_name):
        """Return details about OmnisciDB table.
        """
        return self.thrift_call('get_table_details',
                                self.session_id, table_name)

    def load_table_columnar(self, table_name, **columnar_data):
        """Load columnar data to OmnisciDB table.

        Warning: when connected to OmnisciDB < 5.3, the data is loaded
          row-wise that will be very slow for large data sets.

        Parameters
        ----------
        table_name : str
          The name of table into the data is loaded. The table must exist.

        columnar_data : dict
          A dictionary of column names and the corresponding column
          values. A column values is a sequence of values.
          The table must have the specified columns defined.

        """
        int_col_types = ['TINYINT', 'SMALLINT', 'INT', 'BIGINT', 'BOOL',
                         'DECIMAL', 'TIME', 'TIMESTAMP', 'DATE']
        real_col_types = ['FLOAT', 'DOUBLE']
        str_col_types = ['STR', 'POINT', 'LINESTRING', 'POLYGON',
                         'MULTIPOLYGON']

        datumtype_map = {
            'BOOL': 'boolean8',
            'TINYINT': 'int8',
            'SMALLINT': 'int16',
            'INT': 'int32',
            'BIGINT': 'int64',
            'FLOAT': 'float32',
            'DOUBLE': 'float64',
            'BOOL[]': 'Array<boolean8>',
            'TINYINT[]': 'Array<int8>',
            'SMALLINT[]': 'Array<int16>',
            'INT[]': 'Array<int32>',
            'BIGINT[]': 'Array<int64>',
            'FLOAT[]': 'Array<float32>',
            'DOUBLE[]': 'Array<float64>',
        }

        thrift = self.thrift_client.thrift
        table_details = self.get_table_details(table_name)
        use_sql = self.version[:2] < (5, 3)
        if not use_sql and self.version[:2] < (5, 5):
            for column_data in columnar_data.values():
                for v in column_data:
                    if v is None:
                        break
                    elif isinstance(v, (tuple, list)):
                        for _v in v:
                            if _v is None:
                                break
                        else:
                            continue
                        break
                else:
                    continue
                use_sql = True
                break
        if use_sql:
            rows = None
            for column_name, column_data in columnar_data.items():
                if rows is None:
                    rows = [[None for j in range(len(columnar_data))]
                            for i in range(len(column_data))]
                col_index = None
                for i, ct in enumerate(table_details.row_desc):
                    if ct.col_name == column_name:
                        typeinfo = ct.col_type
                        col_index = i
                        datumtype = thrift.TDatumType._VALUES_TO_NAMES[
                            typeinfo.type]
                        is_array = typeinfo.is_array
                        break
                assert col_index is not None
                for i, v in enumerate(column_data):
                    if v is None:
                        v = "NULL"
                    elif is_array:
                        if datumtype == 'BOOL':
                            v = ["'true'" if v_ else ("'false'"
                                                      if v_ is not None else "NULL")
                                 for v_ in v]
                        else:
                            v = [(str(v_) if v_ is not None else "NULL") for v_ in v]
                        v = ', '.join(v)
                        v = f'ARRAY[{v}]'
                    else:
                        if datumtype == 'BOOL':
                            v = "'true'" if v else "'false'"
                    rows[i][col_index] = v

            for row in rows:
                table_row = ', '.join(map(str, row))
                self.sql_execute(
                    f'INSERT INTO {table_name} VALUES ({table_row})')
            return

        columns = []
        for column_name, column_data in columnar_data.items():
            typeinfo = None
            for ct in table_details.row_desc:
                if ct.col_name == column_name:
                    typeinfo = ct.col_type
                    break
            if typeinfo is None:
                raise ValueError(
                    f'OmnisciDB `{table_name}` has no column `{column_name}`')
            datumtype = thrift.TDatumType._VALUES_TO_NAMES[typeinfo.type]
            nulls = [v is None for v in column_data]
            if typeinfo.is_array:
                arrays = []
                for arr in column_data:
                    if arr is None:
                        arr_nulls = None
                    else:
                        arr_nulls = [v is None for v in arr]
                        if True in arr_nulls:
                            # TODO: null support for text
                            null_value = self._null_values[datumtype_map[datumtype]]
                            arr = [(null_value if v is None else v) for v in arr]
                    if datumtype in int_col_types:
                        arr_data = thrift.TColumnData(int_col=arr)
                    elif datumtype in real_col_types:
                        arr_data = thrift.TColumnData(real_col=arr)
                    elif datumtype in str_col_types:
                        arr_data = thrift.TColumnData(str_col=arr)
                    else:
                        raise NotImplementedError(
                            f'loading {datumtype} array data')
                    arrays.append(thrift.TColumn(data=arr_data, nulls=arr_nulls))
                col_data = thrift.TColumnData(arr_col=arrays)
            else:
                if True in nulls:
                    # TODO: null support for text
                    null_value = self._null_values[datumtype_map[datumtype]]
                    column_data = [(null_value if v is None else v) for v in column_data]
                if datumtype in int_col_types:
                    col_data = thrift.TColumnData(int_col=column_data)
                elif datumtype in real_col_types:
                    col_data = thrift.TColumnData(real_col=column_data)
                elif datumtype in str_col_types:
                    col_data = thrift.TColumnData(str_col=column_data)
                else:
                    raise NotImplementedError(f'loading {datumtype} data')
            columns.append(thrift.TColumn(data=col_data, nulls=nulls))

        self.thrift_call('load_table_binary_columnar',
                         self.session_id, table_name, columns)

    def _make_row_results_set(self, data):
        # The following code is a stripped copy from omnisci/pymapd

        _typeattr = {
            'SMALLINT': 'int',
            'INT': 'int',
            'BIGINT': 'int',
            'TIME': 'int',
            'TIMESTAMP': 'int',
            'DATE': 'int',
            'BOOL': 'int',
            'FLOAT': 'real',
            'DECIMAL': 'real',
            'DOUBLE': 'real',
            'STR': 'str',
            'POINT': 'str',
            'LINESTRING': 'str',
            'POLYGON': 'str',
            'MULTIPOLYGON': 'str',
            'TINYINT': 'int',
            'GEOMETRY': 'str',
            'GEOGRAPHY': 'str',
        }

        thrift = self.thrift_client.thrift

        def extract_col_vals(desc, val):
            typename = thrift.TDatumType._VALUES_TO_NAMES[desc.col_type.type]
            nulls = val.nulls
            if hasattr(val.data, 'arr_col') and val.data.arr_col:
                vals = [
                    None if null else [
                        None if _n else _v
                        for _n, _v in zip(
                                v.nulls, getattr(v.data, _typeattr[typename] + '_col'))]
                    for null, v in zip(nulls, val.data.arr_col)
                ]
            else:
                vals = getattr(val.data, _typeattr[typename] + '_col')
                vals = [None if null else v for null, v in zip(nulls, vals)]
            return vals

        if data.row_set.columns:
            nrows = len(data.row_set.columns[0].nulls)
            ncols = len(data.row_set.row_desc)
            columns = [
                extract_col_vals(desc, col)
                for desc, col in zip(data.row_set.row_desc, data.row_set.columns)
            ]
            for i in range(nrows):
                yield tuple(columns[j][i] for j in range(ncols))

    def sql_execute(self, query):
        """Execute SQL query in OmnisciDB server.

        Parameters
        ----------
        query : str

          SQL query string containing exactly one query. Multiple
          queries are not supported.

        Returns
        -------
        descr : object
          Row description object
        results : iterator
          Iterator over rows.
        """
        self.register()
        columnar = True
        if self.debug:
            print('  %s;' % (query))
        result = self.thrift_call(
            'sql_execute', self.session_id, query, columnar, "", -1, -1)

        Description = namedtuple("Description", ["name", "type_code", "null_ok"])
        descr = []
        for col in result.row_set.row_desc:
            descr.append(Description(col.col_name, col.col_type.type, col.col_type.nullable))
        return descr, self._make_row_results_set(result)

    _ext_arguments_map = None

    def _get_ext_arguments_map(self):
        if self._ext_arguments_map is not None:
            return self._ext_arguments_map
        thrift = self.thrift_client.thrift
        typemap = self.thrift_typemap
        ext_arguments_map = {
            'int8': typemap['TExtArgumentType']['Int8'],
            'int16': typemap['TExtArgumentType']['Int16'],
            'int32': typemap['TExtArgumentType']['Int32'],
            'int64': typemap['TExtArgumentType']['Int64'],
            'float32': typemap['TExtArgumentType']['Float'],
            'float64': typemap['TExtArgumentType']['Double'],
            'int8*': typemap['TExtArgumentType']['PInt8'],
            'int16*': typemap['TExtArgumentType']['PInt16'],
            'int32*': typemap['TExtArgumentType']['PInt32'],
            'int64*': typemap['TExtArgumentType']['PInt64'],
            'float32*': typemap['TExtArgumentType']['PFloat'],
            'float64*': typemap['TExtArgumentType']['PDouble'],
            'bool': typemap['TExtArgumentType']['Bool'],
            'bool*': typemap['TExtArgumentType'].get('PBool'),
            'Array<bool>': typemap['TExtArgumentType'].get(
                'ArrayBool', typemap['TExtArgumentType']['ArrayInt8']),
            'Array<int8_t>': typemap['TExtArgumentType']['ArrayInt8'],
            'Array<int16_t>': typemap['TExtArgumentType']['ArrayInt16'],
            'Array<int32_t>': typemap['TExtArgumentType']['ArrayInt32'],
            'Array<int64_t>': typemap['TExtArgumentType']['ArrayInt64'],
            'Array<float>': typemap['TExtArgumentType']['ArrayFloat'],
            'Array<double>': typemap['TExtArgumentType']['ArrayDouble'],
            'Column<bool>': typemap['TExtArgumentType'].get('ColumnBool'),
            'Column<int8_t>': typemap['TExtArgumentType'].get('ColumnInt8'),
            'Column<int16_t>': typemap['TExtArgumentType'].get('ColumnInt16'),
            'Column<int32_t>': typemap['TExtArgumentType'].get('ColumnInt32'),
            'Column<int64_t>': typemap['TExtArgumentType'].get('ColumnInt64'),
            'Column<float>': typemap['TExtArgumentType'].get('ColumnFloat'),
            'Column<double>': typemap['TExtArgumentType'].get('ColumnDouble'),
            'Cursor': typemap['TExtArgumentType']['Cursor'],
            'void': typemap['TExtArgumentType']['Void'],
            'GeoPoint': typemap['TExtArgumentType'].get('GeoPoint'),
            'GeoLineString': typemap['TExtArgumentType'].get('GeoLineString'),
            'GeoPolygon': typemap['TExtArgumentType'].get('GeoPolygon'),
            'GeoMultiPolygon': typemap['TExtArgumentType'].get(
                'GeoMultiPolygon'),
            'Bytes': typemap['TExtArgumentType'].get('TextEncodingNone'),
            'Text<8>': typemap['TExtArgumentType'].get('TextEncodingDict8'),
            'Text<16>': typemap['TExtArgumentType'].get('TextEncodingDict16'),
            'Text<32>': typemap['TExtArgumentType'].get('TextEncodingDict32'),
        }

        if self.version[:2] < (5, 4):
            ext_arguments_map['Array<bool>'] = typemap[
                'TExtArgumentType']['ArrayInt8']

        for ptr_type, T in [
                ('bool', 'bool'),
                ('int8', 'int8_t'),
                ('int16', 'int16_t'),
                ('int32', 'int32_t'),
                ('int64', 'int64_t'),
                ('float32', 'float'),
                ('float64', 'double'),
        ]:
            ext_arguments_map['OmnisciArrayType<%s>' % ptr_type] \
                = ext_arguments_map.get('Array<%s>' % T)
            ext_arguments_map['OmnisciColumnType<%s>' % ptr_type] \
                = ext_arguments_map.get('Column<%s>' % T)
            ext_arguments_map['OmnisciOutputColumnType<%s>' % ptr_type] \
                = ext_arguments_map.get('Column<%s>' % T)

        ext_arguments_map['OmnisciBytesType<char8>'] = ext_arguments_map.get('Bytes')

        values = list(ext_arguments_map.values())
        for v, n in thrift.TExtArgumentType._VALUES_TO_NAMES.items():
            if v not in values:
                warnings.warn('thrift.TExtArgumentType.%s(=%s) value not '
                              'in ext_arguments_map' % (n, v))
        self._ext_arguments_map = ext_arguments_map
        return ext_arguments_map

    def type_to_extarg(self, t):
        if isinstance(t, typesystem.Type):
            s = t.tostring(use_annotation=False)
            extarg = self._get_ext_arguments_map().get(s)
            if extarg is None:
                raise ValueError(f'cannot convert {t} to ExtArgumentType')
            return extarg
        elif isinstance(t, str):
            extarg = self._get_ext_arguments_map().get(t)
            if extarg is None:
                raise ValueError(f'cannot convert {t} to ExtArgumentType')
            return extarg
        elif isinstance(t, tuple):
            return tuple(map(self.type_to_extarg, t))
        else:
            raise TypeError(f'expected typesystem Type|tuple|str, got {type(t).__name__}')

    def retrieve_targets(self):
        device_params = self.thrift_call('get_device_parameters',
                                         self.session_id)
        thrift = self.thrift_client.thrift
        typemap = self.thrift_typemap
        device_target_map = {}
        messages = []
        for prop, value in device_params.items():
            if prop.endswith('_triple'):
                device = prop.rsplit('_', 1)[0]
                device_target_map[device] = value

            # Update thrift type map from server configuration
            if 'Type.' in prop:
                typ, member = prop.split('.', 1)
                ivalue = int(value)
                # Any warnings below indicate that the client and
                # server thrift configurations are different. Server
                # configuration will win.
                if typ not in typemap:
                    messages.append(f'thrift type map: add new type {typ}')
                elif member not in typemap[typ]:
                    messages.append(
                        f'thrift type map: add new member {typ}.{member}')
                elif ivalue != typemap[typ][member]:
                    messages.append(
                        f'thrift type map: update {typ}.{member}'
                        f' to {ivalue} (was {typemap[typ][member]})')
                else:
                    continue
                typemap[typ][member] = ivalue
                thrift_type = getattr(thrift, typ, None)
                if thrift_type is not None:
                    thrift_type._VALUES_TO_NAMES[ivalue] = member
                    thrift_type._NAMES_TO_VALUES[member] = ivalue
                self._ext_arguments_map = None
        if messages:
            warnings.warn('\n  '.join([''] + messages))

        null_values = device_params.get('null_values')
        null_values_asint = dict()
        null_values_astype = dict()
        if null_values is not None:
            for tname_value in null_values.split(';'):
                if not tname_value:
                    continue
                tname, value = tname_value.split(':')
                null_values_asint[tname] = int(value)
                dtype = tname
                if dtype.startswith('Array<'):
                    dtype = dtype[6:-1]
                bitwidth = int(''.join(filter(str.isdigit, dtype)))
                if bitwidth > 1:
                    null_value = numpy.dtype(f'uint{bitwidth}').type(int(value)).view(
                        'int8' if dtype == 'boolean8' else dtype)
                    null_values_astype[tname] = null_value
        self._null_values = null_values_astype

        targets = {}
        for device, target in device_target_map.items():
            target_info = TargetInfo(name=device)
            for prop, value in device_params.items():
                if not prop.startswith(device + '_'):
                    continue
                target_info.set(prop[len(device)+1:], value)
            if device == 'gpu':
                self.has_cuda = True
                # TODO: remove this hack
                # see https://github.com/numba/numba/issues/4546
                target_info.set('name', 'skylake')
            targets[device] = target_info

            if target_info.is_cpu:
                target_info.set('driver', 'none')
                target_info.add_library('m')
                target_info.add_library('stdio')
                target_info.add_library('stdlib')
                target_info.add_library('omniscidb')
            elif target_info.is_gpu and self.version >= (5, 5):
                target_info.add_library('libdevice')

            version_str = '.'.join(map(str, self.version[:3])) + self.version[3]
            target_info.set('software', f'OmnisciDB {version_str}')

            llvm_version = device_params.get('llvm_version')
            if llvm_version is not None:
                target_info.set('llvm_version', tuple(map(int, llvm_version.split('.'))))

            target_info.set('null_values', null_values_asint)
        return targets

    @property
    def forbidden_names(self):
        """Return a list of forbidden function names. See
        https://github.com/xnd-project/rbc/issues/32
        """
        if self.version < (5, 2):
            return ['sinh', 'cosh', 'tanh', 'rint', 'trunc', 'expm1',
                    'exp2', 'log2', 'log1p', 'fmod']
        return []

    def _make_udtf(self, caller, orig_sig, sig):
        if self.version < (5, 4):
            v = '.'.join(map(str, self.version))
            raise RuntimeError(
                'UDTFs with Column arguments require '
                'omniscidb 5.4 or newer, currently '
                'connected to ', v)
        thrift = self.thrift_client.thrift
        sizer_map = dict(
            ConstantParameter='kUserSpecifiedConstantParameter',
            RowMultiplier='kUserSpecifiedRowMultiplier',
            Constant='kConstant')

        unspecified = object()
        inputArgTypes = []
        outputArgTypes = []
        sqlArgTypes = []
        sizer = None
        sizer_index = -1

        consumed_index = 0
        name = caller.func.__name__
        for i, a in enumerate(orig_sig[1]):
            annot = a.annotation()
            _sizer = annot.get('sizer', unspecified)
            if _sizer is not unspecified:
                if not (a.is_int and a.bits == 32):
                    raise ValueError(
                        'sizer argument must have type int32')
                if _sizer is None:
                    _sizer = 'RowMultiplier'
                _sizer = sizer_map[_sizer]
                # cannot have multiple sizer arguments
                assert sizer_index == -1
                sizer_index = consumed_index + 1
                sizer = _sizer

            if isinstance(a, OmnisciCursorType):
                sqlArgTypes.append(self.type_to_extarg('Cursor'))
                for a_ in a.as_consumed_args:
                    assert not isinstance(
                        a_, OmnisciOutputColumnType), (a_)
                    inputArgTypes.append(self.type_to_extarg(a_))
                    consumed_index += 1
            else:
                if isinstance(a, OmnisciOutputColumnType):
                    atype = self.type_to_extarg(a)
                    outputArgTypes.append(atype)
                else:
                    atype = self.type_to_extarg(a)
                    if isinstance(a, OmnisciColumnType):
                        sqlArgTypes.append(self.type_to_extarg('Cursor'))
                        inputArgTypes.append(atype)
                    else:
                        sqlArgTypes.append(atype)
                        inputArgTypes.append(atype)
                consumed_index += 1
        if sizer is None:
            sizer = 'kConstant'
        if sizer == 'kConstant':
            sizer_index = get_literal_return(
                caller.func, verbose=self.debug)
            if sizer_index is None:
                raise TypeError(
                    f'Table function `{caller.func.__name__}`'
                    ' has no sizing parameter nor has'
                    ' it `return <literal value>` statement')
        if sizer_index < 0:
            raise ValueError(
                f'Table function `{caller.func.__name__}`'
                ' sizing parameter must be non-negative'
                f' integer (got {sizer_index})')
        sizer_type = (thrift.TOutputBufferSizeType
                      ._NAMES_TO_VALUES[sizer])
        return thrift.TUserDefinedTableFunction(
            name + sig.mangling(),
            sizer_type, sizer_index,
            inputArgTypes, outputArgTypes, sqlArgTypes)

    def _make_udtf_old(self, caller, orig_sig, sig):
        # old style UDTF for omniscidb <= 5.3, to be deprecated
        if self.version >= (5, 4):
            v = '.'.join(map(str, self.version))
            raise RuntimeError(
                'Old-style UDTFs require '
                'omniscidb 5.3 or older, currently '
                'connected to ', v)
        thrift = self.thrift_client.thrift

        sizer = None
        sizer_index = -1
        inputArgTypes = []
        outputArgTypes = []
        sqlArgTypes = []
        name = caller.func.__name__
        for i, a in enumerate(sig[1]):
            _sizer = a.annotation().get('sizer')
            if _sizer is not None:
                # expect no more than one sizer argument
                assert sizer_index == -1
                sizer_index = i + 1
                sizer = _sizer
            atype = self.type_to_extarg(a)
            if 'output' in a.annotation():
                outputArgTypes.append(atype)
            else:
                if 'input' in a.annotation():
                    sqlArgTypes.append(atype)
                elif 'cursor' in a.annotation():
                    sqlArgTypes.append(self.type_to_extarg('Cursor'))
                inputArgTypes.append(atype)
        if sizer is None:
            sizer = 'kConstant'
        sizer_type = (thrift.TOutputBufferSizeType
                      ._NAMES_TO_VALUES[sizer])
        return thrift.TUserDefinedTableFunction(
            name + sig.mangling(),
            sizer_type, sizer_index,
            inputArgTypes, outputArgTypes, sqlArgTypes)

    def _make_udf(self, caller, orig_sig, sig):
        name = caller.func.__name__
        thrift = self.thrift_client.thrift
        rtype = self.type_to_extarg(sig[0])
        atypes = self.type_to_extarg(sig[1])
        return thrift.TUserDefinedFunction(
            name + sig.mangling(),
            atypes, rtype)

    def register(self):
        if self.have_last_compile:
            return

        def is_udtf(sig):
            for a in sig[1]:
                if isinstance(a, (OmnisciOutputColumnType, OmnisciColumnType)):
                    return True
            return False

        device_ir_map = {}
        llvm_function_names = []
        fid = 0  # UDF/UDTF id
        udfs, udtfs = [], []
        for device, target_info in self.targets.items():
            with target_info:
                udfs_map, udtfs_map = {}, {}
                functions_and_signatures = []
                function_signatures = defaultdict(list)
                for caller in reversed(self.get_callers()):
                    signatures = {}
                    name = caller.func.__name__
                    if name in self.forbidden_names:
                        raise ForbiddenNameError(
                            f'\n\nAttempt to define function with name `{name}`.\n'
                            f'As a workaround, add a prefix to the function name '
                            f'or define it with another name:\n\n'
                            f'   def prefix_{name}(x):\n'
                            f'       return np.trunc(x)\n\n'
                            f'For more information, see: '
                            f'https://github.com/xnd-project/rbc/issues/32')
                    for sig in caller.get_signatures():
                        i = len(function_signatures[name])
                        if sig in function_signatures[name]:
                            if self.debug:
                                f2 = os.path.basename(
                                    caller.func.__code__.co_filename)
                                n2 = caller.func.__code__.co_firstlineno
                                print(f'{type(self).__name__}.register: ignoring'
                                      f' older definition of `{name}` for `{sig}`'
                                      f' in {f2}#{n2}.')
                            continue
                        fid += 1
                        orig_sig = sig
                        sig = sig[0](*sig.argument_types)
                        function_signatures[name].append(sig)
                        sig_is_udtf = is_udtf(sig)
                        is_old_udtf = 'table' in sig[0].annotation()

                        if i == 0 and (self.version < (5, 2) or is_old_udtf or
                                       (self.version < (5, 5) and sig_is_udtf)):
                            sig.set_mangling('')
                        else:
                            if self.version < (5, 5):
                                sig.set_mangling('__%s' % (i))
                            else:
                                sig.set_mangling('__%s_%s' % (device, i))

                        if sig_is_udtf:
                            # new style UDTF, requires omniscidb version >= 5.4
                            udtfs_map[fid] = self._make_udtf(caller, orig_sig, sig)
                        elif is_old_udtf:
                            # old style UDTF for omniscidb <= 5.3, to be deprecated
                            udtfs_map[fid] = self._make_udtf_old(caller, orig_sig, sig)
                        else:
                            udfs_map[fid] = self._make_udf(caller, orig_sig, sig)
                        signatures[fid] = sig
                    functions_and_signatures.append((caller.func, signatures))

                llvm_module, succesful_fids = compile_to_LLVM(
                    functions_and_signatures,
                    target_info,
                    pipeline_class=OmnisciCompilerPipeline,
                    user_defined_llvm_ir=self.user_defined_llvm_ir.get(device),
                    debug=self.debug)

                assert llvm_module.triple == target_info.triple
                assert llvm_module.data_layout == target_info.datalayout
                for f in llvm_module.functions:
                    llvm_function_names.append(f.name)

                device_ir_map[device] = str(llvm_module)
                skipped_names = []
                for fid, udf in udfs_map.items():
                    if fid in succesful_fids:
                        udfs.append(udf)
                    else:
                        skipped_names.append(udf.name)

                for fid, udtf in udtfs_map.items():
                    if fid in succesful_fids:
                        udtfs.append(udtf)
                    else:
                        skipped_names.append(udtf.name)
                if self.debug:
                    print(f'Skipping: {", ".join(skipped_names)}')
        # Make sure that all registered functions have
        # implementations, otherwise, we will crash the server.
        for f in udtfs:
            assert f.name in llvm_function_names
        for f in udfs:
            assert f.name in llvm_function_names

        if self.debug:
            names = ", ".join([f.name for f in udfs] + [f.name for f in udtfs])
            print(f'Registering: {names}')

        self.set_last_compile(device_ir_map)
        return self.thrift_call(
            'register_runtime_extension_functions',
            self.session_id, udfs, udtfs, device_ir_map)

    def preprocess_callable(self, func):
        func = super().preprocess_callable(func)
        if 'omnisci_backend' not in func.__code__.co_names:
            for symbol in BufferMeta.class_names:
                if symbol in func.__code__.co_names and symbol not in func.__globals__:
                    warnings.warn(
                        f'{func.__name__} uses {symbol} that may be undefined.'
                        f' Inserting {symbol} to global namespace.'
                        f' Use `from rbc.omnisci_backend import {symbol}`'
                        ' to remove this warning.')
                    func.__globals__[symbol] = omnisci_backend.__dict__.get(symbol)
        return func
