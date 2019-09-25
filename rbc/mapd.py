import os
from collections import defaultdict
from .remotejit import RemoteJIT
from .thrift.utils import resolve_includes
from pymapd.cursor import make_row_results_set
from pymapd._parsers import _extract_description  # , _bind_parameters
from .omnisci_array import array_type_converter


class RemoteMapD(RemoteJIT):

    """Usage:

      mapd = RemoteMapD(host=..., port=...)

      @mapd('int(int, int)')
      def add(a, b):
          return a + b

      mapd.register()

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
                                       'mapd.thrift')
        self.thrift_content = resolve_includes(
            open(thrift_filename).read(),
            [os.path.dirname(thrift_filename)]).replace(
                'completion_hints.', '')
        RemoteJIT.__init__(self, host=host, port=port, **options)

        self.target_info.add_converter(array_type_converter)

    @property
    def version(self):
        return self.thrift_call('get_version')

    _session_id = None
    @property
    def session_id(self):
        if self._session_id is None:
            user = self.user
            pw = self.password
            dbname = self.dbname
            self._session_id = self.thrift_call('connect', user, pw, dbname)
        return self._session_id

    def thrift_call(self, name, *args):
        if self.debug:
            msg = 'thrift_call %s%s' % (name, args)
            if len(msg) > 200:
                msg = msg[:180] + '...' + msg[-15:]
            print(msg)
        return self.make_client()(MapD={name: args})['MapD'][name]

    def sql_execute(self, query):
        self.register()
        columnar = True
        if self.debug:
            print('  %s;' % (query))
        result = self.thrift_call(
            'sql_execute', self.session_id, query, columnar, "", -1, -1)
        descr = _extract_description(result.row_set.row_desc)
        return descr, make_row_results_set(result)

    def register(self):
        if self.have_last_compile:
            return

        functions_map = defaultdict(list)
        for caller in self.callers:
            key = caller.func.__name__, caller.nargs
            functions_map[key].extend(caller._signatures)
        ast_signatures = []
        for (name, nargs), signatures in functions_map.items():
            for i, sig in enumerate(signatures):
                if i == 0:
                    sig.set_mangling('')
                else:
                    sig.set_mangling('__%s' % (i))
                ast_signatures.append(
                    "%s%s '%s'" % (name, sig.mangling, sig.toprototype()))
        ast_signatures = '\n'.join(ast_signatures)

        if self.debug:
            print()
            print(' signatures '.center(80, '-'))
            print(ast_signatures)
        device_params = self.thrift_call('get_device_parameters')
        device_target_map = {}
        for prop, value in device_params.items():
            if prop.endswith('_triple'):
                device = prop.rsplit('_', 1)[0]
                device_target_map[device] = value
        ir_map = self.compile_to_LLVM(targets=device_target_map.values())
        device_ir_map = {}
        for device, target in device_target_map.items():
            llvm_module = ir_map[target]

            require_triple = device_params.get(device+'_triple')
            if require_triple is not None:
                if llvm_module.triple != require_triple:
                    raise RuntimeError(
                        'Expected triple `{}` but LLVM contains `{}`'
                        .format(require_triple, llvm_module.triple))

            require_data_layout = device_params.get(device+'_datalayout')
            if require_data_layout is not None:
                if llvm_module.data_layout != require_data_layout:
                    raise RuntimeError(
                        'Expected data layout `{}` but LLVM contains `{}`'
                        .format(require_data_layout, llvm_module.data_layout))

            ir = device_ir_map[device] = str(llvm_module)
            if self.debug:
                print(('IR[%s]' % (device)).center(80, '-'))
                print(ir)
        if self.debug:
            print('=' * 80)

        return self.thrift_call('register_runtime_udf', self.session_id,
                                ast_signatures, device_ir_map)


class Calcite(object):

    multiplexed = False

    def __init__(self,
                 host='localhost',
                 port=6279,
                 **options):
        self.host = host
        self.port = port
        self.debug = options.pop('debug', False)
        thrift_filename = os.path.join(os.path.dirname(__file__),
                                       'calciteserver.thrift')
        self.thrift_content = resolve_includes(
            open(thrift_filename).read(),
            [os.path.dirname(thrift_filename)]).replace(
                'completion_hints.', '')

    def make_client(self):
        from .thrift import Client
        return Client(
            host=self.host,
            port=self.port,
            multiplexed=self.multiplexed,
            thrift_content=self.thrift_content,
            socket_timeout=10000)

    def thrift_call(self, name, *args):
        if self.debug:
            msg = 'thrift_call %s%s' % (name, args)
            if len(msg) > 200:
                msg = msg[:180] + '...' + msg[-15:]
            print(msg)
        return self.make_client()(CalciteServer={name: args})['CalciteServer'][name]
