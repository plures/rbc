import pytest
rbc_mapd = pytest.importorskip('rbc.mapd')


def mapd_is_available():
    """Return True if MapD server is accessible.
    """
    mapd = rbc_mapd.RemoteMapD()
    client = mapd.make_client()
    try:
        version = client(MapD=dict(get_version=()))['MapD']['get_version']
    except Exception as msg:
        return False, 'failed to get MapD version: %s' % (msg)
    if version >= '4.6':
        return True, None
    return False, 'expected MapD version 4.6 or greater, got %s' % (version)


is_available, reason = mapd_is_available()
pytestmark = pytest.mark.skipif(not is_available, reason=reason)


@pytest.fixture(scope='module')
def mapd():
    m = rbc_mapd.RemoteMapD()
    table_name = 'rbc_test_mapd'
    m.sql_execute('DROP TABLE IF EXISTS {table_name}'.format(**locals()))
    sqltypes = ['FLOAT', 'DOUBLE', 'TINYINT', 'SMALLINT', 'INT', 'BIGINT',
                'BOOLEAN']
    # todo: TEXT ENCODING DICT, TEXT ENCODING NONE, TIMESTAMP, TIME,
    # DATE, DECIMAL/NUMERIC, GEOMETRY: POINT, LINESTRING, POLYGON,
    # MULTIPOLYGON, See
    # https://www.omnisci.com/docs/latest/5_datatypes.html
    colnames = ['f4', 'f8', 'i1', 'i2', 'i4', 'i8', 'b']
    table_defn = ',\n'.join('%s %s' % (n, t)
                            for t, n in zip(sqltypes, colnames))
    m.sql_execute(
        'CREATE TABLE IF NOT EXISTS {table_name} ({table_defn});'
        .format(**locals()))

    def row_value(row, col, colname):
        if colname == 'b':
            return ("'true'" if row % 2 == 0 else "'false'")
        return row

    rows = 5
    for i in range(rows):
        table_row = ', '.join(str(row_value(i, j, n))
                              for j, n in enumerate(colnames))
        m.sql_execute(
            'INSERT INTO {table_name} VALUES ({table_row})'.format(**locals()))
    m.table_name = table_name
    yield m
    m.sql_execute('DROP TABLE IF EXISTS {table_name}'.format(**locals()))


def test_redefine(mapd):
    mapd.reset()

    @mapd('i32(i32)')
    def incr(x):
        return x + 1

    desrc, result = mapd.sql_execute(
        'select i4, incr(i4) from {mapd.table_name}'.format(**locals()))
    for x, x1 in result:
        assert x1 == x + 1

    # This should trigger an UserWarning
    @mapd('i32(i32)')  # noqa: F811
    def incr(x):
        return x + 2

    desrc, result = mapd.sql_execute(
        'select i4, incr(i4) from {mapd.table_name}'.format(**locals()))
    for x, x1 in result:
        assert x1 == x + 2


def test_single_argument_overloading(mapd):
    mapd.reset()

    @mapd(
        'f64(f64)',
        'i64(i64)',
        'i32(i32)',
        'f32(f32)',
        'i32(f32)',
    )
    def mydecr(x):
        return x - 1
    desrc, result = mapd.sql_execute(
        'select f4, mydecr(f4) from {mapd.table_name}'.format(**locals()))
    result = list(result)
    assert len(result) > 0
    for x, x1 in result:
        assert x1 == x - 1
        assert isinstance(x1, type(x))
    desrc, result = mapd.sql_execute(
        'select f8, mydecr(f8) from {mapd.table_name}'.format(**locals()))
    result = list(result)
    assert len(result) > 0
    for x, x1 in result:
        assert x1 == x - 1
        assert isinstance(x1, type(x))
    desrc, result = mapd.sql_execute(
        'select i4, mydecr(i4) from {mapd.table_name}'.format(**locals()))
    result = list(result)
    assert len(result) > 0
    for x, x1 in result:
        assert x1 == x - 1
        assert isinstance(x1, type(x))


def test_thrift_api_doc(mapd):
    mapd.reset()

    @mapd('double(int, double)',
          'float(int, float)',
          'int(int, int)')
    def foo(i, v):
        return v * i + 55

    descr, result = mapd.sql_execute(
        'select f8, foo(i4, f8) from {mapd.table_name}'.format(**locals()))
    result = list(result)
    assert len(result) > 0
    for i, (x, x1) in enumerate(result):
        assert x1 == x * i + 55
        assert isinstance(x1, type(x))


def test_manual_ir(mapd):
    descr, result = mapd.sql_execute(
        'SELECT * FROM {mapd.table_name}'.format(**locals()))
    result = list(result)
    assert result == [(0.0, 0.0, 0, 0, 0, 0, 1), (1.0, 1.0, 1, 1, 1, 1, 0),
                      (2.0, 2.0, 2, 2, 2, 2, 1), (3.0, 3.0, 3, 3, 3, 3, 0),
                      (4.0, 4.0, 4, 4, 4, 4, 1)]
    device_params = mapd.thrift_call('get_device_parameters')
    # print(device_params)
    cpu_target_triple = device_params['cpu_triple']
    cpu_target_datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
    gpu_target_triple = device_params.get('gpu_triple')
    gpu_target_datalayout = ("e-p:64:64:64-i1:8:8-i8:8:8-"
                             "i16:16:16-i32:32:32-i64:64:64-"
                             "f32:32:32-f64:64:64-v16:16:16-"
                             "v32:32:32-v64:64:64-v128:128:128-n16:32:64")

    foo_ir = '''\
define i32 @foobar(i32 %.1, i32 %.2) {
entry:
  %.18.i = mul i32 %.2, %.1
  %.33.i = add i32 %.18.i, 55
  ret i32 %.33.i
}
'''
    ast_signatures = "foobar 'int32(int32, int32)'"
    device_ir_map = dict()
    device_ir_map['cpu'] = '''
target datalayout = "{cpu_target_datalayout}"
target triple = "{cpu_target_triple}"
{foo_ir}
'''.format(**locals())

    if gpu_target_triple is not None:
        device_ir_map['gpu'] = '''
target datalayout = "{gpu_target_datalayout}"
target triple = "{gpu_target_triple}"
{foo_ir}
'''.format(**locals())

    mapd.thrift_call('register_runtime_udf', mapd.session_id,
                     ast_signatures, device_ir_map)
    descr, result = mapd.sql_execute(
        'SELECT i4, foobar(i4, i4) FROM {mapd.table_name}'.format(**locals()))
    result = list(result)
    assert len(result) > 0
    for x, r in result:
        assert r == x * x + 55


def test_ir_parse_error(mapd):
    device_params = mapd.thrift_call('get_device_parameters')
    foo_ir = '''\
define i32 @foobar(i32 %.1, i32 %.2) {
entry:
  %.18.i = mul i32 %.2, %.1
  %.33.i = add i32 %.18.i, 55
  ret i32 %.33.i

'''
    ast_signatures = "foobar 'int32(int32, int32)'"
    device_ir_map = dict()
    device_ir_map['cpu'] = foo_ir

    gpu_target_triple = device_params.get('gpu_triple')
    if gpu_target_triple is not None:
        device_ir_map['gpu_triple'] = foo_ir

    with pytest.raises(Exception, match=r".*LLVM IR ParseError:"):
        mapd.thrift_call('register_runtime_udf', mapd.session_id,
                         ast_signatures, device_ir_map)


@pytest.mark.skip(reason='mapd server crashes')
def test_ir_query_error(mapd):
    device_params = mapd.thrift_call('get_device_parameters')
    gpu_target_triple = device_params.get('gpu_triple')
    foo_ir = '''\
define i32 @foobarrr(i32 %.1, i32 %.2) {
entry:
  %.18.i = mul i32 %.2, %.1
  %.33.i = add i32 %.18.i, 55
  ret i32 %.33.i
}
'''
    ast_signatures = "foobar 'int32(int32, int32)'"
    device_ir_map = dict()
    device_ir_map['cpu'] = foo_ir
    if gpu_target_triple is not None:
        device_ir_map['gpu'] = foo_ir

    mapd.thrift_call('register_runtime_udf', mapd.session_id,
                     ast_signatures, device_ir_map)
    descr, result = mapd.sql_execute(
        'SELECT i4, foobar(i4, i4) FROM {mapd.table_name}'.format(**locals()))


def test_multiple_implementation(mapd):
    mapd.reset()

    @mapd('int(f64)', 'int(i64)')  # noqa: F811
    def bits(x):
        return 64

    @mapd('int(f32)', 'int(i32)')  # noqa: F811
    def bits(x):
        return 32

    @mapd('int(i16)')  # noqa: F811
    def bits(x):
        return 16

    @mapd('int(i8)')  # noqa: F811
    def bits(x):
        return 8

    descr, result = mapd.sql_execute(
        'select bits(i1), bits(i2), bits(i4), bits(f4), bits(i8), bits(f8)'
        ' from {mapd.table_name} limit 1'.format(**locals()))
    result = list(result)
    assert len(result) == 1
    assert result[0] == (8, 16, 32, 32, 64, 64)


def test_loadtime_udf(mapd):
    # This test requires that mapd server is started with `--udf
    # sample_udf.cpp` option where the .cpp file defines `udf_diff`
    try:
        desrc, result = mapd.sql_execute(
            'select i4, udf_diff2(i4, i4) from {mapd.table_name}'
            .format(**locals()))
    except Exception as msg:
        assert 'No match found for function signature udf_diff' in str(msg)
        return
    result = list(result)
    for i_, (i, d) in enumerate(result):
        assert i_ == i
        assert d == 0


def test_f32(mapd):
    """If UDF name ends with an underscore, expect strange behaviour. For
    instance, defining

      @mapd('f32(f32)', 'f32(f64)')
      def f32_(x): return x+4.5

    the query `select f32_(0.0E0))` fails but not when defining

      @mapd('f32(f64)', 'f32(f32)')
      def f32_(x): return x+4.5

    (notice the order of signatures in mapd decorator argument).
    """

    @mapd('f32(f32)', 'f32(f64)')  # noqa: F811
    def f_32(x): return x+4.5
    descr, result = mapd.sql_execute(
        'select f_32(0.0E0) from {mapd.table_name} limit 1'
        .format(**locals()))
    assert list(result)[0] == (4.5,)


def test_castop(mapd):
    @mapd('i16(i16)')  # noqa: F811
    def i32(x): return x+2

    @mapd('i32(i32)')  # noqa: F811
    def i32(x): return x+4

    descr, result = mapd.sql_execute(
        'select i32(cast(1 as int))'
        ' from {mapd.table_name} limit 1'.format(**locals()))
    assert list(result)[0] == (5,)


def test_casting(mapd):
    """
    Define UDFs:
      i8(<tinyint>), i16(<smallint>), i32(<int>), i64(<bigint>)
      f32(<float>), f64(<double>)

    The following table defines the behavior of applying these UDFs to
    values with different types:

             | Functions applied to <itype value>
    itype    | i8   | i16  | i32  | i64  | f32  | f64  |
    ---------+------+------+------+------+------+------+
    tinyint  | OK   | OK   | OK   | OK   | FAIL | FAIL |
    smallint | FAIL | OK   | OK   | OK   | FAIL | FAIL |
    int      | FAIL | FAIL | OK   | OK   | FAIL | FAIL |
    bigint   | FAIL | FAIL | FAIL | OK   | FAIL | FAIL |
    float    | FAIL | FAIL | FAIL | FAIL | OK   | OK   |
    double   | FAIL | FAIL | FAIL | FAIL | FAIL | OK   |
    """
    mapd.reset()

    @mapd('i8(i8)')    # noqa: F811
    def i8(x): return x+1

    @mapd('i16(i16)')  # noqa: F811
    def i16(x): return x+2

    @mapd('i32(i32)')  # noqa: F811
    def i32(x): return x+4

    @mapd('i64(i64)')  # noqa: F811
    def i64(x): return x+8

    @mapd('f32(f32)')  # noqa: F811
    def f32(x): return x+4.5

    # cannot create a 8-bit and 16-bit int literals in sql, so, using
    # the following helper functions:
    @mapd('i8(i8)', 'i8(i16)', 'i8(i32)')    # noqa: F811
    def i_8(x): return x+1

    @mapd('i16(i16)', 'i16(i32)')  # noqa: F811
    def i_16(x): return x+2

    # cannot create a 32-bit float literal in sql, so, using a helper
    # function for that:
    @mapd('f32(f32)', 'f32(f64)')  # noqa: F811
    def f_32(x): return x+4.5

    @mapd('f64(f64)')  # noqa: F811
    def f64(x): return x+8.5

    @mapd('i8(i8)')
    def ifoo(x): return x + 1

    @mapd('i16(i16)')  # noqa: F811
    def ifoo(x): return x + 2

    @mapd('i32(i32)')  # noqa: F811
    def ifoo(x): return x + 4

    @mapd('i64(i64)')  # noqa: F811
    def ifoo(x): return x + 8

    @mapd('f32(f32)')
    def ffoo(x): return x + 4.5

    @mapd('f64(f64)')  # noqa: F811
    def ffoo(x): return x + 8.5

    descr, result = mapd.sql_execute(
        'select i_8(0),i_16(0),i32(0),i64(0) from {mapd.table_name} limit 1'
        .format(**locals()))
    assert list(result)[0] == (1, 2, 4, 8)

    descr, result = mapd.sql_execute(
        'select i_8(i1),i16(i2),i32(i4),i64(i8) from {mapd.table_name} limit 1'
        .format(**locals()))
    assert list(result)[0] == (1, 2, 4, 8)

    descr, result = mapd.sql_execute(
        'select ifoo(i_8(0)),ifoo(i_16(0)),ifoo(i32(0)),ifoo(i64(0))'
        ' from {mapd.table_name} limit 1'
        .format(**locals()))
    assert list(result)[0] == (1+1, 2+2, 4+4, 8+8)

    descr, result = mapd.sql_execute(
        'select ifoo(i1),ifoo(i2),ifoo(i4),ifoo(i8)'
        ' from {mapd.table_name} limit 1'
        .format(**locals()))
    assert list(result)[0] == (1, 2, 4, 8)

    descr, result = mapd.sql_execute(
        'select i64(i_8(0)), i64(i_16(0)),i64(i32(0)),i64(i64(0))'
        ' from {mapd.table_name} limit 1'.format(**locals()))
    assert list(result)[0] == (9, 10, 12, 16)

    descr, result = mapd.sql_execute(
        'select i64(i1), i64(i2),i64(i4),i64(i8)'
        ' from {mapd.table_name} limit 1'.format(**locals()))
    assert list(result)[0] == (8, 8, 8, 8)

    descr, result = mapd.sql_execute(
        'select i32(i_8(0)), i32(i_16(0)),i32(i32(0))'
        ' from {mapd.table_name} limit 1'.format(**locals()))
    assert list(result)[0] == (5, 6, 8)

    descr, result = mapd.sql_execute(
        'select i32(i1), i32(i2),i32(i4)'
        ' from {mapd.table_name} limit 1'.format(**locals()))
    assert list(result)[0] == (4, 4, 4)

    descr, result = mapd.sql_execute(
        'select i16(i_8(0)), i16(i_16(0)) from {mapd.table_name} limit 1'
        .format(**locals()))
    assert list(result)[0] == (3, 4)

    descr, result = mapd.sql_execute(
        'select i16(i1), i16(i2) from {mapd.table_name} limit 1'
        .format(**locals()))
    assert list(result)[0] == (2, 2)

    descr, result = mapd.sql_execute(
        'select i8(i_8(0)) from {mapd.table_name} limit 1'.format(**locals()))
    assert list(result)[0] == (2,)

    descr, result = mapd.sql_execute(
        'select i8(i1) from {mapd.table_name} limit 1'.format(**locals()))
    assert list(result)[0] == (1,)

    descr, result = mapd.sql_execute(
        'select f_32(0.0E0), f64(0.0E0) from {mapd.table_name} limit 1'
        .format(**locals()))
    assert list(result)[0] == (4.5, 8.5)

    descr, result = mapd.sql_execute(
        'select f_32(f4), f64(f8) from {mapd.table_name} limit 1'
        .format(**locals()))
    assert list(result)[0] == (4.5, 8.5)

    descr, result = mapd.sql_execute(
        'select f64(f_32(0.0E0)), f64(f64(0.0E0))'
        ' from {mapd.table_name} limit 1'.format(**locals()))
    assert list(result)[0] == (13.0, 17.0)

    descr, result = mapd.sql_execute(
        'select f64(f4), f64(f8)'
        ' from {mapd.table_name} limit 1'.format(**locals()))
    assert list(result)[0] == (8.5, 8.5)

    descr, result = mapd.sql_execute(
        'select f32(f_32(0.0E0)) from {mapd.table_name} limit 1'
        .format(**locals()))
    assert list(result)[0] == (9.0,)

    descr, result = mapd.sql_execute(
        'select f32(f4) from {mapd.table_name} limit 1'
        .format(**locals()))
    assert list(result)[0] == (4.5,)

    descr, result = mapd.sql_execute(
        'select ffoo(f_32(0.0E0)), ffoo(f64(0.0E0))'
        ' from {mapd.table_name} limit 1'.format(**locals()))
    assert list(result)[0] == (9.0, 17.0)

    descr, result = mapd.sql_execute(
        'select ffoo(f4), ffoo(f8)'
        ' from {mapd.table_name} limit 1'.format(**locals()))
    assert list(result)[0] == (4.5, 8.5)

    for v, f, t in [
            ('i64(0)', r'i32', r'BIGINT'),
            ('i64(0)', r'i16', r'BIGINT'),
            ('i64(0)', r'i8', r'BIGINT'),
            ('i8', r'i32', r'BIGINT'),
            ('i8', r'i16', r'BIGINT'),
            ('i8', r'i8', r'BIGINT'),
            ('i32(0)', r'i16', r'INTEGER'),
            ('i32(0)', r'i8', r'INTEGER'),
            ('i4', r'i16', r'INTEGER'),
            ('i4', r'i8', r'INTEGER'),
            ('i_16(0)', r'i8', r'SMALLINT'),
            ('i2', r'i8', r'SMALLINT'),
            ('f64(0)', r'f32', r'DOUBLE'),
            ('f8', r'f32', r'DOUBLE'),
    ]:
        q = ('select '+f+'('+v+') from {mapd.table_name} limit 1'
             .format(**locals()))
        match = r".*Function "+f+r"\("+t+r"\) not supported"
        with pytest.raises(Exception, match=match):
            descr, result = mapd.sql_execute(q)
            print('query: ', q)
            print('expected: ', match)

    for f in [r'f64', r'f32']:
        for at, av, r in [
                (r'BIGINT', 'i64(0)', (16.5,)),
                (r'INTEGER', 'i32(0)', (12.5,)),
                (r'SMALLINT', 'i_16(0)', (10.5,)),
                (r'TINYINT', 'i_8(0)', (9.5,)),
                (r'BIGINT', 'i8', (8.5,)),
                (r'INTEGER', 'i4', (8.5,)),
                (r'SMALLINT', 'i2', (8.5,)),
                (r'TINYINT', 'i1', (8.5,)),
        ]:
            if f == r'f64':  # temporary: allow integers as double arguments
                descr, result = mapd.sql_execute(
                    'select '+f+'('+av+') from {mapd.table_name} limit 1'
                    .format(**locals()))
                assert list(result)[0] == r
                continue
            with pytest.raises(
                    Exception,
                    match=r".*Function "+f+r"\("+at+r"\) not supported"):
                descr, result = mapd.sql_execute(
                    'select '+f+'('+av+') from {mapd.table_name} limit 1'
                    .format(**locals()))

    for f in [r'i64', r'i32', r'i16', r'i8']:
        for at, av in [
                (r'DOUBLE', 'f64(0E0)'),
                (r'FLOAT', 'f_32(0E0)'),
                (r'DOUBLE', 'f8'),
                (r'FLOAT', 'f4'),
        ]:
            with pytest.raises(
                    Exception,
                    match=r".*Function "+f+r"\("+at+r"\) not supported"):
                descr, result = mapd.sql_execute(
                    'select '+f+'('+av+') from {mapd.table_name} limit 1'
                    .format(**locals()))


def test_truncate_issue(mapd):
    mapd.reset()

    @mapd('int(f64)', 'int(i64)')  # noqa: F811
    def bits(x):
        return 64

    @mapd('int(f32)', 'int(i32)')  # noqa: F811
    def bits(x):
        return 32

    @mapd('int(i16)')  # noqa: F811
    def bits(x):
        return 16

    @mapd('int(i8)')  # noqa: F811
    def bits(x):
        return 8

    descr, result = mapd.sql_execute(
        'select bits(truncate(2016, 1)) from {mapd.table_name} limit 1'
        .format(**locals()))
    assert list(result)[0] in [(16,), (32,)]

    descr, result = mapd.sql_execute(
        'select bits(truncate(cast(2016.0 as smallint), 1))'
        ' from {mapd.table_name} limit 1'
        .format(**locals()))
    assert list(result)[0] == (16,)

    descr, result = mapd.sql_execute(
        'select bits(truncate(cast(2016.0 as int), 1))'
        ' from {mapd.table_name} limit 1'
        .format(**locals()))
    assert list(result)[0] == (32,)

    descr, result = mapd.sql_execute(
        'select bits(truncate(cast(2016.0 as bigint), 1))'
        ' from {mapd.table_name} limit 1'
        .format(**locals()))
    assert list(result)[0] == (64,)

    descr, result = mapd.sql_execute(
        'select bits(truncate(cast(2016.0 as float), 1))'
        ' from {mapd.table_name} limit 1'
        .format(**locals()))
    assert list(result)[0] == (32,)

    descr, result = mapd.sql_execute(
        'select bits(truncate(cast(2016.0 as double), 1))'
        ' from {mapd.table_name} limit 1'
        .format(**locals()))
    assert list(result)[0] == (64,)


def test_calcite():
    calcite = rbc_mapd.Calcite()
    r = calcite.thrift_call('getUserDefinedFunctionWhitelist')
    assert r == '[]'
