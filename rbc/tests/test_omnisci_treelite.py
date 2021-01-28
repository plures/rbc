import pytest
from rbc.tests import omnisci_fixture
from rbc.ctools import compile_ccode


@pytest.fixture(scope='module')
def omnisci():
    for o in omnisci_fixture(globals(), minimal_version=(5, 5), debug=not True):
        yield o


def test_boston_house_prices(omnisci):
    import numpy as np
    import tempfile
    try:
        import treelite
    except ImportError as msg:
        pytest.skip(f'test requires treelite: {msg}')
    try:
        import xgboost
    except ImportError as msg:
        pytest.skip(f'test requires xgboost: {msg}')

    # Upload Boston house prices to server, notice that we collect all
    # row values expect the last one (MEDV) to a FLOAT array.
    import os
    csv_file = os.path.join(os.path.dirname(__file__), 'boston_house_prices.csv')
    data = []
    medv = []
    for i, line in enumerate(open(csv_file).readlines()):
        line = line.strip().replace(' ', '')
        if i == 0:
            header = line.split(',')
        else:
            row = list(map(float, line.split(',')))
            assert len(row) == len(header)
            data.append(row[:-1])
            medv.append(row[-1])
    table_name = f'{omnisci.table_name}bhp'
    omnisci.sql_execute(f'DROP TABLE IF EXISTS {table_name}')
    omnisci.sql_execute(f'CREATE TABLE IF NOT EXISTS {table_name} (data FLOAT[], medv FLOAT);')
    omnisci.load_table_columnar(table_name, data=data, medv=medv)

    # Get training data from server:
    descr, result = omnisci.sql_execute(f'SELECT data, medv FROM {table_name} LIMIT 50')
    data, medv = map(np.array, zip(*list(result)))
    assert len(medv) == 50

    # Train model using xgboost
    dtrain = xgboost.DMatrix(data, label=medv)
    params = {'max_depth': 3,
              'eta': 1,
              'objective': 'reg:squarederror',
              'eval_metric': 'rmse'}
    bst = xgboost.train(params, dtrain, len(medv), [(dtrain, 'train')])

    # Compile model to C
    working_dir = tempfile.mkdtemp()
    model = treelite.Model.from_xgboost(bst)
    model.compile(working_dir)
    model_c = open(os.path.join(working_dir, 'main.c')).read()
    # The C model implements
    #  float predict(union Entry* data, int pred_margin)
    # but we wrap it using
    model_c += '''
    float predict_float(float* data, int pred_margin) {
      return predict((union Entry*)data, pred_margin);
    }
    '''
    # to make UDF construction easier.

    # Helper function that makes the wrapper function intrinsic. This
    # is temporary solution that will be hidden from an user.
    from numba.core import cgutils, extending
    from numba.core import types as nb_types
    from llvmlite import ir
    from rbc.typesystem import Type

    @extending.intrinsic
    def predict_float(typingctx, data, pred_margin):
        sig = nb_types.float32(Type.fromstring('float32[]').tonumba(), nb_types.int32)

        def codegen(context, builder, signature, args):
            data, pred_margin = args
            fp32 = ir.FloatType()
            int32_t = ir.IntType(32)
            wrap_fnty = ir.FunctionType(fp32, [fp32.as_pointer(), int32_t])
            wrap_fn = builder.module.get_or_insert_function(wrap_fnty, name="predict_float")
            rawptr = cgutils.alloca_once_value(builder, value=data)
            buf = builder.load(builder.gep(rawptr, [int32_t(0)]))
            ptr = builder.load(builder.gep(buf, [int32_t(0), int32_t(0)]))
            r = builder.call(wrap_fn, [ptr, pred_margin])
            return r

        return sig, codegen

    # Define predict function as UDF. Notice that the xgboost null
    # values are different from omniscidb null values, so we'll remap
    # before calling predict_float:
    null_value = np.int32(-1).view(np.float32)

    @omnisci('float(float[], int32)', devices=['cpu'])
    def predict(data, pred_margin):
        for i in range(len(data)):
            if data.is_null(i):
                data[i] = null_value
        return predict_float(data, pred_margin)

    # Compile C model to LLVM IR. In future, we might want this
    # compilation to happen in the server side as the client might not
    # have clang compiler installed.
    model_llvmir = compile_ccode(model_c, include_dirs=[working_dir])

    # RBC will link_in the LLVM IR module
    omnisci.user_defined_llvm_ir['cpu'] = model_llvmir

    # Call predict on data in the server:
    descr, result = omnisci.sql_execute(f'SELECT predict(data, 2) FROM {table_name}')
    predict_medv = np.array(list(result))[:, 0]

    # predict on the first 50 elements should be close to training labels
    error = abs(predict_medv[:len(medv)] - medv).max()/len(medv)
    assert error < 1e-4, error

    # Clean up
    omnisci.sql_execute(f'DROP TABLE IF EXISTS {table_name}')
