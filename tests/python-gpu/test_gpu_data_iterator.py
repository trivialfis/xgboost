import numpy as np
import xgboost as xgb
import sys
sys.path.append("tests/python")
from test_data_iterator import SingleBatch, IteratorForTest


def test_data_iterator():
    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(return_X_y=True)

    Xy = xgb.DMatrix(SingleBatch(data=X, label=y))
    from_it = xgb.train({"tree_method": "gpu_hist"}, Xy)

    Xy = xgb.DMatrix(X, y)
    from_dmat = xgb.train({"tree_method": "gpu_hist"}, Xy)
    assert from_it.get_dump() == from_dmat.get_dump()

    it = IteratorForTest(16, 8, 4)
    Xy = xgb.DMatrix(it)
    from_it = xgb.train({"tree_method": "gpu_hist", "num_parallel_tree": 4}, Xy)
    it_predt = from_it.predict(Xy)

    Xy = it.as_arrays()
    Xy = xgb.DMatrix(it)
    from_blob = xgb.train({"tree_method": "gpu_hist", "num_parallel_tree": 4}, Xy)
    blob_predt = from_blob.predict(Xy)

    np.testing.assert_allclose(it_predt, blob_predt)
