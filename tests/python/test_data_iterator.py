import xgboost as xgb
import numpy as np


def test_exact():
    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(return_X_y=True)
    X = X[:20].copy()
    y = y[:20].copy()
    Xy = xgb.DeviceQuantileDMatrix(X, y)
    from_it = xgb.train({"tree_method": "hist", "nthread": 1}, Xy)

    Xy = xgb.DMatrix(X, y)
    from_dmat = xgb.train({"tree_method": "hist", "nthread": 1}, Xy)
    assert from_it.get_dump() == from_dmat.get_dump()
