import xgboost as xgb
import numpy as np
from sklearn.datasets import make_regression
import dask
from dask import dataframe as dd
from dask_ml import datasets as dmd


class IteratorForTest(xgb.core.DataIter):
    def __init__(self, n_samples_per_batch, n_features, n_batches: int):
        self.X = []
        self.y = []
        for i in range(n_batches):
            X, y = make_regression(
                n_samples=n_samples_per_batch, n_features=n_features
            )
            self.X.append(X)
            self.y.append(y)
        self.it = 0
        super().__init__()

    def next(self, input_data):
        if self.it == len(self.X):
            return 0

        input_data(data=self.X[self.it], label=self.y[self.it])
        self.it += 1
        return 1

    def reset(self):
        self.it = 0


def test_exact():
    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(return_X_y=True)
    X = X[:20].copy()
    y = y[:20].copy()
    Xy = xgb.DeviceQuantileDMatrix(X, y)
    from_it = xgb.train({"tree_method": "exact", "nthread": 1}, Xy)

    Xy = xgb.DMatrix(X, y)
    from_dmat = xgb.train({"tree_method": "exact", "nthread": 1}, Xy)
    assert from_it.get_dump() == from_dmat.get_dump()

    it = IteratorForTest(16, 8, 4)
    Xy = xgb.DeviceQuantileDMatrix(it)
    from_it = xgb.train({"tree_method": "exact", "nthread": 1}, Xy)


class DaskIterator(xgb.core.DataIter):
    def __init__(self, X, y):
        self.X_parts = X.to_delayed()

        if isinstance(self.X_parts, np.ndarray):
            self.X_parts = self.X_parts.flatten().tolist()

        self.y_parts = y.to_delayed()
        if isinstance(self.y_parts, np.ndarray):
            self.y_parts = self.y_parts.flatten().tolist()

        self._it = 0
        super().__init__()

    def next(self, input_data):
        if self._it == len(self.X_parts):
            return 0

        self.x = self.X_parts[self._it].compute()
        self.y = self.y_parts[self._it].compute()
        input_data(data=self.x, label=self.y)
        self._it += 1
        return 1

    def reset(self):
        self.x = None
        self.y = None
        self._it = 0


def test_dask():
    dX, dy = dmd.make_regression(10000, 10, chunks=100)
    Xy = xgb.DeviceQuantileDMatrix(DaskIterator(dX, dy))
    from_it = xgb.train({"tree_method": "exact", "nthread": 16}, Xy)
