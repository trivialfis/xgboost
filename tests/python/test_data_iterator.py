import xgboost as xgb
import numpy as np
from sklearn.datasets import make_regression


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
    print("x.shape:", X.shape)
    Xy = xgb.DeviceQuantileDMatrix(X, y)
    from_it = xgb.train({"tree_method": "exact", "nthread": 1}, Xy)

    Xy = xgb.DMatrix(X, y)
    from_dmat = xgb.train({"tree_method": "exact", "nthread": 1}, Xy)
    assert from_it.get_dump() == from_dmat.get_dump()

    it = IteratorForTest(16, 8, 4)
    Xy = xgb.DeviceQuantileDMatrix(it)
    from_it = xgb.train({"tree_method": "exact", "nthread": 1}, Xy)
    assert from_it.get_dump() == from_dmat.get_dump()
