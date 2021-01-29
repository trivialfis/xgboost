import xgboost as xgb
import numpy as np


class IteratorForTest(xgb.core.DataIter):
    def __init__(self, n_samples_per_batch, n_features, n_batches: int):
        self.X = []
        self.y = []
        rng = np.random.RandomState(1994)
        for i in range(n_batches):
            X = rng.randn(n_samples_per_batch, n_features)
            y = rng.randn(n_samples_per_batch)
            self.X.append(X)
            self.y.append(y)
        self.it = 0
        super().__init__("./")

    def next(self, input_data):
        if self.it == len(self.X):
            return 0

        input_data(data=self.X[self.it], label=self.y[self.it])
        self.it += 1
        return 1

    def reset(self):
        self.it = 0

    def as_arrays(self):
        X = np.concatenate(self.X, axis=0)
        y = np.concatenate(self.y, axis=0)
        return X, y


class SingleBatch(xgb.core.DataIter):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.it = 0             # pylint: disable=invalid-name
        super().__init__("./")

    def next(self, input_data):
        if self.it == 1:
            return 0
        self.it += 1
        input_data(**self.kwargs)
        return 1

    def reset(self):
        self.it = 0


def test_data_iterator():
    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(return_X_y=True)

    Xy = xgb.DMatrix(SingleBatch(data=X, label=y))
    from_it = xgb.train({"tree_method": "approx"}, Xy)

    Xy = xgb.DMatrix(X, y)
    from_dmat = xgb.train({"tree_method": "approx"}, Xy)
    assert from_it.get_dump() == from_dmat.get_dump()

    n_rounds = 2

    it = IteratorForTest(16, 8, 4)
    Xy = xgb.DMatrix(it)
    assert Xy.num_row() == 16 * 4
    assert Xy.num_col() == 8

    from_it = xgb.train(
        {"tree_method": "hist", "max_depth": 2}, Xy, num_boost_round=n_rounds
    )
    it_predt = from_it.predict(Xy)

    X, y = it.as_arrays()
    Xy = xgb.DMatrix(X, y)
    assert Xy.num_row() == 16 * 4
    assert Xy.num_col() == 8

    from_arrays = xgb.train(
        {"tree_method": "hist", "max_depth": 2}, Xy, num_boost_round=n_rounds
    )
    arr_predt = from_arrays.predict(Xy)
    np.testing.assert_allclose(it_predt, arr_predt)
