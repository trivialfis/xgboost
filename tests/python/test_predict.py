"""Tests for running inplace prediction."""

from concurrent.futures import ThreadPoolExecutor
from typing import List, Type, Union

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from scipy import sparse
from xgboost import testing as tm
from xgboost.testing.data import get_california_housing, np_dtypes, pd_dtypes
from xgboost.testing.predict import run_base_margin_vs_base_score, run_predict_leaf


def run_threaded_predict(X, rows, predict_func):
    results = []
    per_thread = 20
    with ThreadPoolExecutor(max_workers=10) as e:
        for i in range(0, rows, int(rows / per_thread)):
            if hasattr(X, "iloc"):
                predictor = X.iloc[i : i + per_thread, :]
            else:
                predictor = X[i : i + per_thread, ...]
            f = e.submit(predict_func, predictor)
            results.append(f)

    for f in results:
        assert f.result()


@pytest.mark.parametrize("DMatrixT", [xgb.DMatrix, xgb.QuantileDMatrix])
def test_predict_leaf(DMatrixT: Type[xgb.DMatrix]) -> None:
    run_predict_leaf("cpu", DMatrixT)


def test_predict_shape():
    X, y = get_california_housing()
    reg = xgb.XGBRegressor(n_estimators=1)
    reg.fit(X, y)
    predt = reg.get_booster().predict(xgb.DMatrix(X), strict_shape=True)
    assert len(predt.shape) == 2
    assert predt.shape[0] == X.shape[0]
    assert predt.shape[1] == 1

    contrib = reg.get_booster().predict(
        xgb.DMatrix(X), pred_contribs=True, strict_shape=True
    )
    assert len(contrib.shape) == 3
    assert contrib.shape[1] == 1

    contrib = reg.get_booster().predict(
        xgb.DMatrix(X), pred_contribs=True, approx_contribs=True
    )
    assert len(contrib.shape) == 2
    assert contrib.shape[1] == X.shape[1] + 1

    interaction = reg.get_booster().predict(
        xgb.DMatrix(X), pred_interactions=True, approx_contribs=True
    )
    assert len(interaction.shape) == 3
    assert interaction.shape[1] == X.shape[1] + 1
    assert interaction.shape[2] == X.shape[1] + 1

    interaction = reg.get_booster().predict(
        xgb.DMatrix(X), pred_interactions=True, approx_contribs=True, strict_shape=True
    )
    assert len(interaction.shape) == 4
    assert interaction.shape[1] == 1
    assert interaction.shape[2] == X.shape[1] + 1
    assert interaction.shape[3] == X.shape[1] + 1


def _train_for_shape(
    objective: str,
    *,
    n_classes: int = 0,
    multi_strategy: str = "one_output_per_tree",
    n_targets: int = 1,
    num_parallel_tree: int = 1,
) -> "tuple[xgb.Booster, np.ndarray]":
    rng = np.random.RandomState(1994)
    X = rng.randn(64, 7)
    if n_classes:
        y: np.ndarray = rng.randint(0, n_classes, size=X.shape[0])
    elif objective == "binary:logistic":
        y = rng.randint(0, 2, size=X.shape[0])
    elif n_targets > 1:
        y = rng.randn(X.shape[0], n_targets)
    else:
        y = rng.randn(X.shape[0])
    params = {
        "objective": objective,
        "num_parallel_tree": num_parallel_tree,
        "tree_method": "hist",
    }
    if n_classes:
        params["num_class"] = n_classes
    if multi_strategy != "one_output_per_tree":
        params["multi_strategy"] = multi_strategy
    booster = xgb.train(params, xgb.DMatrix(X, label=y), num_boost_round=4)
    return booster, X


@pytest.mark.parametrize(
    "objective,kwargs",
    [
        ("reg:squarederror", {}),
        ("binary:logistic", {}),
        ("multi:softprob", {"n_classes": 3}),
        ("multi:softmax", {"n_classes": 3}),
        ("reg:squarederror", {"multi_strategy": "multi_output_tree", "n_targets": 3}),
        ("reg:squarederror", {"num_parallel_tree": 3}),
    ],
)
def test_predict_shape_api(objective: str, kwargs: dict) -> None:
    """The analytic ``Booster._predict_shape`` must match the shape of a real prediction
    for every supported combination of prediction type, ``strict_shape`` and
    ``iteration_range`` -- including vector-leaf (multi-output tree) models.

    """
    booster, X = _train_for_shape(objective, **kwargs)
    vector_leaf = kwargs.get("multi_strategy") == "multi_output_tree"
    m = xgb.DMatrix(X)

    flag_sets: List[dict] = [
        {},
        {"output_margin": True},
        {"pred_contribs": True},
        {"pred_contribs": True, "approx_contribs": True},
        {"pred_interactions": True},
        {"pred_interactions": True, "approx_contribs": True},
        {"pred_leaf": True},
    ]

    for flags in flag_sets:
        # Vector leaf: skip the combinations that raise at predict-time (approximate
        # contributions and all interactions are not implemented for multi-output trees).
        if vector_leaf and (
            flags.get("approx_contribs") or flags.get("pred_interactions")
        ):
            continue
        for strict_shape in (False, True):
            ranges = [(0, 0)]
            # leaf/contrib/interaction only support an iteration end (begin must be 0).
            if not (
                flags.get("pred_leaf")
                or flags.get("pred_contribs")
                or flags.get("pred_interactions")
            ):
                ranges.append((1, 3))
            for iteration_range in ranges:
                predicted = booster.predict(
                    m,
                    strict_shape=strict_shape,
                    iteration_range=iteration_range,
                    **flags,
                ).shape
                computed = booster._predict_shape(
                    n_samples=X.shape[0],
                    strict_shape=strict_shape,
                    iteration_range=iteration_range,
                    **flags,
                )
                assert predicted == computed, (flags, strict_shape, iteration_range)


def test_base_margin_vs_base_score() -> None:
    run_base_margin_vs_base_score("cpu")


class TestInplacePredict:
    """Tests for running inplace prediction"""

    @classmethod
    def setup_class(cls):
        cls.rows = 1000
        cls.cols = 10

        cls.missing = 11  # set to integer for testing

        cls.rng = np.random.RandomState(1994)

        cls.X = cls.rng.randn(cls.rows, cls.cols)
        missing_idx = [i for i in range(0, cls.cols, 4)]
        cls.X[:, missing_idx] = cls.missing  # set to be missing

        cls.y = cls.rng.randn(cls.rows)

        dtrain = xgb.DMatrix(cls.X, cls.y)
        cls.test = xgb.DMatrix(cls.X[:10, ...], missing=cls.missing)

        cls.num_boost_round = 10
        cls.booster = xgb.train({"tree_method": "hist"}, dtrain, num_boost_round=10)

    def test_predict(self):
        booster = self.booster
        X = self.X
        test = self.test

        predt_from_array = booster.inplace_predict(X[:10, ...], missing=self.missing)
        predt_from_dmatrix = booster.predict(test)

        X_obj = X.copy().astype(object)

        assert X_obj.dtype.hasobject is True
        assert X.dtype.hasobject is False
        np.testing.assert_allclose(
            booster.inplace_predict(X_obj), booster.inplace_predict(X)
        )

        np.testing.assert_allclose(predt_from_dmatrix, predt_from_array)

        predt_from_array = booster.inplace_predict(
            X[:10, ...], iteration_range=(0, 4), missing=self.missing
        )
        predt_from_dmatrix = booster.predict(test, iteration_range=(0, 4))

        np.testing.assert_allclose(predt_from_dmatrix, predt_from_array)

        with pytest.raises(ValueError):
            booster.predict(test, iteration_range=(0, booster.num_boosted_rounds() + 2))

        default = booster.predict(test)

        range_full = booster.predict(test, iteration_range=(0, self.num_boost_round))
        np.testing.assert_allclose(range_full, default)

        range_full = booster.predict(
            test, iteration_range=(0, booster.num_boosted_rounds())
        )
        np.testing.assert_allclose(range_full, default)

        def predict_dense(x):
            inplace_predt = booster.inplace_predict(x)
            d = xgb.DMatrix(x)
            copied_predt = booster.predict(d)
            return np.all(copied_predt == inplace_predt)

        for i in range(10):
            run_threaded_predict(X, self.rows, predict_dense)

        def predict_csr(x):
            inplace_predt = booster.inplace_predict(sparse.csr_matrix(x))
            d = xgb.DMatrix(x)
            copied_predt = booster.predict(d)
            return np.all(copied_predt == inplace_predt)

        for i in range(10):
            run_threaded_predict(X, self.rows, predict_csr)

    @pytest.mark.skipif(**tm.no_pandas())
    def test_predict_pd(self):
        X = self.X
        # construct it in column major style
        df = pd.DataFrame({str(i): X[:, i] for i in range(X.shape[1])})
        booster = self.booster
        df_predt = booster.inplace_predict(df)
        arr_predt = booster.inplace_predict(X)
        dmat_predt = booster.predict(xgb.DMatrix(X))

        X = df.values
        X = np.asfortranarray(X)
        fort_predt = booster.inplace_predict(X)

        np.testing.assert_allclose(dmat_predt, arr_predt)
        np.testing.assert_allclose(df_predt, arr_predt)
        np.testing.assert_allclose(fort_predt, arr_predt)

    def test_base_margin(self):
        booster = self.booster

        base_margin = self.rng.randn(self.rows)
        from_inplace = booster.inplace_predict(data=self.X, base_margin=base_margin)

        dtrain = xgb.DMatrix(self.X, self.y, base_margin=base_margin)
        from_dmatrix = booster.predict(dtrain)
        np.testing.assert_allclose(from_dmatrix, from_inplace)

    @pytest.mark.skipif(**tm.no_pandas())
    def test_dtypes(self) -> None:
        for orig, x in np_dtypes(self.rows, self.cols):
            predt_orig = self.booster.inplace_predict(orig)
            predt = self.booster.inplace_predict(x)
            np.testing.assert_allclose(predt, predt_orig)

        # unsupported types
        for dtype in [
            np.bytes_,
            np.complex64,
            np.complex128,
        ]:
            X: np.ndarray = np.array(orig, dtype=dtype)
            with pytest.raises(ValueError):
                self.booster.inplace_predict(X)

    @pytest.mark.skipif(**tm.no_pandas())
    def test_pd_dtypes(self) -> None:
        import pandas as pd
        from pandas.api.types import is_bool_dtype

        for orig, x in pd_dtypes():
            dtypes: Union[List, pd.Series] = (
                orig.dtypes if isinstance(orig, pd.DataFrame) else [orig.dtypes]
            )
            if isinstance(orig, pd.DataFrame) and is_bool_dtype(dtypes.iloc[0]):
                continue
            y = np.arange(x.shape[0])
            Xy = xgb.DMatrix(orig, y)
            booster = xgb.train({"tree_method": "hist"}, Xy, num_boost_round=1)
            predt_orig = booster.inplace_predict(orig)
            predt = booster.inplace_predict(x)
            np.testing.assert_allclose(predt, predt_orig)
