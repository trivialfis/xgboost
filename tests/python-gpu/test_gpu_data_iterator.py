import sys
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xgboost as xgb
from hypothesis import given, settings, strategies
from xgboost import testing as tm
from xgboost.testing import no_cupy
from xgboost.testing.data_iter import check_invalid_cat_batches, check_uneven_sizes
from xgboost.testing.updater import (
    check_categorical_missing,
    check_categorical_ohe,
    check_extmem_qdm,
    check_quantile_loss_extmem,
)

sys.path.append("tests/python")
from test_data_iterator import run_data_iterator
from test_data_iterator import test_single_batch as cpu_single_batch

# There are lots of warnings if XGBoost is not running on ATS-enabled systems.
pytestmark = pytest.mark.filterwarnings("ignore")


def test_gpu_single_batch() -> None:
    cpu_single_batch("hist", "cuda")


@pytest.mark.skipif(**no_cupy())
@given(
    strategies.integers(0, 1024),
    strategies.integers(1, 7),
    strategies.integers(0, 8),
    strategies.booleans(),
    strategies.booleans(),
    strategies.booleans(),
)
@settings(deadline=None, max_examples=16, print_blob=True)
def test_gpu_data_iterator(
    n_samples_per_batch: int,
    n_features: int,
    n_batches: int,
    subsample: bool,
    use_cupy: bool,
    on_host: bool,
) -> None:
    run_data_iterator(
        n_samples_per_batch,
        n_features,
        n_batches,
        "hist",
        subsample=subsample,
        device="cuda",
        use_cupy=use_cupy,
        on_host=on_host,
    )


def test_cpu_data_iterator() -> None:
    """Make sure CPU algorithm can handle GPU inputs"""
    run_data_iterator(
        1024,
        2,
        3,
        "approx",
        device="cuda",
        subsample=False,
        use_cupy=True,
        on_host=False,
    )


@given(
    strategies.integers(1, 2048),
    strategies.integers(1, 8),
    strategies.integers(1, 4),
    strategies.integers(2, 16),
    strategies.booleans(),
)
@settings(deadline=None, max_examples=10, print_blob=True)
def test_extmem_qdm(
    n_samples_per_batch: int,
    n_features: int,
    n_batches: int,
    n_bins: int,
    on_host: bool,
) -> None:
    check_extmem_qdm(
        n_samples_per_batch,
        n_features,
        n_batches=n_batches,
        n_bins=n_bins,
        device="cuda",
        on_host=on_host,
        is_cat=False,
    )


@pytest.mark.skipif(**no_cupy())
@pytest.mark.parametrize("on_host", [False, True])
@pytest.mark.parametrize("is_cat", [False, True])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("sampling", ["none", "uniform", "gradient_based"])
@pytest.mark.parametrize(
    "growth",
    [
        pytest.param({"max_depth": 1}, id="stump"),
        pytest.param({"max_depth": 3}, id="depth"),
        pytest.param(
            {"max_depth": 2, "grow_policy": "lossguide"}, id="lossguide-depth"
        ),
        pytest.param(
            {"max_depth": 0, "max_leaves": 5, "grow_policy": "lossguide"},
            id="leaves",
        ),
        pytest.param(
            {"max_depth": 12, "min_child_weight": 128}, id="complete-partitions"
        ),
        pytest.param({"gamma": 1e10}, id="no-split"),
    ],
)
def test_extmem_final_position(
    tmp_path: Path,
    on_host: bool,
    is_cat: bool,
    sparse: bool,
    sampling: str,
    growth: dict[str, Any],
) -> None:
    """Deferred splits preserve cached predictions across uneven external-memory pages."""
    import cupy as cp

    rng = np.random.default_rng(2026)
    X = rng.normal(size=(512, 4)).astype(np.float32)
    if is_cat:
        X[:, 0] = rng.integers(0, 8, size=X.shape[0])
    y = cp.asarray(3 * (X[:, 0] > (3 if is_cat else 0)) + X[:, 1], dtype=cp.float32)
    X[rng.random(X.shape) < 0.2] = np.nan
    if sparse:
        # Keep every row shorter than the feature count to exercise sparse ELLPACK lookup.
        X[::2, 2] = np.nan
        X[1::2, 3] = np.nan
    X = cp.asarray(X)
    feature_types = ["c" if is_cat else "q", "q", "q", "q"]
    batches = list(pairwise([0, 1, 65, 240, 400, X.shape[0]]))

    class Iterator(xgb.DataIter):
        def __init__(self) -> None:
            super().__init__(
                cache_prefix=str(tmp_path / "cache"),
                on_host=on_host,
                min_cache_page_bytes=0,
            )
            self.it = 0

        def next(self, input_data: Any) -> bool:
            if self.it == len(batches):
                return False
            begin, end = batches[self.it]
            input_data(
                data=X[begin:end], label=y[begin:end], feature_types=feature_types
            )
            self.it += 1
            return True

        def reset(self) -> None:
            self.it = 0

    in_core = xgb.QuantileDMatrix(
        X, y, max_bin=32, feature_types=feature_types, enable_categorical=is_cat
    )
    external = xgb.ExtMemQuantileDMatrix(
        Iterator(),
        ref=in_core,
        max_bin=32,
        enable_categorical=is_cat,
        cache_host_ratio=1.0,
    )
    params = {
        "tree_method": "hist",
        "device": "cuda",
        "max_bin": 32,
        "max_cat_to_onehot": 1,
        "subsample": 1.0 if sampling == "none" else 0.5,
        "sampling_method": "uniform" if sampling == "none" else sampling,
        **growth,
    }

    def train(data: xgb.DMatrix) -> xgb.Booster:
        booster = xgb.Booster(params, [data])
        for i in range(3):
            booster.update(data, i)
            # Check finalized leaf IDs against independent feature traversal after each
            # round, before xgb.train would reset the training prediction cache. The two
            # prediction paths accumulate tree values in a different order.
            np.testing.assert_allclose(
                booster.predict(data),
                cp.asnumpy(booster.inplace_predict(X)),
                rtol=1e-6,
                atol=1e-6,
            )
        return booster

    expected = train(in_core)
    actual = train(external)
    assert actual.save_raw(raw_format="json") == expected.save_raw(raw_format="json")


@given(
    strategies.integers(1, 2048),
    strategies.integers(1, 4),
    strategies.integers(2, 16),
    strategies.booleans(),
)
@settings(deadline=None, max_examples=10, print_blob=True)
@pytest.mark.skipif(**tm.no_cudf())
@pytest.mark.skipif(**tm.no_cupy())
def test_categorical_extmem_qdm(
    n_samples_per_batch: int,
    n_batches: int,
    n_bins: int,
    on_host: bool,
) -> None:
    check_extmem_qdm(
        n_samples_per_batch,
        4,
        n_batches=n_batches,
        n_bins=n_bins,
        device="cuda",
        on_host=on_host,
        is_cat=True,
    )


def test_invalid_device_extmem_qdm() -> None:
    it = tm.IteratorForTest(
        *tm.make_batches(16, 4, 2, use_cupy=False), cache="cache", on_host=True
    )
    Xy = xgb.ExtMemQuantileDMatrix(it)
    with pytest.raises(ValueError, match="cannot be used for GPU"):
        xgb.train({"device": "cuda"}, Xy)

    it = tm.IteratorForTest(
        *tm.make_batches(16, 4, 2, use_cupy=True), cache="cache", on_host=True
    )
    Xy = xgb.ExtMemQuantileDMatrix(it)
    with pytest.raises(ValueError, match="cannot be used for CPU"):
        xgb.train({"device": "cpu"}, Xy)


@pytest.mark.skipif(**tm.no_cupy())
@pytest.mark.parametrize(
    "objective,n_targets",
    [("reg:absoluteerror", 1), ("reg:squarederror", 1), ("reg:squarederror", 2)],
)
@pytest.mark.parametrize("cache_host_ratio", [0.0, 0.5, 1.0])
def test_concat_pages(objective: str, n_targets: int, cache_host_ratio: float) -> None:
    """Rebatching uneven inputs preserves trees and predictions exactly."""
    import cupy as cp

    rng = np.random.default_rng(2026)
    X = rng.normal(size=(512, 32)).astype(np.float32)
    y = cp.asarray(rng.normal(size=(512, n_targets)), dtype=cp.float32)
    X[rng.random(X.shape) < 0.5] = np.nan
    X = cp.asarray(X)
    batches = list(pairwise([0, 64, 144, 240, 400, 512]))

    def matrix(min_bytes: int, ref: xgb.DMatrix | None = None) -> xgb.DMatrix:
        it = tm.IteratorForTest(
            [X[b:e] for b, e in batches],
            [y[b:e] for b, e in batches],
            None,
            cache=None,
            min_cache_page_bytes=min_bytes,
            on_host=True,
        )
        return xgb.ExtMemQuantileDMatrix(it, ref=ref, cache_host_ratio=cache_host_ratio)

    original = matrix(0)
    params = {
        "device": "cuda",
        "tree_method": "hist",
        "objective": objective,
        "multi_strategy": "multi_output_tree",
        "max_depth": 3,
        "max_cached_hist_node": 1,
    }
    expected = xgb.train(params, original, num_boost_round=3)
    # Compare partial and full concatenation against the original input batches.
    for min_cache_page_bytes in [8000, np.iinfo(np.int64).max]:
        concatenated = matrix(min_cache_page_bytes, ref=original)
        actual = xgb.train(params, concatenated, num_boost_round=3)
        assert actual.save_raw(raw_format="json") == expected.save_raw(
            raw_format="json"
        )
        np.testing.assert_array_equal(
            actual.predict(concatenated), expected.predict(original)
        )
        cp.testing.assert_array_equal(
            actual.inplace_predict(X), expected.inplace_predict(X)
        )


@given(
    strategies.integers(1, 64),
    strategies.integers(1, 8),
    strategies.integers(1, 4),
)
@settings(deadline=None, max_examples=10, print_blob=True)
def test_quantile_objective(
    n_samples_per_batch: int, n_features: int, n_batches: int
) -> None:
    check_quantile_loss_extmem(
        n_samples_per_batch,
        n_features,
        n_batches,
        "hist",
        "cuda",
    )
    check_quantile_loss_extmem(
        n_samples_per_batch,
        n_features,
        n_batches,
        "approx",
        "cuda",
    )


@pytest.mark.parametrize("tree_method", ["hist", "approx"])
@pytest.mark.skipif(**tm.no_cudf())
@pytest.mark.skipif(**tm.no_cupy())
def test_categorical_missing(tree_method: str) -> None:
    check_categorical_missing(
        1024, 4, 5, device="cuda", tree_method=tree_method, extmem=True
    )


@pytest.mark.parametrize(
    "tree_method,multi_target",
    [("hist", False), ("approx", False), ("hist", True)],
)
@pytest.mark.skipif(**tm.no_cudf())
@pytest.mark.skipif(**tm.no_cupy())
def test_categorical_ohe(tree_method: str, multi_target: bool) -> None:
    check_categorical_ohe(
        rows=1024,
        cols=16,
        rounds=4,
        cats=5,
        device="cuda",
        tree_method=tree_method,
        extmem=True,
        multi_target=multi_target,
    )


@pytest.mark.skipif(**tm.no_cudf())
@pytest.mark.skipif(**tm.no_cupy())
def test_invalid_cat_batches() -> None:
    check_invalid_cat_batches("cuda")


def test_uneven_sizes() -> None:
    check_uneven_sizes("cuda")


@pytest.mark.skipif(**tm.no_cupy())
@pytest.mark.parametrize(
    "min_cache_page_bytes", [0, 2048, np.iinfo(np.int64).max, None]
)
def test_cache_host_ratio(min_cache_page_bytes: int | None) -> None:
    """Host/device cache splits preserve models across page-size settings."""
    batches = tm.make_batches(64, 16, 4, use_cupy=True)
    boosters = []
    for cache_host_ratio in [0.0, 0.5, 1.0, None]:
        it = tm.IteratorForTest(
            *batches,
            cache=None,
            min_cache_page_bytes=min_cache_page_bytes,
            on_host=True,
        )
        Xy = xgb.ExtMemQuantileDMatrix(it, cache_host_ratio=cache_host_ratio)
        booster = xgb.train({"device": "cuda"}, Xy)
        boosters.append(booster.save_raw(raw_format="json"))

    for model in boosters[1:]:
        assert model == boosters[0]
