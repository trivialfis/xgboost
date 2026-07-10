# SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

import pytest
import xgboost as xgb
from pytest import fixture
from xgboost import _cross_validation as xcv
from xgboost import testing as tm

if TYPE_CHECKING:
    import cupy as cp


type XywExtQdm = tuple[cp.ndarray, cp.ndarray, cp.ndarray, xgb.ExtMemQuantileDMatrix]


@fixture(scope="module")
def xyw_extqdm() -> XywExtQdm:
    X, y, w = tm.make_batches(16, 4, 2, use_cupy=True)
    it = tm.IteratorForTest(X, y, w, cache=None, min_cache_page_bytes=0, on_host=True)
    Xy = xgb.ExtMemQuantileDMatrix(it)
    return X, y, w, Xy


@pytest.mark.skipif(**tm.no_cupy())
def test_cv_tree_method(xyw_extqdm: XywExtQdm) -> None:
    k_folds = 3
    _, _, _, Xy = xyw_extqdm
    cv_folds = xcv.FoldModels(data=Xy, k_folds=k_folds)

    tree_method = xcv.FoldTreeMethod(
        Xy,
        params={"max_bin": 16, "learning_rate": 0.2, "max_cached_hist_node": 4},
    )
    assert isinstance(tree_method.handle, ctypes.c_void_p)
    assert tree_method.handle.value is not None

    eta_tree_method = xcv.FoldTreeMethod(Xy, params={"eta": 0.1})
    assert isinstance(eta_tree_method.handle, ctypes.c_void_p)
    assert eta_tree_method.handle.value is not None

    with pytest.raises(xgb.core.XGBoostError, match="tree_method"):
        xcv.FoldTreeMethod(Xy, params={"tree_method": "hist"})

    with pytest.raises(xgb.core.XGBoostError, match="updater"):
        xcv.FoldTreeMethod(Xy, params={"updater": "grow_gpu_hist"})

    fold_info = xcv.FoldInfoBatches(Xy, k_folds=k_folds)
    predts = xcv.FoldPredictions()
    assert cv_folds.init_prediction(Xy, fold_info, out=predts) is predts
    gpairs = xcv.FoldGpairs()
    assert cv_folds.get_gradient(Xy, 0, fold_info, predts, out=gpairs) is gpairs
    eta_tree_method.update(cv_folds, Xy, fold_info, gpairs)
    eta_tree_method.update(cv_folds, Xy, fold_info, gpairs)


@pytest.mark.skipif(**tm.no_cupy())
@pytest.mark.skipif(**tm.no_sklearn())
def test_cv_fold_info_batches(xyw_extqdm: XywExtQdm) -> None:
    import cupy as cp
    from sklearn.model_selection import KFold

    X, y, w, Xy = xyw_extqdm
    k_folds = 3

    folds = xcv.FoldInfoBatches(Xy, k_folds=k_folds)

    assert isinstance(folds.handle, ctypes.c_void_p)
    assert folds.handle.value is not None
    assert folds.k_folds == k_folds

    cv_folds = xcv.FoldModels(data=Xy, k_folds=k_folds)
    predts = xcv.FoldPredictions()
    assert cv_folds.init_prediction(Xy, folds, out=predts) is predts
    gpairs = xcv.FoldGpairs()
    assert cv_folds.get_gradient(Xy, 0, folds, predts, out=gpairs) is gpairs

    assert isinstance(gpairs.handle, ctypes.c_void_p)
    assert gpairs.handle.value is not None
    for k in range(k_folds):
        grad, hess = gpairs.get(k, copy=False)
        assert grad.shape == hess.shape
        assert grad.dtype == hess.dtype
        assert grad.data.ptr + ctypes.sizeof(ctypes.c_float) == hess.data.ptr
        assert grad.strides == hess.strides
        assert grad.strides == (
            2 * ctypes.sizeof(ctypes.c_float),
            2 * ctypes.sizeof(ctypes.c_float),
        )

        expected_labels = []
        expected_weights = []
        for batch_y, batch_w in zip(y, w):
            train_idx, _ = list(KFold(n_splits=k_folds).split(batch_y))[k]
            idx = cp.asarray(train_idx)
            expected_labels.append(batch_y[idx])
            expected_weights.append(batch_w[idx])

        expected_labels = (
            cp.concatenate(expected_labels).astype(cp.float32).reshape(grad.shape)
        )
        expected_weights = (
            cp.concatenate(expected_weights).astype(cp.float32).reshape(hess.shape)
        )
        cp.testing.assert_allclose(grad, (0.5 - expected_labels) * expected_weights)
        cp.testing.assert_allclose(hess, expected_weights)

    assert cv_folds.get_gradient(Xy, 1, folds, predts, out=gpairs) is gpairs
