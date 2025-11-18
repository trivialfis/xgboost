import ctypes
import json
from typing import Tuple

import cupy as cp
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

import xgboost.testing as tm

from ._data_utils import array_interface_dict, cuda_array_interface_dict
from ._typing import ArrayLike
from .core import (
    _LIB,
    ExtMemQuantileDMatrix,
    QuantileDMatrix,
    _check_call,
    c_str,
    make_jcargs,
)
from .objective import TreeObjective
from .testing.data import IteratorForTest
from .training import train

n_classes = 3


def make_group_clf(n_groups: int):
    n_samples = 4096
    n_features = 16
    rng = np.random.default_rng(2025)
    X, y = make_classification(
        n_samples,
        n_features,
        random_state=2025,
        n_classes=n_classes,
        n_informative=n_features,
        n_redundant=0,
    )
    groups = rng.integers(0, n_groups, size=(n_samples,))
    return X, y, groups


def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax function with x as input vector."""
    e = np.exp(x)
    return e / np.sum(e)


class CeObj:
    def __init__(self, n_classes: int) -> None:
        self.n_classes = n_classes

    def __call__(
        self, predt: np.ndarray, y_true: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        labels = y_true
        n_samples = predt.shape[0]

        grad = np.zeros((n_samples, self.n_classes), dtype=np.float32)
        hess = np.zeros((n_samples, self.n_classes), dtype=np.float32)
        eps = 1e-6
        for r in range(predt.shape[0]):
            target = labels[r]
            p = softmax(predt[r, :])
            for c in range(predt.shape[1]):
                assert target >= 0 or target <= self.n_classes
                g = p[c] - 1.0 if c == target else p[c]
                h = max((2.0 * p[c] * (1.0 - p[c])).item(), eps)
                grad[r, c] = g
                hess[r, c] = h

        return grad, hess


class LsObj0(TreeObjective):
    """Split grad is the same as value grad."""

    def __call__(
        self,
        y_pred: ArrayLike,
        y_true: ArrayLike,  # fixme
    ) -> Tuple[np.ndarray, np.ndarray]:
        grad, hess = tm.ls_obj(y_true, y_pred, None)
        return np.array(grad.get()), np.array(hess)


def _make_aitfs(all_batches: list[list[ArrayLike]]) -> str:
    all_aitfs = []
    for batch in all_batches:
        # assert len(batch) == n_splits

        aitfs = []
        for k, fold in enumerate(batch):
            if hasattr(fold, "__cuda_array_interface__"):
                f_aitf = cuda_array_interface_dict(fold)
            else:
                f_aitf = array_interface_dict(fold)
            aitfs.append(f_aitf)
        all_aitfs.append(aitfs)
    jindices = json.dumps(all_aitfs, indent=2)
    return jindices


def _split_by_groups(X, y, groups, n_batches: int):
    # Use stratified kfold to batch the grouped data.
    # Assuming data are sorted according to groups
    kfold = StratifiedKFold(n_splits=n_batches, shuffle=False)
    X_batches = []
    y_batches = []
    g_batches = []
    for f, (train_idx, test_idx) in enumerate(kfold.split(X, y=groups)):
        X_batches.append(cp.array(X[test_idx]))
        y_batches.append(cp.array(y[test_idx]))
        g_batches.append(groups[test_idx])
    return X_batches, y_batches, g_batches


class _CvFolds:
    def __init__(self, n_splits: int) -> None:
        hdl = ctypes.c_void_p()
        _check_call(
            _LIB.XGCvFoldsCreate(make_jcargs(n_folds=n_splits), ctypes.byref(hdl))
        )
        self._hdl = hdl

    def __del__(self) -> None:
        if hasattr(self, "_hdl"):
            _check_call(_LIB.XGCvFoldsFree(self._hdl))
            del self._hdl


def in_core() -> None:
    n_groups = 5
    n_splits = 5
    X, y, groups = make_group_clf(n_groups)
    n_batches = 17

    X_batches, y_batches, g_batches = _split_by_groups(X, y, groups, n_batches)

    it = IteratorForTest(
        X_batches,
        y_batches,
        None,
        cache=None,
        on_host=True,
    )
    Xy_ref = QuantileDMatrix(it)

    all_tr_batches = []
    all_te_batches = []
    for batch_idx in range(n_batches):
        kfold = StratifiedGroupKFold(n_splits=n_splits, random_state=2025, shuffle=True)
        batch_tr_idx = []
        batch_te_idx = []
        for tr_idx, te_idx in kfold.split(
            X_batches[batch_idx], y_batches[batch_idx].get(), g_batches[batch_idx]
        ):
            batch_tr_idx.append(tr_idx)
            batch_te_idx.append(te_idx)
        assert len(batch_tr_idx) == n_splits
        all_tr_batches.append(batch_tr_idx)
        all_te_batches.append(batch_te_idx)

    Xy_folds = []
    for fold_idx in range(n_splits):
        X_batches_fold = []
        y_batches_fold = []
        for batch_idx in range(n_batches):
            batch_tr_idx = all_tr_batches[batch_idx][fold_idx]
            X_i = X_batches[batch_idx][batch_tr_idx, :]
            y_i = y_batches[batch_idx][batch_tr_idx]
            X_batches_fold.append(X_i)
            y_batches_fold.append(y_i)
        it_fold = IteratorForTest(
            X_batches_fold, y_batches_fold, None, cache=None, on_host=True
        )
        Xy_fold = QuantileDMatrix(it_fold, ref=Xy_ref)
        Xy_folds.append(Xy_fold)

    booster_0 = train(
        {
            "device": "cuda",
            "multi_strategy": "multi_output_tree",
            "base_score": 0.5,
            "num_class": n_classes,
            "objective": "multi:softprob",
            "eta": 1.0,
        },
        dtrain=Xy_folds[0],
        num_boost_round=2,
        evals=[(Xy_folds[0], "Train")],
    )


def cross_validate() -> None:
    n_groups = 5
    n_splits = 5
    X, y, groups = make_group_clf(n_groups)
    n_batches = 17

    X_batches, y_batches, g_batches = _split_by_groups(X, y, groups, n_batches)

    it = IteratorForTest(
        X_batches,
        y_batches,
        None,
        cache="cache",
        on_host=True,
    )
    Xy = ExtMemQuantileDMatrix(it)

    all_tr_batches = []
    all_te_batches = []
    for batch_idx in range(n_batches):
        kfold = StratifiedGroupKFold(n_splits=n_splits, random_state=2025, shuffle=True)
        batch_tr_idx = []
        batch_te_idx = []
        for tr_idx, te_idx in kfold.split(
            X_batches[batch_idx], y_batches[batch_idx].get(), g_batches[batch_idx]
        ):
            batch_tr_idx.append(tr_idx)
            batch_te_idx.append(te_idx)
        assert len(batch_tr_idx) == n_splits
        all_tr_batches.append(batch_tr_idx)
        all_te_batches.append(batch_te_idx)

    jindices = _make_aitfs(all_tr_batches)

    # fobj = LsObj0()
    fobj = CeObj(n_classes)

    folds = _CvFolds(n_splits=n_splits)
    num_boost_round = 1
    for i in range(num_boost_round):
        # Calculate the gradient
        all_grad = []
        all_hess = []
        for batch_idx, batch_tr_idx in enumerate(all_tr_batches):
            batch_grad = []
            batch_hess = []
            assert len(batch_tr_idx) == n_splits

            for k, fold in enumerate(batch_tr_idx):
                y_pred = cp.zeros((fold.shape[0], n_classes), dtype=np.float32)
                # Generate a batch of gradient for each fold
                grad, hess = fobj(y_pred, cp.array(y_batches[batch_idx][fold]))
                batch_grad.append(grad)
                batch_hess.append(hess)

            all_grad.append(batch_grad)
            all_hess.append(batch_hess)

        g_aitfs = _make_aitfs(all_grad)
        h_aitfs = _make_aitfs(all_hess)

        # Update
        _check_call(
            _LIB.XGCvUpdateOneIter(
                folds._hdl, Xy.handle, c_str(jindices), c_str(g_aitfs), c_str(h_aitfs)
            )
        )
