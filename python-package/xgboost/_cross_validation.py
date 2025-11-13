import json
from typing import Tuple

import cupy as cp
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

import xgboost.testing as tm

from ._data_utils import array_interface_dict, cuda_array_interface_dict
from ._typing import ArrayLike
from .core import _LIB, DMatrix, ExtMemQuantileDMatrix, _check_call, c_str
from .objective import TreeObjective
from .testing.data import IteratorForTest


def make_group_clf(n_groups: int):
    n_samples = 4096
    n_features = 16
    rng = np.random.default_rng(2025)
    X, y = make_classification(
        n_samples,
        n_features,
        random_state=2025,
        n_classes=3,
        n_informative=n_features,
        n_redundant=0,
    )
    groups = rng.integers(0, n_groups, size=(n_samples,))
    return X, y, groups


class LsObj0(TreeObjective):
    """Split grad is the same as value grad."""

    def __call__(
        self,
        y_pred: ArrayLike,
        y_true: ArrayLike,  # fixme
    ) -> Tuple[np.ndarray, np.ndarray]:
        grad, hess = tm.ls_obj(y_true, y_pred, None)
        return np.array(grad.get()), np.array(hess)

    def split_grad(
        self, grad: ArrayLike, hess: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike]:
        return cp.array(grad), cp.array(hess)


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
        # assert len(aitfs) == n_splits
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

    fobj = LsObj0()

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
                y_pred = cp.zeros(fold.shape, dtype=np.float32)
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
            _LIB.XGBCvUpdateOneIter(
                Xy.handle, c_str(jindices), c_str(g_aitfs), c_str(h_aitfs)
            )
        )
