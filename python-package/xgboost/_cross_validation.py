import json
from typing import Tuple

import cupy as cp
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedGroupKFold

import xgboost.testing as tm

from ._data_utils import array_interface_dict
from ._typing import ArrayLike
from .core import _LIB, DMatrix, ExtMemQuantileDMatrix, c_str
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
        self, y_pred: ArrayLike, dtrain: DMatrix
    ) -> Tuple[cp.ndarray, cp.ndarray]:
        y_true = dtrain.get_label().reshape(y_pred.shape)
        grad, hess = tm.ls_obj(y_true, y_pred, None)
        return cp.array(grad), cp.array(hess)

    def split_grad(
        self, grad: ArrayLike, hess: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike]:
        return cp.array(grad), cp.array(hess)


def cross_validate() -> None:
    n_groups = 5
    n_splits = 5
    X, y, groups = make_group_clf(n_groups)
    n_batches = 17

    X_batches = [cp.array(x) for x in np.array_split(X, n_batches, axis=0)]

    it = IteratorForTest(
        X_batches,
        np.array_split(y, n_batches, axis=0),
        None,
        cache="cache",
        on_host=True,
    )
    Xy = ExtMemQuantileDMatrix(it)

    kfold = StratifiedGroupKFold(n_splits=n_splits, random_state=2025, shuffle=True)

    all_tr_batches = []  # len == n_splits, each fold has n_batches
    all_te_batches = []
    for f, (train_idx, test_idx) in enumerate(kfold.split(X, y, groups)):
        tr_batches = np.array_split(train_idx, n_batches, axis=0)
        te_batches = np.array_split(test_idx, n_batches, axis=0)
        all_tr_batches.append(tr_batches)
        all_te_batches.append(te_batches)

    # Convert into the batches[folds] layout
    all_tr_batches_zipped = list(zip(*all_tr_batches))

    assert len(all_tr_batches_zipped) == n_batches

    all_aitfs = []
    for batch in all_tr_batches_zipped:
        assert len(batch) == n_splits

        aitfs = []
        for k, fold in enumerate(batch):
            f_aitf = array_interface_dict(fold)
            aitfs.append(f_aitf)
        assert len(aitfs) == n_splits
        all_aitfs.append(aitfs)
    assert len(all_aitfs) == n_batches
    jindices = json.dumps(all_aitfs, indent=2)

    fobj = LsObj0()

    num_boost_round = 1
    for i in range(num_boost_round):
        _LIB.XGBCvUpdateOneIter(Xy.handle, c_str(jindices))
