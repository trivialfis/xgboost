import json

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedGroupKFold

from ._data_utils import array_interface_dict
from .core import _LIB, c_str


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


def cross_validate() -> None:
    n_groups = 5
    n_splits = 5
    X, y, groups = make_group_clf(n_groups)

    kfold = StratifiedGroupKFold(n_splits=n_splits, random_state=2025, shuffle=True)

    n_batches = 17

    all_tr_batches = []  # len == n_splits, each fold has n_batches
    all_te_batches = []
    for f, (train_idx, test_idx) in enumerate(kfold.split(X, y, groups)):
        tr_batches = np.array_split(train_idx, n_batches, axis=0)
        te_batches = np.array_split(test_idx, n_batches, axis=0)
        all_tr_batches.append(tr_batches)
        all_te_batches.append(te_batches)

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
    _LIB.XGBCvUpdateOneIter(c_str(jindices))
