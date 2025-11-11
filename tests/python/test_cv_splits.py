import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


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


def test_group_kfold() -> None:
    n_groups = 5
    X, y, groups = make_group_clf(n_groups)

    # Shuffle permutes the groups, not individual samples.
    kfold = GroupKFold(n_splits=5, random_state=2025, shuffle=True)

    n_batches = 17
    for f, (train_idx, test_idx) in enumerate(kfold.split(X, y, groups)):
        tr_batches = np.array_split(train_idx, n_batches, axis=0)
        te_batches = np.array_split(test_idx, n_batches, axis=0)
        print(len(tr_batches), len(te_batches))
        print(tr_batches)


def test_stratified_group_kfold() -> None:
    n_groups = 5
    X, y, groups = make_group_clf(n_groups)

    kfold = StratifiedGroupKFold(n_splits=5, random_state=2025, shuffle=True)

    n_batches = 17

    all_tr_batches = []         # len == n_splits, each fold has n_batches
    all_te_batches = []
    for f, (train_idx, test_idx) in enumerate(kfold.split(X, y, groups)):
        tr_batches = np.array_split(train_idx, n_batches, axis=0)
        te_batches = np.array_split(test_idx, n_batches, axis=0)
        print(len(tr_batches), len(te_batches))
        all_tr_batches.append(tr_batches)
        all_te_batches.append(te_batches)


if __name__ == "__main__":
    # test_group_kfold()
    test_stratified_group_kfold()
