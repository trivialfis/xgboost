"""Tests for parsing trees."""

import numpy as np
import pytest
from sklearn.datasets import make_regression

from ..core import DMatrix, QuantileDMatrix
from ..sklearn import XGBRegressor
from ..training import train
from .data import make_categorical
from .utils import Device


def run_tree_to_df_vector_leaf(device: Device) -> None:
    """Tests trees_to_dataframe with vector leaf (multi-target) models."""
    n_samples = 256
    n_targets = 3
    X, y = make_regression(
        n_samples=n_samples, n_features=8, n_targets=n_targets, random_state=2025
    )
    Xy = QuantileDMatrix(X, y)
    booster = train(
        {
            "multi_strategy": "multi_output_tree",
            "device": device,
            "max_depth": 3,
        },
        Xy,
        num_boost_round=4,
    )

    df = booster.trees_to_dataframe()

    # Check basic structure
    assert "Gain" in df.columns
    assert "Cover" in df.columns
    assert "Feature" in df.columns

    # Verify leaf nodes have vector gains (lists)
    leaf_rows = df[df["Feature"] == "Leaf"]
    assert len(leaf_rows) > 0, "Should have leaf nodes"

    for _, row in leaf_rows.iterrows():
        gain = row["Gain"]
        assert isinstance(gain, list), f"Leaf Gain should be a list, got {type(gain)}"
        assert len(gain) == n_targets, f"Leaf should have {n_targets} values"
        for v in gain:
            assert isinstance(v, float), f"Leaf values should be floats, got {type(v)}"

    # Verify split nodes have scalar gains (floats)
    split_rows = df[df["Feature"] != "Leaf"]
    assert len(split_rows) > 0, "Should have split nodes"

    for _, row in split_rows.iterrows():
        gain = row["Gain"]
        assert isinstance(gain, float), f"Split Gain should be float, got {type(gain)}"
        assert gain >= 0, "Split gain should be non-negative"

    # Verify cover is always scalar
    for _, row in df.iterrows():
        cover = row["Cover"]
        assert isinstance(cover, float), f"Cover should be float, got {type(cover)}"
        assert cover > 0, "Cover should be positive"

    # Check that we have the expected number of trees
    assert df["Tree"].nunique() == 4, "Should have 4 trees"


def run_tree_to_df_categorical(tree_method: str, device: Device) -> None:
    """Tests tree_to_df with categorical features."""
    X, y = make_categorical(100, 10, 31, onehot=False)
    Xy = DMatrix(X, y, enable_categorical=True)
    booster = train(
        {"tree_method": tree_method, "device": device}, Xy, num_boost_round=10
    )
    df = booster.trees_to_dataframe()
    for _, x in df.iterrows():
        if x["Feature"] != "Leaf":
            assert len(x["Category"]) >= 1


def run_split_value_histograms(tree_method: str, device: Device) -> None:
    """Tests split_value_histograms with categorical features."""
    X, y = make_categorical(1000, 10, 13, onehot=False)
    reg = XGBRegressor(tree_method=tree_method, enable_categorical=True, device=device)
    reg.fit(X, y)

    with pytest.raises(ValueError, match="doesn't"):
        reg.get_booster().get_split_value_histogram("3", bins=5)
