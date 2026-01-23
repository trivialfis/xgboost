from xgboost.testing.parse_tree import (
    run_split_value_histograms,
    run_tree_to_df_categorical,
    run_tree_to_df_vector_leaf,
)


def test_tree_to_df_categorical() -> None:
    run_tree_to_df_categorical("hist", "cuda")


def test_tree_to_df_vector_leaf() -> None:
    run_tree_to_df_vector_leaf("cuda")


def test_split_value_histograms() -> None:
    run_split_value_histograms("hist", "cuda")
