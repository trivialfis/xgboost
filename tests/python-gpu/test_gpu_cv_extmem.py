"""Tests for the experimental fused cross-validation entry point (GPU + external memory)."""

import pytest

from xgboost.testing import no_cupy
from xgboost.testing.cv import check_fused_cv_extmem

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.mark.skipif(**no_cupy())
@pytest.mark.parametrize("nfold,n_batches", [(2, 1), (3, 4), (4, 4), (5, 4)])
def test_fused_cv_extmem(tmp_path, nfold: int, n_batches: int) -> None:
    cache = str(tmp_path / "cache")
    check_fused_cv_extmem(
        "cuda",
        cache,
        n_samples=2048,
        n_features=8,
        nfold=nfold,
        n_batches=n_batches,
        num_boost_round=8,
    )
