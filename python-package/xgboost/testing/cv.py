"""Experimental helpers for fused cross-validation training.

This is a thin, private wrapper around the experimental ``XGBoosterCVExtMem`` C API entry,
used by the benchmark and tests only. It is **not** part of the public API; the public
``xgboost.cv`` is not re-routed through it.
"""

import ctypes
import json
from typing import Any, Dict, List, Optional

import numpy as np

from ..core import DMatrix, _check_call, _LIB, c_str


def _fused_cv_extmem(
    Xy: DMatrix,
    nfold: int,
    params: Dict[str, Any],
    num_boost_round: int,
    *,
    metric: Optional[str] = None,
) -> Dict[str, List[float]]:
    """Run fused K-fold cross-validation over a single ``ExtMemQuantileDMatrix``.

    Grows all ``nfold`` folds' trees simultaneously from one matrix, reusing the shared
    quantile cuts and reusing each fetched page across folds. The folds are contiguous,
    unshuffled row blocks; shuffling (if any) is the caller's responsibility.

    Parameters
    ----------
    Xy :
        The single shared matrix over all train + validation rows.
    nfold :
        Number of folds.
    params :
        Booster parameters. Should set ``device`` to a CUDA device; the POC supports
        ``reg:squarederror`` + ``rmse`` only.
    num_boost_round :
        Number of boosting rounds.
    metric :
        Optional metric name; defaults to the objective's default metric.

    Returns
    -------
    A dict shaped like :py:func:`xgboost.cv`'s output, with the per-round
    ``test-<metric>-mean`` and ``test-<metric>-std`` lists.
    """
    config: Dict[str, Any] = {
        "num_folds": int(nfold),
        "num_boost_round": int(num_boost_round),
        "params": dict(params),
    }
    if metric is not None:
        config["metric"] = metric

    ret = ctypes.c_char_p()
    _check_call(
        _LIB.XGBoosterCVExtMem(Xy.handle, c_str(json.dumps(config)), ctypes.byref(ret))
    )
    assert ret.value is not None
    result = json.loads(ret.value.decode())
    name = result["metric"]
    return {
        f"test-{name}-mean": result["test-mean"],
        f"test-{name}-std": result["test-std"],
    }


def check_fused_cv_extmem(  # pylint: disable=too-many-locals
    device: str,
    cache_prefix: str,
    *,
    n_samples: int = 2048,
    n_features: int = 8,
    nfold: int = 4,
    n_batches: int = 4,
    num_boost_round: int = 8,
    max_depth: int = 4,
    max_bin: int = 64,
) -> None:
    """Compare fused CV against an independent per-fold reference.

    The reference trains each fold on a standalone ``ExtMemQuantileDMatrix`` that shares the
    cuts of the full matrix (via ``ref``) and evaluates on a standalone validation matrix
    using the same binned representation, with the **same contiguous, unshuffled** folds as
    the fused path (so the comparison is well-defined; ``xgboost.cv`` shuffles by default).
    """
    from ..core import ExtMemQuantileDMatrix
    from ..training import train
    from .data import IteratorForTest

    is_cuda = device.startswith("cuda")
    if is_cuda:
        import cupy as cp

    rng = np.random.RandomState(1994)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    coef = rng.standard_normal(n_features).astype(np.float32)
    y = (X @ coef + rng.standard_normal(n_samples).astype(np.float32) * 0.1).astype(np.float32)

    def to_dev(arr: np.ndarray) -> Any:
        return cp.asarray(arr) if is_cuda else arr

    def make_qdm(
        Xa: np.ndarray, ya: np.ndarray, tag: str, *, ref: Optional[DMatrix], nb: int
    ) -> ExtMemQuantileDMatrix:
        xs = [to_dev(c) for c in np.array_split(Xa, nb)]
        ys = [to_dev(c) for c in np.array_split(ya, nb)]
        it = IteratorForTest(xs, ys, None, cache=cache_prefix + "-" + tag)
        return ExtMemQuantileDMatrix(it, max_bin=max_bin, ref=ref)

    full = make_qdm(X, y, "full", ref=None, nb=n_batches)

    params: Dict[str, Any] = {
        "device": device,
        "tree_method": "hist",
        "objective": "reg:squarederror",
        "max_depth": max_depth,
        "eta": 0.3,
        "max_bin": max_bin,
        "min_child_weight": 0.0,
        "reg_lambda": 1.0,
        "seed": 1994,
    }

    fused = _fused_cv_extmem(full, nfold, params, num_boost_round)

    # Contiguous unshuffled folds, matching `CVFoldInfo::MakeContiguous`.
    bounds = [n_samples * f // nfold for f in range(nfold + 1)]
    ref_hist: List[List[float]] = []
    for f in range(nfold):
        mask = np.zeros(n_samples, dtype=bool)
        mask[bounds[f] : bounds[f + 1]] = True
        # `d_train` shares the full matrix's cuts; `d_valid` references `d_train` (required by
        # `train`) and therefore inherits the same (full) cuts transitively, so validation
        # binning matches the fused path.
        d_train = make_qdm(X[~mask], y[~mask], f"tr{f}", ref=full, nb=1)
        d_valid = make_qdm(X[mask], y[mask], f"va{f}", ref=d_train, nb=1)
        evals_result: Dict[str, Dict[str, List[float]]] = {}
        train(
            params,
            d_train,
            num_boost_round=num_boost_round,
            evals=[(d_valid, "valid")],
            evals_result=evals_result,
            verbose_eval=False,
        )
        ref_hist.append(evals_result["valid"]["rmse"])

    ref_mean = np.mean(np.asarray(ref_hist), axis=0)
    fused_mean = np.asarray(fused["test-rmse-mean"])
    assert fused_mean.shape == ref_mean.shape == (num_boost_round,)
    # Both paths predict validation through the same binned representation with shared cuts,
    # so the per-round means agree closely.
    np.testing.assert_allclose(fused_mean, ref_mean, rtol=1e-3, atol=1e-3)
    # Sanity: validation error improves over boosting rounds.
    assert fused_mean[-1] < fused_mean[0]
