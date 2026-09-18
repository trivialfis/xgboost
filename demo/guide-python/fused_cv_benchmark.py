# SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Compare fused CV with in-core CV on one GPU.

In-core CV uses fold-local cuts; fused CV uses full-data cuts, so RMSE can differ.
Both methods use identical folds and return host OOF predictions. Data generation
and warm-up are excluded; construction, training, and prediction are timed.

Edit the settings below, then run:

    python demo/guide-python/fused_cv_benchmark.py

The in-core baseline needs one fold's training/validation matrices and training
state to fit on the GPU. Results and training speedup are printed without plots.
"""

import gc
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from time import perf_counter

import cupy as cp
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import _cross_validation as xcv

ROW_COUNTS = [2**22]
N_FEATURES = 512
K_FOLDS = 5
N_ROUNDS = 32
BATCH_ROWS = 2**16
SEED = 2026
CACHE_HOST_RATIO = 1.0  # Keep the fused CV cache on the host.
TREE_PARAMS = {
    "max_depth": 6,
    "max_bin": 256,
    "learning_rate": 0.1,
}

BASELINE_PARAMS = {
    **TREE_PARAMS,
    "device": "cuda",
    "tree_method": "hist",
    "objective": "reg:squarederror",
    "base_score": 0.5,
    "boost_from_average": 0,
    "multi_strategy": "multi_output_tree",
}
METHODS = ["Fused CV", "In-core CV"]


def make_data(n_rows, n_features, seed, *, n_threads=None):
    """Generate independent row chunks, reproducibly across thread counts."""
    X = np.empty((n_rows, n_features), dtype=np.float32)
    y = np.empty(n_rows, dtype=np.float32)
    chunk_rows = 2**16
    n_chunks = (n_rows + chunk_rows - 1) // chunk_rows
    seeds = np.random.SeedSequence(seed).spawn(n_chunks)

    def fill_chunk(i):
        begin = i * chunk_rows
        end = min(begin + chunk_rows, n_rows)
        rng = np.random.default_rng(seeds[i])
        batch = X[begin:end]
        rng.standard_normal(batch.shape, dtype=np.float32, out=batch)
        y[begin:end] = (
            2.0 * batch[:, 0]
            + 0.5 * batch[:, 1] ** 2
            + np.sin(batch[:, 2])
            + (batch[:, 0] > 0) * batch[:, 3]
            + 0.1 * rng.standard_normal(end - begin, dtype=np.float32)
        )

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        list(executor.map(fill_chunk, range(n_chunks)))
    return X, y


def make_fold_ids(n_rows):
    ids = np.arange(n_rows, dtype=np.int32) % K_FOLDS
    np.random.default_rng(SEED + 1).shuffle(ids)
    return ids


class HostBatchIter(xgb.DataIter):
    def __init__(self, X, y, ids, *, fold=None, validation=False):
        super().__init__(cache_prefix=None, on_host=True, release_data=True)
        self.X, self.y, self.ids = X, y, ids
        self.fold, self.validation = fold, validation
        self.position = 0

    def reset(self):
        self.position = 0

    def next(self, input_data):
        while self.position < len(self.y):
            begin = self.position
            end = min(begin + BATCH_ROWS, len(self.y))
            self.position = end
            X, y = self.X[begin:end], self.y[begin:end]
            if self.fold is not None:
                keep = self.ids[begin:end] == self.fold
                if not self.validation:
                    keep = ~keep
                if not keep.any():
                    continue
                X, y = X[keep], y[keep]
            input_data(data=cp.asarray(X), label=cp.asarray(y))
            return True
        return False


def make_matrix(X, y, ids, *, fold=None, validation=False, ref=None):
    iterator = HostBatchIter(X, y, ids, fold=fold, validation=validation)
    kwargs = {"max_bin": TREE_PARAMS["max_bin"], "ref": ref}
    if fold is None:
        return xgb.ExtMemQuantileDMatrix(
            iterator, cache_host_ratio=CACHE_HOST_RATIO, **kwargs
        )
    return xgb.QuantileDMatrix(iterator, **kwargs)


def stamp():
    cp.cuda.runtime.deviceSynchronize()
    return perf_counter()


@contextmanager
def timed(times, phase):
    start = stamp()
    yield
    times[phase] += stamp() - start


def release_unused(gpu_pool):
    gc.collect()
    cp.cuda.runtime.deviceSynchronize()
    gpu_pool.free_all_blocks()


def run_fused(X, y, ids, rounds):
    times = dict.fromkeys(["build_s", "train_s", "predict_s"], 0.0)
    with timed(times, "build_s"):
        data = make_matrix(X, y, ids)
        assignment = xcv.FoldAssignment(ids, k_folds=K_FOLDS)
        models = xcv.FoldModels(data, K_FOLDS)
        predictions = xcv.FoldPredictions()
        models.init_prediction(data, predictions, assignment=assignment)
        gradients = xcv.FoldGpairs()
        updater = xcv.FoldTreeMethod(models, data, params=TREE_PARAMS)
    with timed(times, "train_s"):
        for iteration in range(rounds):
            models.get_gradient(data, iteration, predictions, out=gradients)
            updater.update(models, data, gradients, predictions)
    with timed(times, "predict_s"):
        oof = cp.asnumpy(predictions.get_valid(copy=False)).reshape(-1)
    return times, oof


def run_in_core(X, y, ids, rounds):
    times = dict.fromkeys(["build_s", "train_s", "predict_s"], 0.0)
    with timed(times, "build_s"):
        oof = np.full(len(y), np.nan, dtype=np.float32)
    for fold in range(K_FOLDS):
        with timed(times, "build_s"):
            train = make_matrix(X, y, ids, fold=fold)
            valid = make_matrix(X, y, ids, fold=fold, validation=True, ref=train)
            model = xgb.Booster(BASELINE_PARAMS, cache=[train, valid])
        with timed(times, "train_s"):
            for iteration in range(rounds):
                model.update(train, iteration)
        with timed(times, "predict_s"):
            oof[ids == fold] = model.predict(valid, output_margin=True).reshape(-1)
        # Release this fold before constructing the next one.
        del model, valid, train
    return times, oof


def run_once(method, X, y, ids, rounds):
    start = stamp()
    if method == "Fused CV":
        times, oof = run_fused(X, y, ids, rounds)
    else:
        times, oof = run_in_core(X, y, ids, rounds)
    times["total_s"] = stamp() - start
    times["fit_oof_s"] = times["train_s"] + times["predict_s"]
    assert oof.shape == y.shape and np.isfinite(oof).all()
    # Scoring is common to both methods and is outside the timers.
    squared_error = (oof.astype(np.float64) - y) ** 2
    fold_rmse = [float(np.sqrt(squared_error[ids == k].mean())) for k in range(K_FOLDS)]
    times["mean_fold_rmse"] = float(np.mean(fold_rmse))
    return times


def main():
    gpu_pool = cp.cuda.MemoryAsyncPool(pool_handles="default")
    cp.cuda.set_allocator(gpu_pool.malloc)
    xgb.set_config(use_cuda_async_pool=True, verbosity=0)

    X, y = make_data(max(2048, K_FOLDS * 8), N_FEATURES, SEED)
    ids = make_fold_ids(len(y))
    for method in METHODS:
        release_unused(gpu_pool)
        run_once(method, X, y, ids, rounds=2)
    del X, y, ids

    records = []
    for n_rows in ROW_COUNTS:
        X, y = make_data(n_rows, N_FEATURES, SEED)
        ids = make_fold_ids(n_rows)
        for method in METHODS:
            release_unused(gpu_pool)
            stats = run_once(method, X, y, ids, N_ROUNDS)
            records.append(dict(method=method, n_rows=n_rows, **stats))
            print(f"{n_rows:,} rows | {method} | {stats['total_s']:.3f}s")
        del X, y, ids
    release_unused(gpu_pool)

    results = pd.DataFrame(records)
    summary = results.set_index(["n_rows", "method"]).sort_index()
    print(summary.round(4).to_string())

    paired = results.pivot(index="n_rows", columns="method", values="train_s")
    speedup = paired["In-core CV"] / paired["Fused CV"]
    print("\nTraining speedup (in-core / fused):")
    print(speedup.to_string())


if __name__ == "__main__":
    main()
