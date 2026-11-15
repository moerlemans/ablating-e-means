"""All runs of the write-up: SIFT, 200k training vectors, k = 2048.

Every run takes its batch size B and peak step alpha0 from the paper's scaling laws at its
own epoch budget T, as in the paper. Stages (each resumable, results in results/sift_*.csv):
  lloyd    Lloyd, 25 iterations (test/train MSE after every iteration), and Lloyd to
           convergence (<= 100 iterations). Seeds 0-4.
  budgets  one run per configuration (emeans.methods.ALL and NO_REVIVAL) and epoch budget T in {1, 2, 3, 5, 10},
           seeds 0-4: final test MSE, cluster health, churn.
  curves   one 10-epoch run per method, control and no-revival run, seeds 0-4, test MSE after every epoch.
  tune     alpha0 check: T in {1, 3}, alpha0 = law * 2^j for j in {-2, -1.5, ..., 2} (values
           that would reach 1 are skipped), validation MSE, seeds 100-102.
  overlap  at every revival pass of e-means (count revival, E tracked on the side) and of
           e-means + error revival: the clusters that newly fall below each rule's threshold, and
           how the two donor weights rank the other clusters. T in {1, 3}, seeds 0-4.

Every row carries a fingerprint of its full configuration and of the library code; a resumed
stage skips a run only if both its key and its fingerprint are already in the file.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from emeans.data import load_sift
from emeans.device import get_device, to_device
from emeans.laws import alpha0_law, batch_law
from emeans.lloyd import lloyd
from emeans.methods import ALL, CONTROLS, METHODS
from emeans.metrics import dead_clusters, imbalance, mse
from emeans.minibatch import EMeansConfig, dead, emeans
from emeans.results import ResultsCSV

# e-means and centroid EMA+ without any revival: how much do they need it?
NO_REVIVAL = {
    "mini-batch k-means, no revival": {**METHODS["mini-batch k-means"], "revival": "none"},
    "e-means, no revival": {**METHODS["e-means"], "revival": "none"},
    "centroid EMA+, no revival": {**METHODS["centroid EMA+"], "revival": "none"},
}

K = 2048
EPOCHS = [1, 2, 3, 5, 10]
SEEDS = [0, 1, 2, 3, 4]
TUNE_EPOCHS = [1, 3]
TUNE_SEEDS = [100, 101, 102]
TUNE_EXPONENTS = [float(j) for j in np.arange(-2, 2.01, 0.5)]
CURVE_EPOCHS = 10
STAGES = ["lloyd", "budgets", "curves", "tune", "overlap"]

SOURCE = Path(__file__).resolve().parents[1] / "src" / "emeans"
CODE = hashlib.sha1(b"".join(p.read_bytes() for p in sorted(SOURCE.glob("*.py")))).hexdigest()


def fingerprint(**spec):
    """A short hash of everything that defines a run besides its key: the configuration, the
    data split and the library code."""
    spec = {"data": "sift-200k-seed0", "code": CODE, **spec}
    return hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:10]


def configure(seed, **kwargs):
    """An EMeansConfig and the fingerprint of everything in it except the seed."""
    cfg = EMeansConfig(k=K, seed=seed, **kwargs)
    return cfg, fingerprint(**{k: v for k, v in asdict(cfg).items() if k != "seed"})


def evaluate(res, x, xt):
    """fp64 train/test MSE, dead clusters (< 10 points), imbalance, churn and re-death."""
    train, train_idx = mse(x, res.centroids, return_assignment=True)
    ev = res.revival_events
    return {
        "batch_size": res.batch_size,
        "alpha0": res.alpha0,
        "train_mse": train,
        "test_mse": mse(xt, res.centroids),
        "train_mse_before_finalisation": mse(x, res.centroids_before_finalisation),
        "test_mse_before_finalisation": mse(xt, res.centroids_before_finalisation),
        "dead": dead_clusters(train_idx, K),
        "imbalance": imbalance(train_idx, K),
        "revived": len(ev),
        "revived_after_epoch_1": sum(e["revived_at"] > res.batches_per_epoch for e in ev),
        "redeath_rate": sum(e["died_at"] is not None for e in ev) / len(ev) if ev else float("nan"),
    }


def run_lloyd(x, xt):
    per_iter = ResultsCSV("sift_lloyd", ["seed", "config"])
    summary = ResultsCSV("sift_lloyd_summary", ["seed", "config"])
    config = fingerprint(method="lloyd", k=K, iters=25, max_iters=100)
    for seed in SEEDS:
        key = {"seed": seed, "config": config}
        if per_iter.has(key) and summary.has(key):
            continue
        rows = []
        res25 = lloyd(
            x,
            K,
            iters=25,
            seed=seed,
            callback=lambda it, c, rows=rows, key=key: rows.append(
                {**key, "iter": it, "train_mse": mse(x, c), "test_mse": mse(xt, c)}
            ),
        )
        conv = lloyd(x, K, iters=100, seed=seed, until_converged=True)
        if not per_iter.has(key):
            per_iter.append(rows)
        summary.append(
            [
                {
                    **key,
                    "empty_splits_25": res25.empty_splits,
                    "converged": conv.converged,
                    "converged_iters": conv.iters,
                    "converged_empty_splits": conv.empty_splits,
                    "converged_test_mse": mse(xt, conv.centroids),
                }
            ]
        )
        print(f"lloyd seed {seed}: test@25 {rows[-1]['test_mse']:.1f}, converged after {conv.iters}", flush=True)


def run_budgets(x, xt):
    runs = ResultsCSV("sift_budgets", ["method", "epochs", "seed", "config"])
    for name, preset in {**ALL, **NO_REVIVAL}.items():
        for T in EPOCHS:
            for seed in SEEDS:
                cfg, config = configure(seed, epochs=T, **preset)
                key = {"method": name, "epochs": T, "seed": seed, "config": config}
                if not runs.has(key):
                    runs.append([{**key, **evaluate(emeans(x, cfg), x, xt)}])
            print(f"budgets {name}, T={T}", flush=True)


def run_curves(x, xt):
    out = ResultsCSV("sift_curves", ["method", "seed", "config"])
    for name, preset in {**METHODS, **CONTROLS, **NO_REVIVAL}.items():
        for seed in SEEDS:
            cfg, config = configure(seed, epochs=CURVE_EPOCHS, **preset)
            key = {"method": name, "seed": seed, "config": config}
            if out.has(key):
                continue
            rows = []
            emeans(
                x,
                cfg,
                on_epoch=lambda e, c, rows=rows, key=key: rows.append({**key, "epoch": e, "test_mse": mse(xt, c)}),
            )
            out.append(rows)
        print(f"curves {name}", flush=True)


def run_tune(x, xv):
    out = ResultsCSV("sift_tune", ["method", "epochs", "exponent", "seed", "config"])
    n = x.shape[0]
    for T in TUNE_EPOCHS:
        law = alpha0_law(K, batch_law(K, n, T), n, T)
        for name, preset in METHODS.items():
            for j in TUNE_EXPONENTS:
                if law * 2.0**j >= 1:
                    continue
                for seed in TUNE_SEEDS:
                    cfg, config = configure(seed, epochs=T, alpha0=law * 2.0**j, **preset)
                    key = {"method": name, "epochs": T, "exponent": j, "seed": seed, "config": config}
                    if not out.has(key):
                        res = emeans(x, cfg)
                        out.append([{**key, "alpha0": res.alpha0, "val_mse": mse(xv, res.centroids)}])
            print(f"tune T={T} {name}", flush=True)


def run_overlap(x):
    """At every revival pass: the clusters that newly fall below each rule's threshold (so a
    cluster that stays below is counted once), whether the other rule calls them dead too, and
    the rank correlation of the two donor weights over the clusters neither rule calls dead."""
    out = ResultsCSV("sift_overlap", ["method", "epochs", "seed", "config"])
    runs = {
        "e-means": {**METHODS["e-means"], "track_error": True},
        "e-means + error revival": METHODS["e-means + error revival"],
    }
    for name, preset in runs.items():
        for T in TUNE_EPOCHS:
            for seed in SEEDS:
                cfg, config = configure(seed, epochs=T, **preset)
                key = {"method": name, "epochs": T, "seed": seed, "config": config}
                if out.has(key):
                    continue
                rows, before = [], {}

                def compare(batch, state, cfg=cfg, key=key, rows=rows, before=before):
                    by_error, by_count = dead(state.E, cfg.tau), dead(state.N, cfg.tau)
                    new_error = by_error & ~before.get("error", by_error & False)
                    new_count = by_count & ~before.get("count", by_count & False)
                    live = ~(by_error | by_count)
                    rows.append(
                        {
                            **key,
                            "batch": batch,
                            "new_by_error": int(new_error.sum()),
                            "new_by_count": int(new_count.sum()),
                            "new_by_error_also_count": int((new_error & by_count).sum()),
                            "new_by_count_also_error": int((new_count & by_error).sum()),
                            "donor_rank_corr": spearmanr(state.E[live].cpu(), state.N[live].cpu()).statistic,
                        }
                    )
                    before["error"], before["count"] = by_error, by_count

                res = emeans(x, cfg, on_revival=compare)
                out.append([{**r, "batches_per_epoch": res.batches_per_epoch} for r in rows])
            print(f"overlap {name} T={T}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = get_device(args.device)
    data = load_sift()
    x, xv, xt = (to_device(a, device) for a in (data.train, data.val, data.test))
    for T in EPOCHS:
        B = batch_law(K, x.shape[0], T)
        print(f"T={T:2}: B={B}, alpha0={alpha0_law(K, B, x.shape[0], T):.3f}", flush=True)
    runners = {
        "lloyd": lambda: run_lloyd(x, xt),
        "budgets": lambda: run_budgets(x, xt),
        "curves": lambda: run_curves(x, xt),
        "tune": lambda: run_tune(x, xv),
        "overlap": lambda: run_overlap(x),
    }
    for stage in args.stages:
        runners[stage]()


if __name__ == "__main__":
    main()
