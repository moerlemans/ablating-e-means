# ablating-e-means

A small ablation of **e-means** (Bessa et al., *e-means: outperforming k-means (in one-tenth the
epochs)*, arXiv [2611.09501](https://arxiv.org/abs/2611.09501)).

e-means is mini-batch k-means that keeps, per cluster, an exponential moving average (EMA) of a
sum and a count, and reads the centroid off as their ratio. We take that apart: what do the sum
and the count do, and can an EMA of the centroid itself do the same? This repository holds the
code and the raw results; the write-up is the blog post
[Exploring e-means, the faster brother of k-means](https://moerlemans.github.io/2026/11/15/exploring-e-means.html).

## Methods

| name in the write-up | update | revival | state per cluster |
|---|---|---|---|
| e-means (the paper) | EMA of sums and counts, c = S/N | by count, every 5 batches | S, N |
| centroid EMA | EMA of the centroid, fixed step | by count, every 5 batches | c, N |
| centroid EMA+ | EMA of the centroid, count-scaled step, warm-up | by count, every 5 batches | c, N |
| centroid EMA++ | as centroid EMA+ | by error, every batch | c, E |
| e-means + error revival | as e-means | by error, every batch | S, N, E |

All share the paper's random initialisation, cosine step schedule, scaling laws for the batch
size and step (at each epoch budget), dead-cluster split, and last-epoch-mean finalisation.
They are the presets in `emeans.methods`, together with the controls (count revival every
batch) and the ablations (centroid EMA with only the step or only the warm-up).

## Install

The environment is managed with [uv](https://docs.astral.sh/uv/) and pinned in `uv.lock`
(Python 3.12, PyTorch 2.5.1 built for CUDA 12.1, which is a ~2.5 GB download even if you only
use the CPU; without a GPU everything runs on the CPU, more slowly).

1. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Create the environment: `uv sync`
3. Run the tests: `uv run pytest`

On WSL with the repository on a Windows drive, use `UV_LINK_MODE=copy uv sync`.

## Reproduce

The results of the write-up are committed in `results/`, so step 2 works on its own.

1. Delete `results/sift_*.csv`, then run `uv run python experiments/sift.py`. It downloads
   SIFT-128 from ann-benchmarks (~500 MB, into `data/`) and runs everything for 5 seeds:
   Lloyd, every configuration at epoch budgets 1, 2, 3, 5 and 10, the 10-epoch training curves,
   a step-size check and the revival overlap. About 40 minutes on a 4 GB laptop GPU. It skips
   runs that are already in `results/` with the same configuration, so it can be interrupted.
2. `uv run python plots/blog.py` makes the figures of the write-up and a supplement with all
   numbers behind it, in `blog/` (not part of the repository).

Runs are bit-reproducible on the CPU. On a GPU, per-cluster sums use atomic additions, so a
rerun can flip an assignment now and then and end up somewhere slightly different, at the level
of the seed-to-seed spread.

## Layout

```
src/emeans/
  minibatch.py  the e-means loop: update rules (sums / centroid with step and warm-up),
                revival rules (count / error)
  lloyd.py      full-batch k-means (the reference) and the centroid split used by revival
  assign.py     chunked nearest-centroid assignment, per-cluster counts and sums
  laws.py       the paper's scaling laws for batch size and step
  schedule.py   cosine step schedule
  data.py       SIFT-128 loader
  metrics.py    fp64 MSE, dead clusters, imbalance
  methods.py    the named configurations of the write-up
  results.py    resumable CSV results
  device.py     device helpers
experiments/sift.py   all runs of the write-up
plots/blog.py         figures and supplement of the write-up
results/              the results of those runs
tests/                pytest suite
```

## About this repository

Most of the code, experiments and analysis in this repository were written together with an AI
coding assistant (Claude Code), with the author directing the work and reviewing the results.
All outputs are my own: I have checked them and I stand behind them.
