"""Full-batch k-means (Lloyd) with random init and Faiss-style empty-cluster splits."""

from dataclasses import dataclass

import torch

from .assign import assign, cluster_stats

SPLIT_DELTA = 2.0**-7  # the perturbation of a centroid split; the paper uses the bf16 epsilon (App. H.1)


def init_random(x, k, generator):
    """k distinct data points, sampled uniformly without replacement."""
    perm = torch.randperm(x.shape[0], generator=generator)[:k]
    return x[perm.to(x.device)].clone()


def split_pair(c, delta=SPLIT_DELTA):
    """Split one centroid into two children c*(1 + delta*s), c*(1 - delta*s), with
    s = (+1, -1, +1, ...) (Faiss, and the paper's revival). The perturbation is
    multiplicative, so it depends on where the origin is."""
    s = torch.ones_like(c)
    s[1::2] = -1
    return c * (1 + delta * s), c * (1 - delta * s)


def split_empty(c, counts, generator):
    """Fill every empty cluster by splitting a donor drawn proportionally to size.

    Each donor's centroid is replaced by one child and the empty slot gets the
    other; the donor's count is halved so later draws see the split.
    """
    counts = counts.clone()
    for e in torch.nonzero(counts == 0).flatten().tolist():
        donor = int(torch.multinomial(counts.cpu(), 1, generator=generator))
        c[e], c[donor] = split_pair(c[donor])
        counts[e] = counts[donor] / 2
        counts[donor] -= counts[e]
    return c


@dataclass
class LloydResult:
    centroids: torch.Tensor
    iters: int  # iterations run (with until_converged: those that still changed the centroids)
    empty_splits: int  # empty clusters refilled over the run
    converged: bool  # an assignment stopped changing (only checked with until_converged)


@torch.no_grad()
def lloyd(x, k, iters=25, seed=0, until_converged=False, callback=None):
    """Lloyd's algorithm on x, an (n, d) float32 tensor on the target device.

    `iters` is the number of iterations; with `until_converged` it is the maximum and the
    loop stops as soon as an assignment no longer changes. `callback(iteration, centroids)`
    is called after the initialisation (iteration 0) and after every update.
    """
    gen = torch.Generator().manual_seed(seed)
    c = init_random(x, k, gen)
    if callback:
        callback(0, c)
    prev, empty_splits = None, 0
    for it in range(1, iters + 1):
        idx, _ = assign(x, c)
        if until_converged and prev is not None and torch.equal(idx, prev):
            return LloydResult(c, it - 1, empty_splits, True)
        counts, sums = cluster_stats(x, idx, k)
        alive = counts > 0
        c[alive] = (sums[alive] / counts[alive, None]).float()
        if not alive.all():
            empty_splits += int((~alive).sum())
            c = split_empty(c, counts, gen)
        prev = idx
        if callback:
            callback(it, c)
    return LloydResult(c, iters, empty_splits, False)
