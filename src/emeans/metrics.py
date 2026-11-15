"""Evaluation metrics. All reported MSEs are computed in fp64."""

import torch

from .assign import DEFAULT_MEM_CAP, assign

DEAD_THRESHOLD = 10  # a cluster with fewer assigned points is counted as dead (paper, App. H.2)


@torch.no_grad()
def mse(x, c, return_assignment=False):
    """MSE(X, C) = (1/n) sum_i ||x_i - c_a(i)||^2, accumulated in fp64.

    Assignment is done in the dtype of x (fp32 during training); the squared
    distances of the chosen pairs are then recomputed exactly in fp64.
    """
    idx, _ = assign(x, c)
    c64 = c.double()
    total = 0.0
    step = max(1, DEFAULT_MEM_CAP // (8 * x.shape[1]))
    for i in range(0, x.shape[0], step):
        diff = x[i : i + step].double() - c64[idx[i : i + step]]
        total += (diff * diff).sum().item()
    value = total / x.shape[0]
    return (value, idx) if return_assignment else value


def dead_clusters(idx, k, threshold=DEAD_THRESHOLD):
    """Clusters with fewer than `threshold` assigned points."""
    return int((torch.bincount(idx, minlength=k) < threshold).sum())


def imbalance(idx, k):
    """k * sum_j p_j^2; 1 for perfectly balanced clusters (Faiss definition)."""
    p = torch.bincount(idx, minlength=k).double() / idx.numel()
    return float(k * (p * p).sum())
