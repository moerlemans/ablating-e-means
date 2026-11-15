"""Chunked nearest-centroid assignment.

Distances use the expansion ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2, computed in
blocks of at most `mem_cap` bytes (or 256 rows of one centroid, if mem_cap is smaller).
Blocks run over points and, when k is large, also over centroids, keeping a running
min / argmin.
"""

import torch

DEFAULT_MEM_CAP = 256 * 2**20  # bytes per distance block
MIN_ROWS = 256  # below this many rows per block we chunk over centroids instead


def _chunk_sizes(n, k, itemsize, mem_cap):
    rows = max(1, mem_cap // (itemsize * k))
    if rows >= MIN_ROWS or rows >= n:
        return min(rows, n), k
    return min(MIN_ROWS, n), max(1, mem_cap // (itemsize * MIN_ROWS))


@torch.no_grad()
def assign(x, c, mem_cap=DEFAULT_MEM_CAP):
    """Nearest centroid for every row of x.

    Args:
        x: (n, d) points, on the same device and dtype as c.
        c: (k, d) centroids.
        mem_cap: maximum size in bytes of one distance block.

    Returns:
        idx: (n,) int64 index of the nearest centroid.
        dist: (n,) squared distance to it (clamped at 0), in the dtype of x.
    """
    n, k = x.shape[0], c.shape[0]
    rows, cols = _chunk_sizes(n, k, x.element_size(), mem_cap)
    c_sq = (c * c).sum(1)
    idx = torch.empty(n, dtype=torch.int64, device=x.device)
    dist = torch.empty(n, dtype=x.dtype, device=x.device)
    for i in range(0, n, rows):
        xb = x[i : i + rows]
        xb_sq = (xb * xb).sum(1, keepdim=True)
        best_d = torch.full((xb.shape[0],), float("inf"), dtype=x.dtype, device=x.device)
        best_i = torch.zeros(xb.shape[0], dtype=torch.int64, device=x.device)
        for j in range(0, k, cols):
            d = torch.addmm(c_sq[j : j + cols].unsqueeze(0), xb, c[j : j + cols].T, alpha=-2.0)
            d_min, d_arg = d.min(1)
            better = d_min < best_d
            best_d = torch.where(better, d_min, best_d)
            best_i = torch.where(better, d_arg + j, best_i)
        idx[i : i + rows] = best_i
        dist[i : i + rows] = (best_d + xb_sq.squeeze(1)).clamp_min_(0)
    return idx, dist


@torch.no_grad()
def cluster_stats(x, idx, k, mem_cap=DEFAULT_MEM_CAP):
    """Per-cluster count n_j and vector sum s_j, in fp64.

    Accumulates in row chunks so the fp64 copy of x never exceeds mem_cap bytes.
    """
    n, d = x.shape
    counts = torch.bincount(idx, minlength=k).double()
    sums = torch.zeros(k, d, dtype=torch.float64, device=x.device)
    step = max(1, mem_cap // (8 * d))
    for i in range(0, n, step):
        sums.index_add_(0, idx[i : i + step], x[i : i + step].double())
    return counts, sums
