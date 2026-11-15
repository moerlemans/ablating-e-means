import pytest
import torch

from emeans.assign import assign, cluster_stats

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def brute_force(x, c):
    d = torch.cdist(x.double(), c.double()) ** 2
    return d.min(1)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "n,k,mem_cap",
    [
        (1000, 37, 2**30),  # single block
        (1000, 37, 4 * 37 * 50),  # chunk over points only, n not divisible
        (1000, 1000, 4 * 256 * 37),  # chunk over centroids too, k not divisible by 37
        (300, 513, 64),  # tiny cap: one centroid per block
    ],
)
def test_assign_matches_brute_force(device, n, k, mem_cap):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, 16, generator=g).to(device)
    c = torch.randn(k, 16, generator=g).to(device)
    idx, dist = assign(x, c, mem_cap=mem_cap)
    ref_d, ref_i = brute_force(x, c)
    # ties / fp32 rounding may pick a different but equally close centroid
    chosen = ((x.double() - c.double()[idx]) ** 2).sum(1)
    assert torch.allclose(chosen, ref_d, rtol=1e-5, atol=1e-5)
    assert (idx == ref_i).float().mean() > 0.999
    assert torch.allclose(dist.double(), ref_d, rtol=1e-4, atol=1e-4)


def test_cluster_stats():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(500, 8, generator=g)
    idx = torch.randint(0, 7, (500,), generator=g)
    counts, sums = cluster_stats(x, idx, 9, mem_cap=8 * 8 * 33)
    for j in range(9):
        m = idx == j
        assert counts[j] == m.sum()
        assert torch.allclose(sums[j], x[m].double().sum(0))
