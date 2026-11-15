import torch
from helpers import blobs

from emeans.lloyd import lloyd, split_empty
from emeans.metrics import mse


def test_mse_non_increasing():
    x, _ = blobs(k=20, n=5000, spread=1.0)
    mses = []
    res = lloyd(x, 20, iters=25, seed=0, callback=lambda it, c: mses.append(mse(x, c)))
    assert res.empty_splits == 0  # a split may legitimately increase the MSE
    for a, b in zip(mses, mses[1:]):
        assert b <= a * (1 + 1e-9)


def test_until_converged_counts_the_iterations_that_moved():
    x, _ = blobs(n=3000)
    seen = []
    res = lloyd(x, 10, iters=100, seed=0, until_converged=True, callback=lambda it, c: seen.append(c.clone()))
    assert res.iters == len(seen) - 1 < 100
    assert not torch.equal(seen[-1], seen[-2])  # the last counted iteration still moved
    again = lloyd(x, 10, iters=res.iters + 5, seed=0).centroids
    assert torch.allclose(again, res.centroids)  # and nothing moves after it


def test_finds_well_separated_centres():
    x, centres = blobs(k=8, d=16, n=8000, spread=8.0)
    # random init often puts two centroids in one true cluster; keep the best of 10 starts
    runs = [lloyd(x, 8, iters=50, seed=s, until_converged=True).centroids for s in range(10)]
    c = min(runs, key=lambda c: mse(x, c))
    # every true centre has a centroid on it (sampling error of a mean ~ sqrt(16 / 1000) = 0.13;
    # the closest two centres are ~30 apart)
    assert torch.cdist(centres, c).min(1).values.max() < 0.5


def test_split_empty_fills_all_and_keeps_others():
    c = torch.arange(12.0).reshape(4, 3) + 1
    counts = torch.tensor([10.0, 0.0, 6.0, 0.0])
    out = split_empty(c.clone(), counts, torch.Generator().manual_seed(0))
    assert torch.unique(out, dim=0).shape[0] == 4  # no duplicate centroids left
    # every row is an untouched centroid or a perturbed copy of a donor (rows 0 or 2)
    for row in out:
        assert any(torch.allclose(row, c[j], rtol=2**-6) for j in (0, 2))


def test_reproducible():
    x, _ = blobs(n=3000)
    assert torch.equal(lloyd(x, 10, iters=5, seed=3).centroids, lloyd(x, 10, iters=5, seed=3).centroids)
