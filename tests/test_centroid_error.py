import pytest
import torch
from helpers import blobs

from emeans.methods import ALL, METHODS
from emeans.minibatch import CentroidState, EMeansConfig, SumsState, emeans, revive


def batch(counts, means):
    counts = torch.tensor(counts).double()
    return counts, counts[:, None] * torch.tensor(means).double()


@pytest.mark.parametrize("step", ["scaled", "fixed"])
def test_centroid_step_sizes(step):
    state = CentroidState(torch.zeros(3, 2), step, warmup=False, ref=4.0, revival="count")
    counts, sums = batch([4.0, 0.0, 1.0], [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    state.update(0.5, counts, sums, torch.zeros(3).double())
    assert torch.allclose(state.C[0], torch.full((2,), 0.5).double())  # average count: step alpha
    assert torch.allclose(state.C[1], torch.zeros(2).double())  # no points: no move
    expected = 0.5 if step == "fixed" else 1 - 0.5**0.25  # few points: small step if count-scaled
    assert torch.allclose(state.C[2], torch.full((2,), expected).double())


def test_warmup_is_the_bias_correction_of_a_zero_initialised_ema():
    state = CentroidState(torch.zeros(1, 1), "fixed", warmup=True, ref=1.0, revival="count")
    counts, sums = batch([1.0], [[1.0]])
    state.update(0.2, counts, sums, torch.zeros(1).double())
    assert torch.allclose(state.C, torch.ones(1, 1).double())  # first step: a full jump
    counts, sums = batch([1.0], [[0.0]])
    state.update(0.2, counts, sums, torch.zeros(1).double())
    a2 = 0.2 / (1 - 0.8 * 0.8)
    assert torch.allclose(state.C, torch.full((1, 1), 1 - a2).double())
    # the same as e-means for a cluster with a steady count
    sums_state = SumsState(torch.zeros(1, 1))
    for m in (1.0, 0.0):
        sums_state.update(0.2, torch.ones(1).double(), torch.full((1, 1), m).double(), torch.zeros(1).double())
    assert torch.allclose(sums_state.centroids().double(), state.C, atol=1e-8)


def test_scaled_step_equals_fixed_step_for_average_clusters():
    g = torch.Generator().manual_seed(0)
    c0 = torch.randn(5, 3, generator=g)
    a = CentroidState(c0, "scaled", warmup=True, ref=7.0, revival="count")
    b = CentroidState(c0, "fixed", warmup=True, ref=7.0, revival="count")
    for alpha in [0.4, 0.2, 0.1]:
        counts = torch.full((5,), 7.0).double()  # every cluster gets exactly B / k points
        sums = counts[:, None] * torch.randn(5, 3, generator=g).double()
        for s in (a, b):
            s.update(alpha, counts, sums, torch.zeros(5).double())
    assert torch.allclose(a.C, b.C)


def test_reference_count_is_batch_size_over_k():
    x, _ = blobs()
    refs = []
    cfg = EMeansConfig(k=10, epochs=1, batch_size=400, **METHODS["centroid EMA+"])
    emeans(x, cfg, on_revival=lambda b, s: refs.append(s.ref))
    assert refs and refs[0] == 400 / 10


def test_centroid_split_conserves_weight():
    state = CentroidState(torch.randn(3, 4), "scaled", warmup=True, ref=1.0, revival="error")
    counts = torch.tensor([3.0, 2.0, 0.0]).double()
    state.update(1.0, counts, torch.randn(3, 4).double(), torch.tensor([4.0, 6.0, 0.0]).double())
    e_before, c_before = state.E[1].clone(), state.C[1].clone()
    state.split(donor=1, slot=2)
    assert torch.allclose(state.E[1] + state.E[2], e_before)
    assert torch.allclose((state.C[1] + state.C[2]) / 2, c_before)  # children straddle the donor


@pytest.mark.parametrize(
    "make",
    [
        lambda c: SumsState(c, track_error=True),
        lambda c: CentroidState(c, "scaled", warmup=True, ref=1.0, revival="error"),
    ],
)
def test_error_revival_revives_zero_distortion_cluster(make):
    state = make(torch.randn(4, 3))
    counts, sums = batch([5.0, 5.0, 5.0, 5.0], [[0.0] * 3] * 4)
    state.update(1.0, counts, sums, torch.tensor([3.0, 0.0, 2.0, 4.0]).double())
    revived = revive(state, state.E, 0.1, torch.Generator().manual_seed(0))
    assert revived == [1]  # same count as the others, but contributes nothing to the MSE
    assert state.E[1] > 0


@pytest.mark.parametrize("name", list(ALL))
def test_every_configuration_revives_a_planted_dead_cluster(name):
    x, _ = blobs()
    init = x[:10].clone()
    init[0] = 1e3  # far from all data: never gets a point
    res = emeans(x, EMeansConfig(k=10, epochs=2, batch_size=256, **ALL[name]), init=init)
    assert res.revival_events and res.revival_events[0]["slot"] == 0
    assert res.centroids.abs().max() < 100  # the far centroid was moved into the data


@pytest.mark.parametrize("name", list(ALL))
def test_every_configuration_reproducible(name):
    x, _ = blobs()
    cfg = EMeansConfig(k=10, epochs=1.5, batch_size=256, seed=2, **ALL[name])
    assert torch.equal(emeans(x, cfg).centroids, emeans(x, cfg).centroids)


@pytest.mark.parametrize("alpha", [0.1, 0.18129, 0.3, 0.7, 0.914443])
def test_warmed_up_scaled_step_is_finite_on_the_first_update(alpha):
    state = CentroidState(torch.zeros(2, 1), "scaled", warmup=True, ref=3.0, revival="count")
    counts, sums = batch([2.0, 0.0], [[1.0], [5.0]])
    state.update(alpha, counts, sums, torch.zeros(2).double())
    assert torch.isfinite(state.C).all()
    assert torch.allclose(state.C, torch.tensor([[1.0], [0.0]]).double())  # full jump; empty stays
