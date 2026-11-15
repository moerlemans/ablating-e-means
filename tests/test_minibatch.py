import math

import pytest
import torch
from helpers import blobs

from emeans.assign import assign, cluster_stats
from emeans.laws import alpha0_law, batch_law
from emeans.lloyd import lloyd
from emeans.minibatch import EMeansConfig, RevivalTracker, SumsState, emeans, revive
from emeans.schedule import cosine_alpha


def test_full_batch_alpha_one_equals_lloyd_step():
    x, _ = blobs()
    cfg = EMeansConfig(k=10, epochs=1, batch_size=x.shape[0], alpha0=1.0, floor=1.0, revival="none")
    res = emeans(x, cfg)
    ref = lloyd(x, 10, iters=1, seed=cfg.seed)
    assert res.num_updates == 1 and ref.empty_splits == 0
    assert torch.allclose(res.centroids, ref.centroids, atol=1e-5)


def test_finalisation_is_the_mean_of_the_last_epoch():
    # one batch per epoch: the last epoch assigns every point with the centroids after epoch 1
    x, _ = blobs(n=2000)
    after_epoch = {}
    cfg = EMeansConfig(k=10, epochs=2, batch_size=x.shape[0], alpha0=0.3, revival="none", seed=1)
    res = emeans(x, cfg, on_epoch=lambda e, c: after_epoch.setdefault(e, c.clone()))
    idx, _ = assign(x, after_epoch[1])
    counts, sums = cluster_stats(x, idx, 10)
    assert (counts > 0).all()
    assert torch.allclose(res.centroids, (sums / counts[:, None]).float(), atol=1e-5)


def test_centroid_is_weighted_mean_of_batch_means():
    g = torch.Generator().manual_seed(0)
    k, d, steps = 3, 4, 6
    c0 = torch.randn(k, d, generator=g)
    state = SumsState(c0)
    alphas = [0.9, 0.5, 0.3, 0.7, 0.2, 0.4]
    counts = torch.randint(1, 20, (steps, k), generator=g).double()
    means = torch.randn(steps, k, d, generator=g).double()
    for a, n, m in zip(alphas, counts, means):
        state.update(a, n, n[:, None] * m, torch.zeros(k, dtype=torch.float64))
    # weight of step t is alpha_t * prod_{u > t} (1 - alpha_u), times the batch count
    w = torch.tensor([a * math.prod(1 - b for b in alphas[t + 1 :]) for t, a in enumerate(alphas)]).double()
    num = (w[:, None, None] * counts[:, :, None] * means).sum(0)
    den = (w[:, None] * counts).sum(0)
    init_w = math.prod(1 - a for a in alphas) * 1e-10  # the tiny initial weight on c0
    expected = (num + init_w * c0.double()) / (den + init_w)[:, None]
    assert torch.allclose(state.centroids().double(), expected, atol=1e-6)


def test_split_conserves_donor_mass():
    state = SumsState(torch.randn(4, 6), track_error=True)
    counts = torch.tensor([5.0, 3.0, 0.0, 2.0]).double()
    state.update(1.0, counts, torch.randn(4, 6).double(), torch.rand(4).double() * 10)
    n_before, s_before, e_before = state.N[1].clone(), state.S[1].clone(), state.E[1].clone()
    state.split(donor=1, slot=2)
    assert torch.allclose(state.N[1] + state.N[2], n_before)
    assert torch.allclose(state.S[1] + state.S[2], s_before)
    assert torch.allclose(state.E[1] + state.E[2], e_before)
    assert not torch.allclose(state.S[1] / state.N[1], state.S[2] / state.N[2])  # distinct children


def test_revival_uses_distinct_live_donors_and_tracks_redeath():
    state = SumsState(torch.randn(6, 3))
    counts = torch.tensor([10.0, 0.0, 8.0, 0.0, 12.0, 0.0]).double()
    state.update(1.0, counts, torch.randn(6, 3).double(), torch.ones(6).double())
    before = state.centroids().clone()
    revived = revive(state, state.N, 0.1, torch.Generator().manual_seed(0))
    assert sorted(revived) == [1, 3, 5]
    for donor in (0, 2, 4):  # three victims, three distinct donors: every live centroid split once
        assert torch.allclose(state.N[donor], counts[donor] / 2)
        assert not torch.allclose(state.centroids()[donor], before[donor])
    tracker = RevivalTracker()
    tracker.record(5, [1, 3])
    tracker.record(10, [])
    tracker.record(15, [3])  # slot 3 dies again
    died = {(e["slot"], e["revived_at"]): e["died_at"] for e in tracker.events}
    assert died == {(1, 5): None, (3, 5): 15, (3, 15): None}


def test_revival_with_more_victims_than_donors():
    state = SumsState(torch.randn(4, 3))
    state.update(1.0, torch.tensor([9.0, 0.0, 0.0, 0.0]).double(), torch.randn(4, 3).double(), torch.ones(4).double())
    assert revive(state, state.N, 0.1, torch.Generator().manual_seed(0)) == [1]  # one live donor
    assert revive(state, state.N, 0.1, torch.Generator().manual_seed(0)) == [2, 3]  # now two


def test_revival_every():
    x, _ = blobs()
    init = x[:10].clone()
    init[0] = 1e3  # never gets a point
    res = emeans(x, EMeansConfig(k=10, epochs=1, batch_size=400, alpha0=0.3, revival_every=4), init=init)
    assert res.revival_events[0]["revived_at"] == 4


@pytest.mark.parametrize("update", ["sums", "centroid"])
def test_alpha_one_keeps_empty_clusters_and_revives_them(update):
    x, _ = blobs()
    init = x[:10].clone()
    init[0] = 1e3  # never gets a point
    cfg = EMeansConfig(k=10, epochs=1, batch_size=400, alpha0=1.0, floor=1.0, update=update, revival="none")
    res = emeans(x, cfg, init=init)
    assert torch.isfinite(res.centroids).all() and torch.allclose(res.centroids[0], init[0])
    cfg.revival = "count"
    assert emeans(x, cfg, init=init).revival_events[0]["slot"] == 0


def test_revival_does_not_change_the_batch_order(monkeypatch):
    # every shuffle comes from the seed's own generator, whatever revival draws in between
    import emeans.minibatch as mb

    orders = {}
    real = torch.randperm

    for revival in ("count", "none"):
        perms = []
        monkeypatch.setattr(
            mb.torch,
            "randperm",
            lambda n, generator, perms=perms: perms.append(real(n, generator=generator)) or perms[-1],
        )
        x, _ = blobs()
        init = x[:10].clone()
        init[0] = 1e3  # forces revivals in the "count" run
        emeans(
            x,
            EMeansConfig(k=10, epochs=3, batch_size=400, alpha0=0.3, revival=revival, revival_every=1, seed=3),
            init=init,
        )
        orders[revival] = perms
    assert len(orders["count"]) == 3 and all(torch.equal(a, b) for a, b in zip(orders["count"], orders["none"]))


def test_invalid_config_raises():
    x, _ = blobs()
    for bad in (dict(batch_size=x.shape[0] + 1), dict(batch_size=0), dict(update="nope"), dict(alpha0=1.5)):
        with pytest.raises(ValueError):
            emeans(x, EMeansConfig(k=10, **bad))


def test_schedule_and_laws():
    assert cosine_alpha(0, 100, 0.5) == 0.5
    assert math.isclose(cosine_alpha(50, 100, 0.5), 0.5 * (0.1 + 0.9 * 0.5))
    assert cosine_alpha(99, 100, 0.5) > 0.05
    assert batch_law(2048, 200_000, 3) == 8192
    assert batch_law(2048, 200_000, 10) == 32768
    assert batch_law(65536, 100, 25) == 100  # law gives 128, capped at n
    assert abs(alpha0_law(2048, 8192, 200_000, 3) - 0.2998) < 1e-4
    assert alpha0_law(10, 10**6, 100, 1) == 1.0


def test_no_revival_after_the_last_batch():
    x, _ = blobs()
    init = x[:10].clone()
    init[0] = 1e3
    res = emeans(x, EMeansConfig(k=10, epochs=1, batch_size=400, alpha0=0.3, revival_every=1), init=init)
    assert all(e["revived_at"] < res.num_updates for e in res.revival_events)


def test_cumulative_update_is_the_running_mean():
    state = SumsState(torch.zeros(1, 2), forget=False)
    for n, m in [(2.0, 1.0), (6.0, 5.0)]:
        state.update(0.3, torch.tensor([n]).double(), torch.full((1, 2), n * m).double(), torch.zeros(1).double())
    assert torch.allclose(state.centroids().double(), torch.full((1, 2), (2 * 1 + 6 * 5) / 8).double())
