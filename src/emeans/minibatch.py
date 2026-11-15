"""Mini-batch k-means with EMA updates (e-means, paper Algorithm 1), with switchable parts.

Per batch: read off centroids, assign, compute per-cluster counts n_j, sums s_j and
distortions e_j = sum ||x - c_j||^2, apply the update rule; every `revival_every`
batches apply the revival rule. After training, every centroid is replaced by the mean of
the points assigned to it in the last epoch (the paper's last-epoch-mean finalisation).

Update rules (`update`):
1. "sums": the paper. EMA of count N_j and sum S_j; c_j = S_j / N_j. Written as a centroid
   step this is c_j <- c_j + a_j (m_j - c_j), m_j the batch mean, with
   a_j = alpha n_j / (gamma N_j + alpha n_j): a step that grows with the batch count, and,
   because N_j starts at 1e-10, a warm-up: for steady counts a_j = alpha / (1 - prod gamma),
   the bias correction of a zero-initialised EMA, which is 1 on the first update.
2. "cumulative": classic mini-batch k-means (Sculley, 2010): a running sum and count without
   forgetting, N_j <- N_j + n_j, S_j <- S_j + s_j, c_j = S_j / N_j, i.e. a step of
   n_j / (total count so far). The schedule does not enter.
3. "centroid": EMA of the centroid itself, c_j <- c_j + a_j (m_j - c_j), for every cluster
   that received points, with two independent switches:
   - step="fixed":  a_j = alpha;  step="scaled": a_j = 1 - (1 - alpha)^(n_j / r), r = B / k
     (each point acts as a small EMA step, so an average-sized cluster moves alpha);
   - warmup=True: alpha is replaced by its bias-corrected value alpha / (1 - prod gamma),
     one number shared by all clusters.

Revival rules (`revival`):
1. "count": the paper, with weight w = N (EMA of the counts). Every `revival_every` batches,
   clusters with w_j < tau mean(w) are refilled by splitting a donor drawn without
   replacement, proportionally to w, among the other clusters.
2. "error": the same procedure with w = E, an EMA of the per-cluster distortion e_j.
3. "none".

Each state keeps what its update and revival rule read, per cluster: sums + count: S, N;
centroid + count: C, N; centroid + error: C, E; sums + error: S, N, E. `track_error` adds E
to a count-revival run for analysis only. Named configurations are in emeans.methods.

Implementation choices not fixed by the paper:
1. The EMA state is kept in fp64; assignment runs in fp32. The state is tiny (k x d).
2. Revival draws donors only among non-victim clusters (splitting a dead cluster would
   revive nothing). If there are more victims than live clusters, the extra victims wait
   for the next pass.
3. A slot revived during the last epoch forgets its earlier last-epoch points, so its
   finalised centroid is the mean of the points it got after the revival; no revival is done
   after the last batch.
4. Two random generators per seed: one for the initial centroids and the shuffles, one for
   the revival donors. So runs with the same seed see the same batches in the same order,
   whatever their revival does.
5. Fractional epochs: M = round(T * floor(n/B)) updates; each epoch is a fresh shuffle
   and drops the last n mod B points. The "last epoch" for finalisation is the final
   floor(n/B) batches; points not seen there are ignored.
"""

from dataclasses import dataclass, field

import torch

from .assign import assign, cluster_stats
from .laws import alpha0_law, batch_law
from .lloyd import init_random, split_pair
from .schedule import cosine_alpha

UPDATES = ("sums", "cumulative", "centroid")
STEPS = ("fixed", "scaled")
REVIVALS = ("count", "error", "none")
N_INIT = 1e-10
REVIVAL_SEED_OFFSET = 1_000_003  # the revival generator is seeded with seed + this


@dataclass
class EMeansConfig:
    k: int
    epochs: float = 3.0
    batch_size: int | None = None  # None: the paper's scaling law
    alpha0: float | None = None  # None: the paper's scaling law
    update: str = "sums"
    step: str = "fixed"  # centroid update only
    warmup: bool = False  # centroid update only
    revival: str = "count"
    tau: float = 0.1  # revival threshold, as a fraction of the mean weight
    revival_every: int = 5
    floor: float = 0.1  # cosine schedule floor, as a fraction of alpha0
    seed: int = 0
    track_error: bool = False  # keep E in a count-revival run too (analysis only)


@dataclass
class EMeansResult:
    centroids: torch.Tensor  # after last-epoch-mean finalisation
    centroids_before_finalisation: torch.Tensor
    batch_size: int
    alpha0: float
    num_updates: int
    batches_per_epoch: int
    revival_events: list = field(default_factory=list)  # one dict per revived slot


def _ema_(t, alpha, value):
    t.mul_(1.0 - alpha).add_(value, alpha=alpha)


def _weights(k, device, count, error):
    n = torch.full((k,), N_INIT, dtype=torch.float64, device=device) if count else None
    e = torch.zeros(k, dtype=torch.float64, device=device) if error else None
    return n, e


class SumsState:
    """The paper's state: EMA of count N_j and sum S_j; centroids are S_j / N_j.
    With track_error it also keeps E_j, the EMA of the cluster's distortion. With
    forget=False the sums and counts are plain running totals (mini-batch k-means)."""

    def __init__(self, c0, track_error=False, forget=True):
        self.N, self.E = _weights(c0.shape[0], c0.device, True, track_error)
        self.S = self.N[:, None] * c0.double()
        self.forget = forget

    def centroids(self):
        return (self.S / self.N[:, None]).float()

    def update(self, alpha, counts, sums, distortion):
        if not self.forget:
            self.N += counts
            self.S += sums
            if self.E is not None:
                _ema_(self.E, alpha, distortion)
            return
        if alpha >= 1:
            # gamma = 0: clusters with points are overwritten; empty ones would get N = S = 0.
            # They keep their centroid with the tiny initial weight, so they count as dead.
            empty = counts == 0
            self.S = torch.where(empty[:, None], N_INIT * self.S / self.N[:, None], sums)
            self.N = torch.where(empty, N_INIT, counts)
            if self.E is not None:
                self.E = distortion.clone()
            return
        _ema_(self.N, alpha, counts)
        _ema_(self.S, alpha, sums)
        if self.E is not None:
            _ema_(self.E, alpha, distortion)

    def split(self, donor, slot):
        """Split the donor's centroid into two children; the victim `slot` gets one.
        Each child carries half the donor's N (and E), and S = c_child * N."""
        c_slot, c_donor = split_pair(self.S[donor] / self.N[donor])
        self.N[slot] = self.N[donor] = self.N[donor] / 2
        if self.E is not None:
            self.E[slot] = self.E[donor] = self.E[donor] / 2
        self.S[slot] = c_slot * self.N[slot]
        self.S[donor] = c_donor * self.N[donor]


class CentroidState:
    """Centroid EMA: c_j <- c_j + a_j (m_j - c_j) for clusters with points; see the module
    docstring for `step` and `warmup`. Besides C it keeps only the revival weight (N for count
    revival, E for error revival) and, for the warm-up, one number: the product of the decays."""

    def __init__(self, c0, step, warmup, ref, revival):
        self.C = c0.double().clone()
        self.N, self.E = _weights(c0.shape[0], c0.device, revival == "count", revival == "error")
        self.step, self.ref = step, ref
        self.decay = 1.0 if warmup else None  # prod of (1 - alpha) over all updates so far

    def centroids(self):
        return self.C.float()

    def update(self, alpha, counts, sums, distortion):
        a = alpha
        if self.decay is not None:
            self.decay *= 1 - alpha
            a = min(alpha / (1 - self.decay), 1.0)  # 1 on the first update; rounding can exceed it
        if self.step == "scaled":
            step = 1 - (1 - a) ** (counts / self.ref)  # 0 for clusters without points
        else:
            step = (counts > 0).double() * a
        means = sums / counts.clamp_min(1)[:, None]
        self.C += step[:, None] * (means - self.C)
        if self.N is not None:
            if alpha < 1:
                _ema_(self.N, alpha, counts)
            else:  # as in SumsState: empty clusters keep the tiny initial weight, so they count as dead
                self.N = torch.where(counts == 0, N_INIT, counts)
        if self.E is not None:
            _ema_(self.E, alpha, distortion)

    def split(self, donor, slot):
        self.C[slot], self.C[donor] = split_pair(self.C[donor])
        for w in (self.N, self.E):
            if w is not None:
                w[slot] = w[donor] = w[donor] / 2


def make_state(cfg, c0, batch_size):
    if cfg.update in ("sums", "cumulative"):
        return SumsState(c0, track_error=cfg.revival == "error" or cfg.track_error, forget=cfg.update == "sums")
    return CentroidState(c0, cfg.step, cfg.warmup, ref=batch_size / c0.shape[0], revival=cfg.revival)


def dead(weight, tau):
    """Clusters whose revival weight is below tau times the mean weight."""
    return weight < tau * weight.mean()


def revive(state, weight, tau, generator):
    """The paper's revival with a per-cluster weight (N for "count", E for "error"): every
    dead cluster is refilled by splitting a donor drawn without replacement, proportionally
    to the weight, among the live clusters. Returns the revived slots."""
    victims_mask = dead(weight, tau)
    victims = torch.nonzero(victims_mask).flatten()
    weights = weight.clone()
    weights[victims_mask] = 0
    victims = victims[: int((weights > 0).sum())]
    if victims.numel() == 0:
        return []
    donors = torch.multinomial(weights.cpu(), victims.numel(), replacement=False, generator=generator)
    for v, u in zip(victims.tolist(), donors.tolist()):
        state.split(u, v)
    return victims.tolist()


class RevivalTracker:
    """Churn and re-death: every revival opens an event for its slot; if the slot is revived
    again later, the event is closed with the batch at which it re-died. Events still open
    at the end survived (right-censored)."""

    def __init__(self):
        self.open = {}  # slot -> index of its open event
        self.events = []

    def record(self, batch, revived):
        for slot in revived:
            if slot in self.open:
                self.events[self.open[slot]]["died_at"] = batch
            self.open[slot] = len(self.events)
            self.events.append({"slot": slot, "revived_at": batch, "died_at": None})


def check(cfg, n):
    """The batch size and peak step of a run, validated."""
    if cfg.update not in UPDATES or cfg.step not in STEPS or cfg.revival not in REVIVALS:
        raise ValueError(f"unknown update, step or revival rule: {cfg}")
    if cfg.update != "centroid" and (cfg.step != "fixed" or cfg.warmup):
        raise ValueError("step and warmup only apply to the centroid update")
    if cfg.update == "centroid" and cfg.track_error:
        raise ValueError("track_error only applies to the sums update")
    B = batch_law(cfg.k, n, cfg.epochs) if cfg.batch_size is None else cfg.batch_size
    if not 0 < B <= n:
        raise ValueError(f"batch size {B} must be in [1, n = {n}]")
    alpha0 = alpha0_law(cfg.k, B, n, cfg.epochs) if cfg.alpha0 is None else cfg.alpha0
    if not 0 < alpha0 <= 1:
        raise ValueError(f"alpha0 = {alpha0} must be in (0, 1]")
    return B, alpha0


@torch.no_grad()
def emeans(x, cfg, init=None, on_epoch=None, on_revival=None):
    """Run e-means on x (n, d), a float32 tensor on the target device.

    `init` overrides the random initial centroids (used in tests only).
    `on_epoch(epoch, centroids)` is called after every full epoch with the current
    (not finalised) centroids, and `on_revival(batch, state)` before every revival pass.
    """
    n, k = x.shape[0], cfg.k
    B, alpha0 = check(cfg, n)
    per_epoch = n // B
    total = max(1, round(cfg.epochs * per_epoch))

    gen = torch.Generator().manual_seed(cfg.seed)  # initial centroids and shuffles
    revival_gen = torch.Generator().manual_seed(cfg.seed + REVIVAL_SEED_OFFSET)
    c0 = init_random(x, k, gen) if init is None else init.to(x.device, torch.float32)
    state = make_state(cfg, c0, B)
    tracker = RevivalTracker()
    last = torch.full((n,), -1, dtype=torch.int64, device=x.device)

    for t in range(total):
        if t % per_epoch == 0:
            perm = torch.randperm(n, generator=gen).to(x.device)
        rows = perm[(t % per_epoch) * B : (t % per_epoch + 1) * B]
        xb = x[rows]
        idx, dist = assign(xb, state.centroids())
        counts, sums = cluster_stats(xb, idx, k)
        distortion = torch.bincount(idx, weights=dist.double(), minlength=k)
        state.update(cosine_alpha(t, total, alpha0, cfg.floor), counts, sums, distortion)
        if t >= total - per_epoch:
            last[rows] = idx
        # no revival after the last batch: finalisation would overwrite it
        if cfg.revival != "none" and (t + 1) % cfg.revival_every == 0 and t + 1 < total:
            if on_revival is not None:
                on_revival(t + 1, state)
            weight = state.N if cfg.revival == "count" else state.E
            revived = revive(state, weight, cfg.tau, revival_gen)
            tracker.record(t + 1, revived)
            if revived:  # a revived slot is a new cluster: forget its last-epoch points
                last[torch.isin(last, torch.tensor(revived, device=last.device))] = -1
        if on_epoch is not None and (t + 1) % per_epoch == 0:
            on_epoch((t + 1) // per_epoch, state.centroids())

    before = state.centroids()
    c = before.clone()
    seen = last >= 0
    counts, sums = cluster_stats(x[seen], last[seen], k)
    alive = counts > 0
    c[alive] = (sums[alive] / counts[alive, None]).float()
    return EMeansResult(c, before, B, alpha0, total, per_epoch, tracker.events)
