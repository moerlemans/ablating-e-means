"""The named configurations of the write-up (keyword arguments of EMeansConfig)."""

METHODS = {
    "mini-batch k-means": dict(update="cumulative", revival="count", revival_every=5),
    "e-means": dict(update="sums", revival="count", revival_every=5),
    "centroid EMA": dict(update="centroid", step="fixed", warmup=False, revival="count", revival_every=5),
    "centroid EMA+": dict(update="centroid", step="scaled", warmup=True, revival="count", revival_every=5),
    "centroid EMA++": dict(update="centroid", step="scaled", warmup=True, revival="error", revival_every=1),
    "e-means + error revival": dict(update="sums", revival="error", revival_every=1),
}

# the same updates with count revival every batch, the schedule of error revival
CONTROLS = {
    "e-means, count revival every batch": {**METHODS["e-means"], "revival_every": 1},
    "centroid EMA+, count revival every batch": {**METHODS["centroid EMA+"], "revival_every": 1},
}

# centroid EMA+ with one of its two changes only
ABLATIONS = {
    "centroid EMA, warm-up only": {**METHODS["centroid EMA"], "warmup": True},
    "centroid EMA, count-scaled step only": {**METHODS["centroid EMA"], "step": "scaled"},
}

ALL = {**METHODS, **CONTROLS, **ABLATIONS}
