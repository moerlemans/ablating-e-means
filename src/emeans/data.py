"""SIFT-128 (ann-benchmarks): a 200k training sample, a validation set and the test queries."""

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SIFT_URL = "http://ann-benchmarks.com/sift-128-euclidean.hdf5"


@dataclass
class Splits:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def download_sift():
    path = DATA_DIR / "sift-128-euclidean.hdf5"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        print(f"downloading {SIFT_URL} -> {path}")
        urllib.request.urlretrieve(SIFT_URL, tmp)
        tmp.rename(path)
    return path


def load_sift(n_train=200_000, n_val=10_000, seed=0):
    """SIFT-128: train = random n_train of the 1M base vectors, val = n_val other
    base vectors, test = the 10k provided queries.

    All splits are shifted by the train mean, which keeps the fp32 expansion
    ||x||^2 - 2x.c + ||c||^2 accurate (raw SIFT has ||x||^2 ~ 2.6e5). Assignments and MSEs
    are translation invariant; the centroid split used for revival (split_pair) is not, as
    it perturbs coordinates multiplicatively. The paper does the same shift only in its INT8
    path (App. J.2).
    """
    with h5py.File(download_sift(), "r") as f:
        base = f["train"][:].astype(np.float32)
        test = f["test"][:].astype(np.float32)
    perm = np.random.default_rng(seed).permutation(base.shape[0])
    train = base[perm[:n_train]]
    val = base[perm[n_train : n_train + n_val]]
    mean = train.mean(0, dtype=np.float64).astype(np.float32)
    return Splits(train - mean, val - mean, test - mean)
