import torch


def blobs(k=10, d=8, n=4000, spread=3.0, seed=0):
    """Isotropic Gaussian blobs with unit variance around centres drawn from N(0, spread^2 I).
    Returns (points, centres) as float32 CPU tensors."""
    g = torch.Generator().manual_seed(seed)
    centres = spread * torch.randn(k, d, generator=g)
    labels = torch.randint(0, k, (n,), generator=g)
    return centres[labels] + torch.randn(n, d, generator=g), centres
