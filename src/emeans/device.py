"""Device helpers."""

import torch


def get_device(device=None):
    """The given device, or CUDA if available, else the CPU."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device(x, device):
    """A float32 tensor of x on `device`."""
    return torch.as_tensor(x, dtype=torch.float32).to(device)
