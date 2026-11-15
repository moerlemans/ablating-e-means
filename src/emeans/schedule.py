"""Cosine step schedule (paper Eq. 5)."""

import math


def cosine_alpha(t, total, alpha0, floor=0.1):
    """alpha(t) = alpha0 * (floor + (1 - floor) * (1 + cos(pi t / M)) / 2), t in {0..M-1}."""
    return alpha0 * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t / total)))
