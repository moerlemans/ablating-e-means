"""The paper's scaling laws for the batch size B and the peak step alpha0 (Table 2).

Both were fitted on DEEP for 1e5 <= n <= 1e8, 256 <= k <= 65536, n/k >= 8. We use the
default batch law (the paper divides it by 8 only for n > 2M, which we never reach).
"""

import math


def batch_law(k, n, epochs):
    """B = 0.035 k^0.13 n^0.88 T^0.81, rounded to the nearest power of two (in log scale), capped at n."""
    b = 0.035 * k**0.13 * n**0.88 * epochs**0.81
    return min(2 ** round(math.log2(b)), n)


def alpha0_law(k, batch_size, n, epochs):
    """alpha0 = min(0.824 k^-0.46 B^0.90 n^-0.45 T^-0.11, 1)."""
    return min(0.824 * k**-0.46 * batch_size**0.90 * n**-0.45 * epochs**-0.11, 1.0)
