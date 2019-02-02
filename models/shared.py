"""
Short utility functions.
"""

import torch


def cumsum_shifted(xs):
    """
    Assumes `xs` is a 1-d array.
    The usual cumsum has elements [x[1], x[1] + x[2], ...]. This one has elements
    [0, x[1], x[1] + x[2], ...]. Useful for calculating sentence offsets.
    """
    cs = xs.cumsum(dim=0)
    shift = torch.zeros(1, dtype=torch.long)
    return torch.cat([shift, cs[:-1]], dim=0)
