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
    shift = torch.zeros(1, dtype=torch.long, device=cs.device)  # Put on correct device.
    return torch.cat([shift, cs[:-1]], dim=0)


def batch_identity(batch_size, matrix_size, *args, **kwargs):
    """
    Tile the identity matrix along axis 0, `batch_size` times.
    """
    ident = torch.eye(matrix_size, *args, **kwargs).unsqueeze(0)
    res = ident.repeat(batch_size, 1, 1)
    return res


def fields_to_batches(d):
    """
    The input is a dict whose items are batched tensors. The output is a list of dictionaries - one
    per entry in the batch - with the slices of the tensors for that entry. Here's an example.

    Input:
    d = {"a": [[1, 2], [3,4]], "b": [1, 2]}
    Output:
    res = [{"a": [1, 2], "b": 1}, {"a": [3, 4], "b": 2}].
    """
    # Make sure all input dicts have same length.
    lengths = [len(x) for x in d.values()]
    assert len(set(lengths)) == 1
    length = lengths[0]
    keys = d.keys()
    res = [{k: d[k][i] for k in keys} for i in range(length)]
    return res


# Got this one from https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/33, under
# posting "A batch version that works".
def one_hot(x, num_classes):
    """
    Given an integer matrix of size (batch_size x n_tokens), convert to a one-hot matrix of size
    (batch_size x n_tokens x n_classes).
    Assumes that null class has label 0 (not -1).
    """
    assert x.min().item() == 0
    batch_size, seq_length = x.size()
    res = torch.zeros(batch_size, seq_length, num_classes, device=x.device)
    res.scatter_(2, x.unsqueeze(2), 1)
    return res
