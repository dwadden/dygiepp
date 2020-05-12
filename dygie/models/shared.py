"""
Short utility functions.
"""

import torch
import pandas as pd


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


def fields_to_batches(d, keys_to_ignore=[]):
    """
    The input is a dict whose items are batched tensors. The output is a list of dictionaries - one
    per entry in the batch - with the slices of the tensors for that entry. Here's an example.
    Input:
    d = {"a": [[1, 2], [3,4]], "b": [1, 2]}
    Output:
    res = [{"a": [1, 2], "b": 1}, {"a": [3, 4], "b": 2}].
    """
    # Make sure all input dicts have same length.
    keys = [key for key in d.keys() if key not in keys_to_ignore]
    lengths = [len(d[k]) for k in keys]
    assert len(set(lengths)) == 1
    length = lengths[0]
    res = [{k: d[k][i] for k in keys} for i in range(length)]
    return res


def is_seed_datasets(metadata):
    """
    For CORD experiments, all the seed datasets have a `:` in their `doc_key`s
    but the CORD documents do not. This whether we're dealing with CORD or the
    seed datasets.
    """
    doc_keys = pd.Series([x["doc_key"] for x in metadata])
    has_colon = doc_keys.str.contains(":").values
    # If they all have a colon then we're doing the seed datasets. If none of
    # them do we're doing the CORD datasets. If there's some with and some
    # without, there's a problem.
    if not (has_colon.all() or (~has_colon).all()):
        raise Exception("Can't tell whether we should make irrelevant labels.")
    has_colon = has_colon.all()
    return has_colon
