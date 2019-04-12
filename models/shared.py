"""
Short utility functions.
"""

import torch

from allennlp.modules import TimeDistributed, Pruner


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


def make_pruner(scorer, entity_beam):
    """
    Create a pruner that either takes outputs of other scorers (i.e. entity beam), or uses its own
    scorer (the `default_scorer`).
    """
    item_scorer = torch.nn.Sequential(
        TimeDistributed(scorer),
        TimeDistributed(torch.nn.Linear(scorer.get_output_dim(), 1)))
    min_score_to_keep = 1e-10 if entity_beam else None

    return Pruner(item_scorer, entity_beam, min_score_to_keep)
