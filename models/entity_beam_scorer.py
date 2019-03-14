"""
It may be useful for the relation and event modules to score candidates using the ner scorer. In
particular, compute the ner score for all classes, and get the span candidate score by taking the
maximum score over the non-null classes.
"""

import torch


class EntityBeamScorer(torch.nn.Module):
    def __init__(self, scorer):
        super(EntityBeamScorer, self).__init__()
        self._scorer = scorer  # The ner scorer should NOT return scores for the null label.

    def forward(self, candidates):
        # [batch_size, n_spans, n_labels]
        scores = self._scorer(candidates)
        max_scores, _ = scores.max(dim=-1)
        # [batch_size, n_spans, 1]
        return max_scores.unsqueeze(-1)
