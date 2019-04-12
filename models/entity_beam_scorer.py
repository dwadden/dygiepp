"""
It may be useful for the relation and event modules to score candidates using the ner scorer. In
particular, compute the ner score for all classes, and get the span candidate score by taking the
maximum score over the non-null classes.

When using the "entity beam" setting, only let through candidates that would be called as non-null.
"""

import torch

from allennlp.modules import TimeDistributed, Pruner


def make_pruner(default_scorer, entity_beam_scorer, entity_beam):
    """
    Create a pruner that either uses an entity beam scorer or a default scorer, depending on whether
    the entity_beam flag is true or not.
    """
    if entity_beam:
        item_scorer = EntityBeamScorer(entity_beam_scorer)
        min_score_to_keep = 1e-10  # Not exactly zero, to make sure no 0's get through due to numerical noise.
    else:
        item_scorer = torch.nn.Sequential(
            TimeDistributed(default_scorer),
            TimeDistributed(torch.nn.Linear(default_scorer.get_output_dim(), 1)))
        min_score_to_keep = None

    return Pruner(item_scorer, min_score_to_keep)


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
