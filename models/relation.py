from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder, Pruner


class RelationExtractor(Model):
    """
    Named entity recognition module of DyGIE model.
    """
    # TODO(dwadden) add option to make `mention_feedforward` be the NER tagger.
    def __init__(self,
                 vocab: Vocabulary,
                 mention_feedforward: FeedForward,
                 relation_feedforward: FeedForward,
                 feature_size: int,
                 spans_per_word: float,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(RelationExtractor, self).__init__(vocab, regularizer)

        # TODO(dwadden) Do we want TimeDistributed for this one?
        # TODO(dwadden) make sure I've got the input dim right on this one.
        self._relation_feedforward = TimeDistributed(relation_feedforward)
        feedforward_scorer = torch.nn.Sequential(
            TimeDistributed(mention_feedforward),
            TimeDistributed(torch.nn.Linear(mention_feedforward.get_output_dim(), 1)))
        self._mention_pruner = Pruner(feedforward_scorer)
        self._relation_scorer = TimeDistributed(torch.nn.Linear(relation_feedforward.get_output_dim(), 1))

        self._spans_per_word = spans_per_word

        # TODO(dwadden) Add code to compute relation F1.
        self._relation_scores = None

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                spans: torch.IntTensor,
                span_mask,
                span_embeddings,  # TODO(dwadden) add type.
                sentence_lengths,
                relation_labels: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """
        TODO(dwadden) Write documentation.
        """
        import ipdb; ipdb.set_trace()
