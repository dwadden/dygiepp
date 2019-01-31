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
    def __init__(self,
                 vocab: Vocabulary,
                 ner_tagger,    # Use the NER tagger to get mention scores.
                 relation_feedforward: FeedForward,
                 feature_size: int,
                 spans_per_word: float,
                 # initializer: InitializerApplicator = InitializerApplicator(), # TODO(dwadden add this).
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(RelationExtractor, self).__init__(vocab, regularizer)

        self._ner = ner_tagger

        # TODO(dwadden) Do we want TimeDistributed for this one?
        # TOOD(dwadden) make sure I've got hte input dim right on this one.
        self._relation_feedforward = TimeDistributed(relation_feedforward)

        self._spans_per_word = spans_per_word

        # TODO(dwadden) Add this.
        # initializer(self)


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
