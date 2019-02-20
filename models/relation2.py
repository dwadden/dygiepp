import logging
import math
import itertools
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder, Pruner

from dygie.models import shared
from dygie.training.relation_metrics2 import RelationMetrics, CandidateRecall

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


# TODO(dwadden) add tensor dimension comments.
# TODO(dwadden) Different sentences should have different number of relation candidates depending on
# length.
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

        self._n_labels = vocab.get_vocab_size("relation_labels")

        feedforward_scorer = torch.nn.Sequential(
            TimeDistributed(mention_feedforward),
            TimeDistributed(torch.nn.Linear(mention_feedforward.get_output_dim(), 1)))
        self._mention_pruner = Pruner(feedforward_scorer)

        self._relation_feedforward = TimeDistributed(relation_feedforward)
        self._relation_scorer = TimeDistributed(torch.nn.Linear(
            relation_feedforward.get_output_dim(), self._n_labels))

        self._spans_per_word = spans_per_word

        self._relation_metrics = RelationMetrics()
        self._candidate_recall = CandidateRecall()

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                spans: torch.IntTensor,
                span_mask,
                span_embeddings,  # TODO(dwadden) add type.
                sentence_lengths,
                max_sentence_length,
                relation_labels: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """
        TODO(dwadden) Write documentation.
        """
        num_spans_to_keep = torch.floor(sentence_lengths.float() * self._spans_per_word)
        num_spans_to_keep = torch.max(num_spans_to_keep, torch.ones_like(num_spans_to_keep)).long()
        (top_span_embeddings, top_span_mask,
         top_span_indices, top_span_mention_scores) = self._mention_pruner(span_embeddings,
                                                                           span_mask,
                                                                           num_spans_to_keep)
        # TODO(dwadden) Add an assert that they're all in range.
        top_spans = util.batched_index_select(spans, top_span_indices)

        span_pair_embeddings = self._compute_span_pair_embeddings(top_span_embeddings)

        relation_scores = self._compute_relation_scores(span_pair_embeddings, top_span_mention_scores)

        _, predicted_relations = relation_scores.max(dim=-1)
        predicted_relations -= 1

        output_dict = {"top_spans": top_spans,
                       "relation_scores": relation_scores,
                       "predicted_relations": predicted_relations,
                       "num_spans_to_keep": num_spans_to_keep}

        ########################################

        # Evaluate.

        if relation_labels is not None:
            gold_relations = self._get_pruned_gold_relations(relation_labels, top_span_indices)
            cross_entropy = self._get_cross_entropy_loss(relation_scores, gold_relations, top_span_mask)
            output_dict["loss"] = cross_entropy

            decoded = self.decode(output_dict)
            self._relation_metrics(decoded, metadata)
            self._candidate_recall(decoded, metadata)

        return output_dict


    @overrides
    def decode(self, output_dict):
        """
        Take the output and convert it into a list of dicts. Each entry is a sentence. Each key is a
        pair of span indices for that sentence, and each value is the relation label on that span
        pair.
        """
        top_spans = output_dict["top_spans"].detach().cpu()
        num_spans_to_keep = output_dict["num_spans_to_keep"].detach().cpu()
        predicted_relations = output_dict["predicted_relations"].detach().cpu()
        batch_size = top_spans.size(0)
        predictions = []
        for i in range(batch_size):
            predictions_batch = {}
            n_spans = num_spans_to_keep[i].item()
            for j in range(n_spans):
                for k in range(n_spans):
                    label = predicted_relations[i, j, k].item()
                    if label >= 0:
                        spans = (tuple(top_spans[i][j].tolist()),
                                 tuple(top_spans[i][k].tolist()))
                        text_label = self.vocab.get_token_from_index(
                            label, namespace="relation_labels")
                        predictions_batch[spans] = text_label
            predictions.append(predictions_batch)
        return predictions

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        candidate_recall = self._candidate_recall.get_metric(reset)
        precision, recall, f1 = self._relation_metrics.get_metric(reset)
        return {"rel_precision": precision,
                "rel_recall": recall,
                "rel_f1": f1,
                "rel_span_recall": candidate_recall}

    @staticmethod
    def _compute_span_pair_embeddings(top_span_embeddings: torch.FloatTensor):
        """
        TODO(dwadden) document me and add comments.
        """
        n_candidates = top_span_embeddings.size(1)

        embs1 = top_span_embeddings.unsqueeze(dim=2)
        embs2 = top_span_embeddings.unsqueeze(dim=1)

        embs1_tiled = embs1.repeat(1, 1, n_candidates, 1)
        embs2_tiled = embs2.repeat(1, n_candidates, 1, 1)

        similarity_emb = embs1 * embs2

        pair_emb = torch.cat([embs1_tiled, embs2_tiled, similarity_emb], dim=-1)
        return pair_emb

    def _compute_relation_scores(self, pairwise_embeddings, top_span_mention_scores):
        scores = self._relation_scorer(self._relation_feedforward(pairwise_embeddings))
        scores += top_span_mention_scores.unsqueeze(1) + top_span_mention_scores.unsqueeze(2)
        dummy_size = [scores.size(0), scores.size(1), scores.size(2)]
        dummy_scores = torch.zeros(dummy_size, device=scores.device).unsqueeze(-1)
        scores = torch.cat([dummy_scores, scores], dim=-1)
        return scores

    @staticmethod
    def _get_pruned_gold_relations(relation_labels, top_span_indices):
        """
        Loop over each slice and get the labels for the spans from that slice.
        Mask out all relations where either of the spans involved was masked out.
        TODO(dwadden) document me better.
        """
        res = []
        for labels, ixs in zip(relation_labels, top_span_indices):
            res.append(labels[ixs][:, ixs].unsqueeze(0))
        return torch.cat(res, dim=0)

    def _get_cross_entropy_loss(self, relation_scores, relation_labels, mask):
        """
        Compute cross-entropy loss on relation labels. Ignore diagonal entries and entries giving
        relations between masked out spans.
        """
        keep = []
        for entry in mask:
            keep_slice = entry.unsqueeze(0) & entry.unsqueeze(1)
            keep.append(keep_slice.unsqueeze(0))
        keep = torch.cat(keep, dim=0).view(-1).byte()

        flat_scores = relation_scores.view(-1, self._n_labels + 1)

        shifted_labels = relation_labels + 1  # Shift by 1 to match predictions.
        flat_labels = shifted_labels.view(-1).long()

        return F.cross_entropy(flat_scores[keep], flat_labels[keep])
