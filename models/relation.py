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
from dygie.training.relation_metrics import RelationMetrics

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

        # TODO(dwadden) Do we want TimeDistributed for this one?
        # TODO(dwadden) make sure I've got the input dim right on this one.
        self._relation_feedforward = TimeDistributed(relation_feedforward)
        feedforward_scorer = torch.nn.Sequential(
            TimeDistributed(mention_feedforward),
            TimeDistributed(torch.nn.Linear(mention_feedforward.get_output_dim(), 1)))
        self._mention_pruner = Pruner(feedforward_scorer)
        # Output dim is num labels - 1 since we set the score on the null label to 0.
        self._relation_scorer = TimeDistributed(torch.nn.Linear(
            relation_feedforward.get_output_dim(), self._n_labels))

        self._spans_per_word = spans_per_word

        # TODO(dwadden) Add code to compute relation F1.
        self._relation_metrics = RelationMetrics()

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
        num_spans = spans.size(1)  # Max number of spans for the minibatch.

        # Keep different number of spans for each minibatch entry.
        num_spans_to_keep = torch.floor(sentence_lengths.float() * self._spans_per_word).long()

        (top_span_embeddings, top_span_mask,
         top_span_indices, top_span_mention_scores) = self._mention_pruner(span_embeddings,
                                                                           span_mask,
                                                                           num_spans_to_keep)

        # Convert to Boolean for logical indexing operations later.
        top_span_mask = top_span_mask.unsqueeze(-1).byte()

        flat_top_span_indices = util.flatten_and_batch_shift_indices(top_span_indices, num_spans)
        top_spans = util.batched_index_select(spans,
                                              top_span_indices,
                                              flat_top_span_indices)

        span_pair_embeddings = self._compute_span_pair_embeddings(top_span_embeddings)
        relation_scores = self._compute_relation_scores(span_pair_embeddings,
                                                        top_span_mention_scores)
        # Subtract 1 so that the "null" relation corresponds to -1.
        _, predicted_relations = relation_scores.max(-1)
        predicted_relations -= 1

        output_dict = {"top_spans": top_spans,
                       "predicted_relations": predicted_relations}

        # Evaluate loss and F1 if labels were probided.
        if relation_labels is not None:
            # TODO(dwadden) Figure out why the relation labels didn't get read in as ints.
            relation_labels = relation_labels.long()

            # Compute cross-entropy loss.
            gold_relations, relation_mask = self._get_pruned_gold_relations(
                relation_labels, top_span_indices, top_span_mask)

            cross_entropy = self._get_cross_entropy_loss(
                relation_scores, gold_relations, relation_mask)

            # Compute F1.
            predictions = self.decode(output_dict)
            assert len(predictions) == len(metadata)  # Make sure length of predictions is right.
            self._relation_metrics(predictions, metadata)

            output_dict["loss"] = cross_entropy

        return output_dict

    @overrides
    def decode(self, output_dict):
        """
        Take the output and convert it into a list of dicts. Each entry is a sentence. Each key is a
        pair of span indices for that sentence, and each value is the relation label on that span
        pair.
        """
        # TODO(dwadden) Should I enforce that we can't have self-relations, etc?
        top_spans_batch = output_dict["top_spans"].detach().cpu()
        predicted_relations_batch = output_dict["predicted_relations"].detach().cpu()
        res = []

        # Collect predictions for each sentence in minibatch.
        for top_spans, predicted_relations in zip(top_spans_batch, predicted_relations_batch):
            entry = self._decode_sentence(top_spans, predicted_relations)
            res.append(entry)

        return res

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        precision, recall, f1 = self._relation_metrics.get_metric(reset)
        return {"relation_precision": precision,
                "relation_recall": recall,
                "relation_f1": f1}

    def _decode_sentence(self, top_spans, predicted_relations):
        # Convert to native Python lists for easy iteration.
        top_spans = [tuple(x) for x in top_spans.tolist()]
        predicted_label_ixs = predicted_relations.view(-1).tolist()
        predicted_labels = [self.vocab.get_token_from_index(x, namespace="relation_labels")
                            if x >= 0 else None for x in predicted_label_ixs]

        # Iterate over all span pairs and labels. Record the span if the label isn't null.
        res = {}
        for (span_1, span_2), predicted_label in zip(itertools.product(top_spans, top_spans),
                                                     predicted_labels):
            if predicted_label is not None:
                res[(span_1, span_2)] = predicted_label

        return res

    @staticmethod
    def _compute_span_pair_embeddings(top_span_embeddings: torch.FloatTensor):
        """
        TODO(dwadden) document me and add comments.
        """
        # Shape: (batch_size, num_spans_to_keep, num_spans_to_keep, embedding_size)
        num_candidates = top_span_embeddings.size(1)

        embeddings_1_expanded = top_span_embeddings.unsqueeze(2)
        embeddings_1_tiled = embeddings_1_expanded.repeat(1, 1, num_candidates, 1)

        embeddings_2_expanded = top_span_embeddings.unsqueeze(1)
        embeddings_2_tiled = embeddings_2_expanded.repeat(1, num_candidates, 1, 1)

        similarity_embeddings = embeddings_1_expanded * embeddings_2_expanded

        pair_embeddings_list = [embeddings_1_tiled, embeddings_2_tiled, similarity_embeddings]
        pair_embeddings = torch.cat(pair_embeddings_list, dim=3)

        return pair_embeddings

    def _compute_relation_scores(self, pairwise_embeddings, top_span_mention_scores):
        relation_scores = self._relation_scorer(
            self._relation_feedforward(pairwise_embeddings)).squeeze(-1)
        # Add the mention scores for each of the candidates.

        relation_scores += (top_span_mention_scores.unsqueeze(-1) +
                            top_span_mention_scores.transpose(1, 2).unsqueeze(-1))

        shape = [relation_scores.size(0), relation_scores.size(1), relation_scores.size(2), 1]
        dummy_scores = relation_scores.new_zeros(*shape)

        relation_scores = torch.cat([dummy_scores, relation_scores], -1)
        return relation_scores

    @staticmethod
    def _get_pruned_gold_relations(relation_labels, top_span_indices, top_span_masks):
        """
        Loop over each slice and get the labels for the spans from that slice.
        Mask out all relations where either of the spans involved was masked out.
        TODO(dwadden) document me better.
        """
        # TODO(dwadden) Test and possibly optimize.
        relations = []
        mask = []

        for sliced, ixs, top_span_mask in zip(relation_labels, top_span_indices, top_span_masks):
            entry = sliced[ixs][:, ixs].unsqueeze(0)
            mask_entry = top_span_mask & top_span_mask.transpose(0, 1).unsqueeze(0)
            relations.append(entry)
            mask.append(mask_entry)

        return torch.cat(relations, dim=0), torch.cat(mask, dim=0)

    def _get_cross_entropy_loss(self, relation_scores, relation_labels, relation_mask):
        """
        Compute cross-entropy loss on relation labels. Ignore diagonal entries and entries giving
        relations between masked out spans.
        """
        # Don't compute loss on predicted relation between a span and itself. When computing
        # softmax, mask out the losses from the diagonal elements of the relation matrix. Also mask
        # out invalid spans.
        batch_size = relation_scores.size(0)
        mat_size = relation_scores.size(1)
        identity_mask = shared.batch_identity(
            batch_size, mat_size, dtype=torch.uint8, device=relation_mask.device)
        mask = (~identity_mask & relation_mask).view(-1)
        # Need to add one for the null class.
        scores_flat = relation_scores.view(-1, self._n_labels + 1)[mask]
        # Need to add 1 so that the null label is 0, to line up with indices into prediction matrix.
        labels_flat = relation_labels.view(-1)[mask] + 1
        # Compute cross-entropy loss.
        loss = F.cross_entropy(scores_flat, labels_flat)
        return loss
