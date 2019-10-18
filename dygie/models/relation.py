import logging
import itertools
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.modules import TimeDistributed

from dygie.training.relation_metrics import RelationMetrics, CandidateRecall
from dygie.models.entity_beam_pruner import Pruner

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


# TODO(dwadden) add tensor dimension comments.
# TODO(dwadden) Different sentences should have different number of relation candidates depending on
# length.
class RelationExtractor(Model):
    """
    Relation extraction module of DyGIE model.
    """
    # TODO(dwadden) add option to make `mention_feedforward` be the NER tagger.
    def __init__(self,
                 vocab: Vocabulary,
                 mention_feedforward: FeedForward,
                 relation_feedforward: FeedForward,
                 feature_size: int,
                 spans_per_word: float,
                 span_emb_dim: int,
                 rel_prop: int = 0,
                 rel_prop_dropout_A: float = 0.0,
                 rel_prop_dropout_f: float = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 positive_label_weight: float = 1.0,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(RelationExtractor, self).__init__(vocab, regularizer)

        # Need to hack this for cases where there's no relation data. It breaks Ulme's code.
        self._n_labels = max(vocab.get_vocab_size("relation_labels"), 1)

        # Span candidate scorer.
        # TODO(dwadden) make sure I've got the input dim right on this one.
        feedforward_scorer = torch.nn.Sequential(
            TimeDistributed(mention_feedforward),
            TimeDistributed(torch.nn.Linear(mention_feedforward.get_output_dim(), 1)))
        self._mention_pruner = Pruner(feedforward_scorer)

        # Relation scorer.
        self._relation_feedforward = relation_feedforward
        self._relation_scorer = torch.nn.Linear(relation_feedforward.get_output_dim(), self._n_labels)

        self._spans_per_word = spans_per_word

        # TODO(dwadden) Add code to compute relation F1.
        self._candidate_recall = CandidateRecall()
        self._relation_metrics = RelationMetrics()

        class_weights = torch.cat([torch.tensor([1.0]), positive_label_weight * torch.ones(self._n_labels)])
        self._loss = torch.nn.CrossEntropyLoss(reduction="sum", ignore_index=-1, weight=class_weights)
        self.rel_prop = rel_prop

        # Relation Propagation
        self._A_network = FeedForward(input_dim=self._n_labels,
                                      num_layers=1,
                                      hidden_dims=span_emb_dim,
                                      activations=lambda x: x,
                                      dropout=rel_prop_dropout_A)
        self._f_network = FeedForward(input_dim=2*span_emb_dim,
                                      num_layers=1,
                                      hidden_dims=span_emb_dim,
                                      activations=torch.nn.Sigmoid(),
                                      dropout=rel_prop_dropout_f)

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

        output_dict = self.compute_representations(
            spans, span_mask, span_embeddings, sentence_lengths, relation_labels, metadata)

        if self.rel_prop > 0:
            output_dict = self.relation_propagation(output_dict)

        if self._loss_weights['relation'] > 0:
            output_dict = self.predict_labels(relation_labels, output_dict, metadata)

        return output_dict

    def compute_representations(self,  # type: ignore
                                spans: torch.IntTensor,
                                span_mask,
                                span_embeddings,  # TODO(dwadden) add type.
                                sentence_lengths,
                                relation_labels: torch.IntTensor = None,
                                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        (top_span_embeddings, top_span_mention_scores,
         num_spans_to_keep, top_span_mask,
         top_span_indices, top_spans) = self._prune_spans(spans, span_mask, span_embeddings, sentence_lengths)

        relation_scores = self.get_relation_scores(top_span_embeddings,
                                              top_span_mention_scores)

        output_dict = {"top_spans": top_spans,
                       "top_span_embeddings": top_span_embeddings,
                       "top_span_mention_scores": top_span_mention_scores,
                       "relation_scores": relation_scores,
                       "num_spans_to_keep": num_spans_to_keep,
                       "top_span_indices": top_span_indices,
                       "top_span_mask": top_span_mask,
                       "loss": 0}

        return output_dict

    def _prune_spans(self, spans, span_mask, span_embeddings, sentence_lengths):
        # Prune
        num_spans = spans.size(1)  # Max number of spans for the minibatch.

        # Keep different number of spans for each minibatch entry.
        num_spans_to_keep = torch.ceil(sentence_lengths.float() * self._spans_per_word).long()

        (top_span_embeddings, top_span_mask,
         top_span_indices, top_span_mention_scores, num_spans_kept) = self._mention_pruner(
             span_embeddings, span_mask, num_spans_to_keep)

        top_span_mask = top_span_mask.unsqueeze(-1)

        flat_top_span_indices = util.flatten_and_batch_shift_indices(top_span_indices, num_spans)
        top_spans = util.batched_index_select(spans,
                                              top_span_indices,
                                              flat_top_span_indices)

        return top_span_embeddings, top_span_mention_scores, num_spans_to_keep, top_span_mask, top_span_indices, top_spans


    def relation_propagation(self, output_dict):
        relation_scores = output_dict["relation_scores"]
        top_span_embeddings = output_dict["top_span_embeddings"]
        var = output_dict["top_span_mask"]
        top_span_mask_tensor = (var.repeat(1, 1, var.shape[1]) * var.view(var.shape[0], 1, var.shape[1]).repeat(1, var.shape[1], 1)).float()
        span_num = relation_scores.shape[1]
        normalization_factor = var.view(var.shape[0], span_num).sum(dim=1).float()
        for t in range(self.rel_prop):
            # TODO(Ulme) There is currently an implicit assumption that the null label is in the 0-th index.
            # Come up with how to deal with this
            relation_scores = F.relu(relation_scores[:, :, :, 1:], inplace=False)
            relation_embeddings = self._A_network(relation_scores)
            relation_embeddings = (relation_embeddings.transpose(3, 2).transpose(2, 1).transpose(1, 0) * top_span_mask_tensor).transpose(0, 1).transpose(1, 2).transpose(2, 3)
            entity_embs = torch.sum(relation_embeddings.transpose(2, 1).transpose(1, 0) * top_span_embeddings, dim=0)
            entity_embs = (entity_embs.transpose(0, 2) / normalization_factor).transpose(0, 2)
            f_network_input = torch.cat([top_span_embeddings, entity_embs], dim=-1)
            f_weights = self._f_network(f_network_input)
            top_span_embeddings = f_weights * top_span_embeddings + (1.0 - f_weights) * entity_embs
            relation_scores = self.get_relation_scores(top_span_embeddings, self._mention_pruner._scorer(top_span_embeddings))

        output_dict["relation_scores"] = relation_scores
        output_dict["top_span_embeddings"] = top_span_embeddings
        return output_dict
    def predict_labels(self, relation_labels, output_dict, metadata):
        relation_scores = output_dict["relation_scores"]

        # Subtract 1 so that the "null" relation corresponds to -1.
        _, predicted_relations = relation_scores.max(-1)
        predicted_relations -= 1

        output_dict["predicted_relations"] = predicted_relations

        # Evaluate loss and F1 if labels were provided.
        if relation_labels is not None:
            # Compute cross-entropy loss.
            gold_relations = self._get_pruned_gold_relations(
                relation_labels, output_dict["top_span_indices"], output_dict["top_span_mask"])

            cross_entropy = self._get_cross_entropy_loss(relation_scores, gold_relations)

            # Compute F1.
            predictions = self.decode(output_dict)["decoded_relations_dict"]
            assert len(predictions) == len(metadata)  # Make sure length of predictions is right.
            self._candidate_recall(predictions, metadata)
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
        top_spans_batch = output_dict["top_spans"].detach().cpu()
        predicted_relations_batch = output_dict["predicted_relations"].detach().cpu()
        num_spans_to_keep_batch = output_dict["num_spans_to_keep"].detach().cpu()
        res_dict = []
        res_list = []

        # Collect predictions for each sentence in minibatch.
        zipped = zip(top_spans_batch, predicted_relations_batch, num_spans_to_keep_batch)
        for top_spans, predicted_relations, num_spans_to_keep in zipped:
            entry_dict, entry_list = self._decode_sentence(
                top_spans, predicted_relations, num_spans_to_keep)
            res_dict.append(entry_dict)
            res_list.append(entry_list)

        output_dict["decoded_relations_dict"] = res_dict
        output_dict["decoded_relations"] = res_list
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        precision, recall, f1 = self._relation_metrics.get_metric(reset)
        candidate_recall = self._candidate_recall.get_metric(reset)
        return {"rel_precision": precision,
                "rel_recall": recall,
                "rel_f1": f1,
                "rel_span_recall": candidate_recall}

    def _decode_sentence(self, top_spans, predicted_relations, num_spans_to_keep):
        # TODO(dwadden) speed this up?
        # Throw out all predictions that shouldn't be kept.
        keep = num_spans_to_keep.item()
        top_spans = [tuple(x) for x in top_spans.tolist()]

        # Iterate over all span pairs and labels. Record the span if the label isn't null.
        res_dict = {}
        res_list = []
        for i, j in itertools.product(range(keep), range(keep)):
            span_1 = top_spans[i]
            span_2 = top_spans[j]
            label = predicted_relations[i, j].item()
            if label >= 0:
                label_name = self.vocab.get_token_from_index(label, namespace="relation_labels")
                res_dict[(span_1, span_2)] = label_name
                list_entry = (span_1[0], span_1[1], span_2[0], span_2[1], label_name)
                res_list.append(list_entry)

        return res_dict, res_list

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

    def get_relation_scores(self, top_span_embeddings, top_span_mention_scores):
        return self._compute_relation_scores(self._compute_span_pair_embeddings(top_span_embeddings),
                                             top_span_mention_scores)

    def _compute_relation_scores(self, pairwise_embeddings, top_span_mention_scores):
        batch_size = pairwise_embeddings.size(0)
        max_num_spans = pairwise_embeddings.size(1)
        feature_dim = self._relation_feedforward.input_dim

        embeddings_flat = pairwise_embeddings.view(-1, feature_dim)

        relation_projected_flat = self._relation_feedforward(embeddings_flat)
        relation_scores_flat = self._relation_scorer(relation_projected_flat)

        relation_scores = relation_scores_flat.view(batch_size, max_num_spans, max_num_spans, -1)

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
        All labels are offset by 1 so that the "null" label gets class zero. This is the desired
        behavior for the softmax. Labels corresponding to masked relations keep the label -1, which
        the softmax loss ignores.
        """
        # TODO(dwadden) Test and possibly optimize.
        relations = []

        for sliced, ixs, top_span_mask in zip(relation_labels, top_span_indices, top_span_masks.bool()):
            entry = sliced[ixs][:, ixs].unsqueeze(0)
            mask_entry = top_span_mask & top_span_mask.transpose(0, 1).unsqueeze(0)
            entry[mask_entry] += 1
            entry[~mask_entry] = -1
            relations.append(entry)

        return torch.cat(relations, dim=0)

    def _get_cross_entropy_loss(self, relation_scores, relation_labels):
        """
        Compute cross-entropy loss on relation labels. Ignore diagonal entries and entries giving
        relations between masked out spans.
        """
        # Need to add one for the null class.
        scores_flat = relation_scores.view(-1, self._n_labels + 1)
        # Need to add 1 so that the null label is 0, to line up with indices into prediction matrix.
        labels_flat = relation_labels.view(-1)
        # Compute cross-entropy loss.
        loss = self._loss(scores_flat, labels_flat)
        return loss
