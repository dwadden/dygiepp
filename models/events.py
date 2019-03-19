import logging
import itertools
from typing import Any, Dict, List, Optional

import torch
from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.modules import TimeDistributed, Pruner

from dygie.training.relation_metrics import RelationMetrics, CandidateRecall
from dygie.models.entity_beam_scorer import EntityBeamScorer
from dygie.training.event_metrics import EventMetrics
from dygie.models.shared import fields_to_batches
# TODO(dwadden) rename NERMetrics

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


# TODO(dwadden) add tensor dimension comments.
# TODO(dwadden) Different sentences should have different number of relation candidates depending on
# length.
class EventExtractor(Model):
    """
    Event extraction for DyGIE.
    """
    # TODO(dwadden) add option to make `mention_feedforward` be the NER tagger.
    def __init__(self,
                 vocab: Vocabulary,
                 trigger_feedforward: FeedForward,
                 ner_scorer: FeedForward,
                 argument_feedforward: FeedForward,
                 feature_size: int,
                 trigger_spans_per_word: float,
                 argument_spans_per_word: float,
                 loss_weights,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 positive_label_weight: float = 1.0,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(EventExtractor, self).__init__(vocab, regularizer)

        self._n_trigger_labels = vocab.get_vocab_size("trigger_labels")
        self._n_argument_labels = vocab.get_vocab_size("argument_labels")

        # Weight on trigger labeling and argument labeling.
        self._loss_weights = loss_weights.as_dict()

        # Trigger candidate scorer.
        null_label = vocab.get_token_index("", "trigger_labels")
        assert null_label == 0  # If not, the dummy class won't correspond to the null label.
        self._trigger_scorer = torch.nn.Sequential(
            TimeDistributed(trigger_feedforward),
            TimeDistributed(torch.nn.Linear(trigger_feedforward.get_output_dim(),
                                            self._n_trigger_labels - 1)))
        self._trigger_pruner = Pruner(EntityBeamScorer(self._trigger_scorer))

        # Use the same argument scorer as the NER module used, since arguments are always named
        # entities.
        argument_scorer = EntityBeamScorer(ner_scorer)
        self._argument_pruner = Pruner(argument_scorer)

        # Argument scorer.
        self._argument_feedforward = argument_feedforward
        self._argument_scorer = torch.nn.Linear(argument_feedforward.get_output_dim(), self._n_argument_labels)

        self._trigger_spans_per_word = trigger_spans_per_word
        self._argument_spans_per_word = argument_spans_per_word

        # TODO(dwadden) Add metrics.
        self._metrics = EventMetrics()

        self._trigger_loss = torch.nn.CrossEntropyLoss(reduction="sum")
        # TODO(dwadden) add loss weights.
        self._argument_loss = torch.nn.CrossEntropyLoss(reduction="sum", ignore_index=-1)
        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                trigger_mask,
                trigger_embeddings,
                spans,
                span_mask,
                span_embeddings,  # TODO(dwadden) add type.
                sentence_lengths,
                trigger_labels,
                argument_labels,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """
        TODO(dwadden) Write documentation.
        The trigger embeddings are just the contextualized token embeddings, and the trigger mask is
        the text mask. For the arguments, we consider all the spans.
        """
        # Compute trigger scores.
        trigger_scores = self._compute_trigger_scores(trigger_embeddings, trigger_mask)

        # Get trigger candidates for event argument labeling.
        num_trigs_to_keep = torch.floor(
            sentence_lengths.float() * self._trigger_spans_per_word).long()
        num_trigs_to_keep = torch.max(num_trigs_to_keep,
                                      torch.ones_like(num_trigs_to_keep))

        (top_trig_embeddings, top_trig_mask,
         top_trig_indices, top_trig_scores) = self._trigger_pruner(trigger_embeddings,
                                                                   trigger_mask,
                                                                   num_trigs_to_keep)
        top_trig_mask = top_trig_mask.unsqueeze(-1)

        # Compute the number of argument spans to keep.
        num_arg_spans_to_keep = torch.floor(
            sentence_lengths.float() * self._argument_spans_per_word).long()
        num_arg_spans_to_keep = torch.max(num_arg_spans_to_keep,
                                          torch.ones_like(num_arg_spans_to_keep))

        (top_arg_embeddings, top_arg_mask,
         top_arg_indices, top_arg_scores) = self._argument_pruner(span_embeddings,
                                                                  span_mask,
                                                                  num_arg_spans_to_keep)

        top_arg_mask = top_arg_mask.unsqueeze(-1)
        top_arg_spans = util.batched_index_select(spans,
                                                  top_arg_indices)

        trig_arg_embeddings = self._compute_trig_arg_embeddings(top_trig_embeddings,
                                                                top_arg_embeddings)

        argument_scores = self._compute_argument_scores(trig_arg_embeddings,
                                                        top_trig_scores,
                                                        top_arg_scores)

        _, predicted_triggers = trigger_scores.max(-1)
        _, predicted_arguments = argument_scores.max(-1)
        predicted_arguments -= 1  # The null argument has label -1.

        output_dict = {"top_trigger_indices": top_trig_indices,
                       "top_argument_spans": top_arg_spans,
                       "trigger_scores": trigger_scores,
                       "argument_scores": argument_scores,
                       "predicted_triggers": predicted_triggers,
                       "predicted_arguments": predicted_arguments,
                       "num_triggers_to_keep": num_trigs_to_keep,
                       "num_argument_spans_to_keep": num_arg_spans_to_keep,
                       "sentence_lengths": sentence_lengths}

        # Evaluate loss and F1 if labels were provided.
        if trigger_labels is not None and argument_labels is not None:
            # Compute the loss for both triggers and arguments.
            trigger_loss = self._get_trigger_loss(trigger_scores, trigger_labels, trigger_mask)

            gold_arguments = self._get_pruned_gold_arguments(
                argument_labels, top_trig_indices, top_arg_indices, top_trig_mask, top_arg_mask)

            argument_loss = self._get_argument_loss(argument_scores, gold_arguments)

            # Compute F1.
            predictions = self.decode(output_dict)["decoded_events"]
            assert len(predictions) == len(metadata)  # Make sure length of predictions is right.
            self._metrics(predictions, metadata)

            output_dict["loss"] = (self._loss_weights["trigger"] * trigger_loss +
                                   self._loss_weights["arguments"] * argument_loss)

        return output_dict

    @overrides
    def decode(self, output_dict):
        """
        Take the output and convert it into a list of dicts. Each entry is a sentence. Each key is a
        pair of span indices for that sentence, and each value is the relation label on that span
        pair.
        """
        outputs = fields_to_batches({k: v.detach().cpu() for k, v in output_dict.items()})

        res = []

        # Collect predictions for each sentence in minibatch.
        for output in outputs:
            decoded_trig = self._decode_trigger(output)
            decoded_args = self._decode_arguments(output)
            entry = dict(trigger_dict=decoded_trig, argument_dict=decoded_args)
            res.append(entry)

        output_dict["decoded_events"] = res
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._metrics.get_metric(reset)

    def _decode_trigger(self, output):
        trigger_dict = {}
        for i in range(output["sentence_lengths"]):
            trig_label = output["predicted_triggers"][i].item()
            if trig_label > 0:
                trigger_dict[i] = self.vocab.get_token_from_index(trig_label, namespace="trigger_labels")

        return trigger_dict

    def _decode_arguments(self, output):
        argument_dict = {}
        for i, j in itertools.product(range(output["num_triggers_to_keep"]),
                                      range(output["num_argument_spans_to_keep"])):
            trig_ix = output["top_trigger_indices"][i].item()
            arg_span = tuple(output["top_argument_spans"][i].tolist())
            arg_label = output["predicted_arguments"][i, j].item()
            if arg_label >= 0:
                label_name = self.vocab.get_token_from_index(arg_label, namespace="argument_labels")
                argument_dict[(trig_ix, arg_span)] = label_name

        return argument_dict

    def _compute_trigger_scores(self, trigger_embeddings, trigger_mask):
        """
        Compute trigger scores for all tokens.
        """
        trigger_scores = self._trigger_scorer(trigger_embeddings)
        dummy_dims = [trigger_scores.size(0), trigger_scores.size(1), 1]
        dummy_scores = trigger_scores.new_zeros(*dummy_dims)
        trigger_scores = torch.cat((dummy_scores, trigger_scores), -1)
        # Give large negative scores to the masked-out values.
        mask = trigger_mask.unsqueeze(-1)
        trigger_scores = util.replace_masked_values(trigger_scores, mask, -1e20)
        return trigger_scores


    @staticmethod
    def _compute_trig_arg_embeddings(top_trig_embeddings: torch.FloatTensor,
                                     top_arg_embeddings: torch.FloatTensor):
        """
        TODO(dwadden) document me and add comments.
        """
        # TODO(dwadden) this is the same pattern as in the relation module. Maybe this should be
        # refactored somehow?
        num_trigs = top_trig_embeddings.size(1)
        num_args = top_arg_embeddings.size(1)

        trig_emb_expanded = top_trig_embeddings.unsqueeze(2)
        trig_emb_tiled = trig_emb_expanded.repeat(1, 1, num_args, 1)

        arg_emb_expanded = top_arg_embeddings.unsqueeze(1)
        arg_emb_tiled = arg_emb_expanded.repeat(1, num_trigs, 1, 1)

        pair_embeddings_list = [trig_emb_tiled, arg_emb_tiled]
        pair_embeddings = torch.cat(pair_embeddings_list, dim=3)

        return pair_embeddings


    def _compute_argument_scores(self, pairwise_embeddings, top_trig_scores, top_arg_scores):
        batch_size = pairwise_embeddings.size(0)
        max_num_trigs = pairwise_embeddings.size(1)
        max_num_args = pairwise_embeddings.size(2)
        feature_dim = self._argument_feedforward.input_dim

        embeddings_flat = pairwise_embeddings.view(-1, feature_dim)

        arguments_projected_flat = self._argument_feedforward(embeddings_flat)
        argument_scores_flat = self._argument_scorer(arguments_projected_flat)

        argument_scores = argument_scores_flat.view(batch_size, max_num_trigs, max_num_args, -1)

        # Add the mention scores for each of the candidates.

        argument_scores += (top_trig_scores.unsqueeze(-1) +
                            top_arg_scores.transpose(1, 2).unsqueeze(-1))

        shape = [argument_scores.size(0), argument_scores.size(1), argument_scores.size(2), 1]
        dummy_scores = argument_scores.new_zeros(*shape)

        argument_scores = torch.cat([dummy_scores, argument_scores], -1)
        return argument_scores


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
    def _get_pruned_gold_arguments(argument_labels, top_trig_indices, top_arg_indices,
                                   top_trig_masks, top_arg_masks):
        """
        Loop over each slice and get the labels for the spans from that slice.
        All labels are offset by 1 so that the "null" label gets class zero. This is the desired
        behavior for the softmax. Labels corresponding to masked relations keep the label -1, which
        the softmax loss ignores.
        """
        arguments = []

        zipped = zip(argument_labels, top_trig_indices, top_arg_indices,
                     top_trig_masks.byte(), top_arg_masks.byte())

        for sliced, trig_ixs, arg_ixs, trig_mask, arg_mask in zipped:
            entry = sliced[trig_ixs][:, arg_ixs].unsqueeze(0)
            mask_entry = trig_mask & arg_mask.transpose(0, 1).unsqueeze(0)
            entry[mask_entry] += 1
            entry[~mask_entry] = -1
            arguments.append(entry)

        return torch.cat(arguments, dim=0)

    def _get_trigger_loss(self, trigger_scores, trigger_labels, trigger_mask):
        trigger_scores_flat = trigger_scores.view(-1, self._n_trigger_labels)
        trigger_labels_flat = trigger_labels.view(-1)
        mask_flat = trigger_mask.view(-1).byte()

        loss = self._trigger_loss(trigger_scores_flat[mask_flat], trigger_labels_flat[mask_flat])
        return loss

    def _get_argument_loss(self, argument_scores, argument_labels):
        """
        Compute cross-entropy loss on argument labels.
        """
        # Need to add one for the null class.
        scores_flat = argument_scores.view(-1, self._n_argument_labels + 1)
        # Need to add 1 so that the null label is 0, to line up with indices into prediction matrix.
        labels_flat = argument_labels.view(-1)
        # Compute cross-entropy loss.
        loss = self._argument_loss(scores_flat, labels_flat)
        return loss
