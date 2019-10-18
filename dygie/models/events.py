import logging
from os import path
import itertools
from typing import Any, Dict, List, Optional

import torch
from torch.nn import functional as F
from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2SeqEncoder
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.modules import TimeDistributed
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.matrix_attention.bilinear_matrix_attention import BilinearMatrixAttention

from dygie.training.relation_metrics import RelationMetrics, CandidateRecall
from dygie.training.event_metrics import EventMetrics, ArgumentStats
from dygie.models.shared import fields_to_batches
from dygie.models.one_hot import make_embedder
from dygie.models.entity_beam_pruner import make_pruner
from dygie.models.span_prop import SpanProp


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
                 trigger_candidate_feedforward: FeedForward,
                 mention_feedforward: FeedForward,  # Used if entity beam is off.
                 argument_feedforward: FeedForward,
                 context_attention: BilinearMatrixAttention,
                 trigger_attention: Seq2SeqEncoder,
                 span_prop: SpanProp,
                 cls_projection: FeedForward,
                 feature_size: int,
                 trigger_spans_per_word: float,
                 argument_spans_per_word: float,
                 loss_weights,
                 trigger_attention_context: bool,
                 event_args_use_trigger_labels: bool,
                 event_args_use_ner_labels: bool,
                 event_args_label_emb: int,
                 shared_attention_context: bool,
                 label_embedding_method: str,
                 event_args_label_predictor: str,
                 event_args_gold_candidates: bool = False,  # If True, use gold argument candidates.
                 context_window: int = 0,
                 softmax_correction: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 positive_label_weight: float = 1.0,
                 entity_beam: bool = False,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(EventExtractor, self).__init__(vocab, regularizer)

        self._n_ner_labels = vocab.get_vocab_size("ner_labels")
        self._n_trigger_labels = vocab.get_vocab_size("trigger_labels")
        self._n_argument_labels = vocab.get_vocab_size("argument_labels")

        # Embeddings for trigger labels and ner labels, to be used by argument scorer.
        # These will be either one-hot encodings or learned embeddings, depending on "kind".
        self._ner_label_emb = make_embedder(kind=label_embedding_method,
                                            num_embeddings=self._n_ner_labels,
                                            embedding_dim=event_args_label_emb)
        self._trigger_label_emb = make_embedder(kind=label_embedding_method,
                                                num_embeddings=self._n_trigger_labels,
                                                embedding_dim=event_args_label_emb)
        self._label_embedding_method = label_embedding_method

        # Weight on trigger labeling and argument labeling.
        self._loss_weights = loss_weights

        # Trigger candidate scorer.
        null_label = vocab.get_token_index("", "trigger_labels")
        assert null_label == 0  # If not, the dummy class won't correspond to the null label.

        self._trigger_scorer = torch.nn.Sequential(
            TimeDistributed(trigger_feedforward),
            TimeDistributed(torch.nn.Linear(trigger_feedforward.get_output_dim(),
                                            self._n_trigger_labels - 1)))

        self._trigger_attention_context = trigger_attention_context
        if self._trigger_attention_context:
            self._trigger_attention = trigger_attention

        # Make pruners. If `entity_beam` is true, use NER and trigger scorers to construct the beam
        # and only keep candidates that the model predicts are actual entities or triggers.
        self._mention_pruner = make_pruner(mention_feedforward, entity_beam=entity_beam,
                                           gold_beam=event_args_gold_candidates)
        self._trigger_pruner = make_pruner(trigger_candidate_feedforward, entity_beam=entity_beam,
                                           gold_beam=False)

        # Argument scorer.
        self._event_args_use_trigger_labels = event_args_use_trigger_labels  # If True, use trigger labels.
        self._event_args_use_ner_labels = event_args_use_ner_labels  # If True, use ner labels to predict args.
        assert event_args_label_predictor in ["hard", "softmax", "gold"]  # Method for predicting labels at test time.
        self._event_args_label_predictor = event_args_label_predictor
        self._event_args_gold_candidates = event_args_gold_candidates
        # If set to True, then construct a context vector from a bilinear attention over the trigger
        # / argument pair embeddings and the text.
        self._context_window = context_window                # If greater than 0, concatenate context as features.
        self._argument_feedforward = argument_feedforward
        self._argument_scorer = torch.nn.Linear(argument_feedforward.get_output_dim(), self._n_argument_labels)

        # Distance embeddings.
        self._num_distance_buckets = 10  # Just use 10 which is the default.
        self._distance_embedding = Embedding(self._num_distance_buckets, feature_size)

        # Class token projection.
        self._cls_projection = cls_projection
        self._cls_n_triggers = torch.nn.Linear(self._cls_projection.get_output_dim(), 5)
        self._cls_event_types = torch.nn.Linear(self._cls_projection.get_output_dim(),
                                                self._n_trigger_labels - 1)

        self._trigger_spans_per_word = trigger_spans_per_word
        self._argument_spans_per_word = argument_spans_per_word

        # Context attention for event argument scorer.
        self._shared_attention_context = shared_attention_context
        if self._shared_attention_context:
            self._shared_attention_context_module = context_attention

        # Span propagation object.
        # TODO(dwadden) initialize with `from_params` instead if this ends up working.
        self._span_prop = span_prop
        self._span_prop._trig_arg_embedder = self._compute_trig_arg_embeddings
        self._span_prop._argument_scorer = self._compute_argument_scores

        # Softmax correction parameters.
        self._softmax_correction = softmax_correction
        self._softmax_log_temp = torch.nn.Parameter(
            torch.zeros([1, 1, 1, self._n_argument_labels]))
        self._softmax_log_multiplier = torch.nn.Parameter(
            torch.zeros([1, 1, 1, self._n_argument_labels]))

        # TODO(dwadden) Add metrics.
        self._metrics = EventMetrics()
        self._argument_stats = ArgumentStats()

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
                cls_embeddings,
                sentence_lengths,
                output_ner,     # Needed if we're using entity beam approach.
                trigger_labels,
                argument_labels,
                ner_labels,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """
        TODO(dwadden) Write documentation.
        The trigger embeddings are just the contextualized token embeddings, and the trigger mask is
        the text mask. For the arguments, we consider all the spans.
        """
        cls_projected = self._cls_projection(cls_embeddings)
        auxiliary_loss = self._compute_auxiliary_loss(cls_projected, trigger_labels, trigger_mask)

        ner_scores = output_ner["ner_scores"]
        predicted_ner = output_ner["predicted_ner"]

        # Compute trigger scores.
        trigger_scores = self._compute_trigger_scores(trigger_embeddings, cls_projected, trigger_mask)
        _, predicted_triggers = trigger_scores.max(-1)

        # Get trigger candidates for event argument labeling.
        num_trigs_to_keep = torch.floor(
            sentence_lengths.float() * self._trigger_spans_per_word).long()
        num_trigs_to_keep = torch.max(num_trigs_to_keep,
                                      torch.ones_like(num_trigs_to_keep))
        num_trigs_to_keep = torch.min(num_trigs_to_keep,
                                      15 * torch.ones_like(num_trigs_to_keep))

        (top_trig_embeddings, top_trig_mask,
         top_trig_indices, top_trig_scores, num_trigs_kept) = self._trigger_pruner(
             trigger_embeddings, trigger_mask, num_trigs_to_keep, trigger_scores)
        top_trig_mask = top_trig_mask.unsqueeze(-1)

        # Compute the number of argument spans to keep.
        num_arg_spans_to_keep = torch.floor(
            sentence_lengths.float() * self._argument_spans_per_word).long()
        num_arg_spans_to_keep = torch.max(num_arg_spans_to_keep,
                                          torch.ones_like(num_arg_spans_to_keep))
        num_arg_spans_to_keep = torch.min(num_arg_spans_to_keep,
                                          30 * torch.ones_like(num_arg_spans_to_keep))

        # If we're using gold event arguments, include the gold labels.
        gold_labels = ner_labels if self._event_args_gold_candidates else None
        (top_arg_embeddings, top_arg_mask,
         top_arg_indices, top_arg_scores, num_arg_spans_kept) = self._mention_pruner(
             span_embeddings, span_mask, num_arg_spans_to_keep, ner_scores, gold_labels)

        top_arg_mask = top_arg_mask.unsqueeze(-1)
        top_arg_spans = util.batched_index_select(spans,
                                                  top_arg_indices)

        # Collect trigger and ner labels, in case they're included as features to the argument
        # classifier.
        # At train time, use the gold labels. At test time, use the labels predicted by the model,
        # or gold if specified.
        if self.training or self._event_args_label_predictor == "gold":
            top_trig_labels = trigger_labels.gather(1, top_trig_indices)
            top_ner_labels = ner_labels.gather(1, top_arg_indices)
        else:
            # Hard predictions.
            if self._event_args_label_predictor == "hard":
                top_trig_labels = predicted_triggers.gather(1, top_trig_indices)
                top_ner_labels = predicted_ner.gather(1, top_arg_indices)
            # Softmax predictions.
            else:
                softmax_triggers = trigger_scores.softmax(dim=-1)
                top_trig_labels = util.batched_index_select(softmax_triggers, top_trig_indices)
                softmax_ner = ner_scores.softmax(dim=-1)
                top_ner_labels = util.batched_index_select(softmax_ner, top_arg_indices)

        # Make a dict of all arguments that are needed to make trigger / argument pair embeddings.
        trig_arg_emb_dict = dict(cls_projected=cls_projected,
                                 top_trig_labels=top_trig_labels,
                                 top_ner_labels=top_ner_labels,
                                 top_trig_indices=top_trig_indices,
                                 top_arg_spans=top_arg_spans,
                                 text_emb=trigger_embeddings,
                                 text_mask=trigger_mask)

        # Run span graph propagation, if asked for
        if self._span_prop._n_span_prop > 0:
            top_trig_embeddings, top_arg_embeddings = self._span_prop(
                top_trig_embeddings, top_arg_embeddings, top_trig_mask, top_arg_mask,
                top_trig_scores, top_arg_scores, trig_arg_emb_dict)

            top_trig_indices_repeat = (top_trig_indices.unsqueeze(-1).
                                       repeat(1, 1, top_trig_embeddings.size(-1)))
            updated_trig_embeddings = trigger_embeddings.scatter(
                1, top_trig_indices_repeat, top_trig_embeddings)

            # Recompute the trigger scores.
            trigger_scores = self._compute_trigger_scores(updated_trig_embeddings, cls_projected, trigger_mask)
            _, predicted_triggers = trigger_scores.max(-1)

        trig_arg_embeddings = self._compute_trig_arg_embeddings(
            top_trig_embeddings, top_arg_embeddings, **trig_arg_emb_dict)
        argument_scores = self._compute_argument_scores(
            trig_arg_embeddings, top_trig_scores, top_arg_scores, top_arg_mask)

        _, predicted_arguments = argument_scores.max(-1)
        predicted_arguments -= 1  # The null argument has label -1.

        output_dict = {"top_trigger_indices": top_trig_indices,
                       "top_argument_spans": top_arg_spans,
                       "trigger_scores": trigger_scores,
                       "argument_scores": argument_scores,
                       "predicted_triggers": predicted_triggers,
                       "predicted_arguments": predicted_arguments,
                       "num_triggers_kept": num_trigs_kept,
                       "num_argument_spans_kept": num_arg_spans_kept,
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
            self._argument_stats(predictions)

            loss = (self._loss_weights["trigger"] * trigger_loss +
                    self._loss_weights["arguments"] * argument_loss +
                    0.05 * auxiliary_loss)

            output_dict["loss"] = loss

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
            decoded_args, decoded_args_with_scores = self._decode_arguments(output, decoded_trig)
            entry = dict(trigger_dict=decoded_trig, argument_dict=decoded_args,
                         argument_dict_with_scores=decoded_args_with_scores)
            res.append(entry)

        output_dict["decoded_events"] = res
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        f1_metrics = self._metrics.get_metric(reset)
        argument_stats = self._argument_stats.get_metric(reset)
        res = {}
        res.update(f1_metrics)
        res.update(argument_stats)
        return res

    def _decode_trigger(self, output):
        trigger_dict = {}
        for i in range(output["sentence_lengths"]):
            trig_label = output["predicted_triggers"][i].item()
            if trig_label > 0:
                trigger_dict[i] = self.vocab.get_token_from_index(trig_label, namespace="trigger_labels")

        return trigger_dict

    def _decode_arguments(self, output, decoded_trig):
        argument_dict = {}
        argument_dict_with_scores = {}
        for i, j in itertools.product(range(output["num_triggers_kept"]),
                                      range(output["num_argument_spans_kept"])):
            trig_ix = output["top_trigger_indices"][i].item()
            arg_span = tuple(output["top_argument_spans"][j].tolist())
            arg_label = output["predicted_arguments"][i, j].item()
            # Only include the argument if its putative trigger is predicted as a real trigger.
            if arg_label >= 0 and trig_ix in decoded_trig:
                arg_score = output["argument_scores"][i, j, arg_label + 1].item()
                label_name = self.vocab.get_token_from_index(arg_label, namespace="argument_labels")
                argument_dict[(trig_ix, arg_span)] = label_name
                # Keep around a version with the predicted labels and their scores, for debugging
                # purposes.
                argument_dict_with_scores[(trig_ix, arg_span)] = (label_name, arg_score)

        return argument_dict, argument_dict_with_scores

    def _compute_auxiliary_loss(self, cls_projected, trigger_labels, trigger_mask):
        num_triggers = ((trigger_labels > 0) * trigger_mask.bool()).sum(dim=1)
        # Truncate at 4.
        num_triggers = torch.min(num_triggers, 4 * torch.ones_like(num_triggers))
        predicted_num_triggers = self._cls_n_triggers(cls_projected)
        num_trigger_loss = F.cross_entropy(
            predicted_num_triggers, num_triggers,
            weight=torch.tensor([1, 3, 3, 3, 3], device=trigger_labels.device, dtype=torch.float),
            reduction="sum")

        label_present = [torch.any(trigger_labels == i, dim=1).unsqueeze(1)
                         for i in range(1, self._n_trigger_labels)]
        label_present = torch.cat(label_present, dim=1)
        if cls_projected.device.type != "cpu":
            label_present = label_present.cuda(cls_projected.device)
        predicted_event_type_logits = self._cls_event_types(cls_projected)
        trigger_label_loss = F.binary_cross_entropy_with_logits(
            predicted_event_type_logits, label_present.float(), reduction="sum")

        return num_trigger_loss + trigger_label_loss

    def _compute_trigger_scores(self, trigger_embeddings, cls_projected, trigger_mask):
        """
        Compute trigger scores for all tokens.
        """
        cls_repeat = cls_projected.unsqueeze(dim=1).repeat(1, trigger_embeddings.size(1), 1)
        trigger_embeddings = torch.cat([trigger_embeddings, cls_repeat], dim=-1)
        if self._trigger_attention_context:
            context = self._trigger_attention(trigger_embeddings, trigger_mask)
            trigger_embeddings = torch.cat([trigger_embeddings, context], dim=2)
        trigger_scores = self._trigger_scorer(trigger_embeddings)
        # Give large negative scores to masked-out elements.
        mask = trigger_mask.unsqueeze(-1)
        trigger_scores = util.replace_masked_values(trigger_scores, mask, -1e20)
        dummy_dims = [trigger_scores.size(0), trigger_scores.size(1), 1]
        dummy_scores = trigger_scores.new_zeros(*dummy_dims)
        trigger_scores = torch.cat((dummy_scores, trigger_scores), -1)
        # Give large negative scores to the masked-out values.
        return trigger_scores

    def _compute_trig_arg_embeddings(self,
                                     top_trig_embeddings, top_arg_embeddings, cls_projected,
                                     top_trig_labels, top_ner_labels, top_trig_indices,
                                     top_arg_spans, text_emb, text_mask):
        """
        Create trigger / argument pair embeddings, consisting of:
        - The embeddings of the trigger and argument pair.
        - Optionally, the embeddings of the trigger and argument labels.
        - Optionally, embeddings of the words surrounding the trigger and argument.
        """
        trig_emb_extras = []
        arg_emb_extras = []

        if self._context_window > 0:
            # Include words in a window around trigger and argument.
            # For triggers, the span start and end indices are the same.
            trigger_context = self._get_context(top_trig_indices, top_trig_indices, text_emb)
            argument_context = self._get_context(
                top_arg_spans[:, :, 0], top_arg_spans[:, :, 1], text_emb)
            trig_emb_extras.append(trigger_context)
            arg_emb_extras.append(argument_context)

        # TODO(dwadden) refactor this. Way too many conditionals.
        if self._event_args_use_trigger_labels:
            if self._event_args_label_predictor == "softmax" and not self.training:
                if self._label_embedding_method == "one_hot":
                    # If we're using one-hot encoding, just return the scores for each class.
                    top_trig_embs = top_trig_labels
                else:
                    # Otherwise take the average of the embeddings, weighted by softmax scores.
                    top_trig_embs = torch.matmul(top_trig_labels, self._trigger_label_emb.weight)
                trig_emb_extras.append(top_trig_embs)
            else:
                trig_emb_extras.append(self._trigger_label_emb(top_trig_labels))
        if self._event_args_use_ner_labels:
            if self._event_args_label_predictor == "softmax" and not self.training:
                # Same deal as for trigger labels.
                if self._label_embedding_method == "one_hot":
                    top_ner_embs = top_ner_labels
                else:
                    top_ner_embs = torch.matmul(top_ner_labels, self._ner_label_emb.weight)
                arg_emb_extras.append(top_ner_embs)
            else:
                # Otherwise, just return the embeddings.
                arg_emb_extras.append(self._ner_label_emb(top_ner_labels))

        num_trigs = top_trig_embeddings.size(1)
        num_args = top_arg_embeddings.size(1)

        trig_emb_expanded = top_trig_embeddings.unsqueeze(2)
        trig_emb_tiled = trig_emb_expanded.repeat(1, 1, num_args, 1)

        arg_emb_expanded = top_arg_embeddings.unsqueeze(1)
        arg_emb_tiled = arg_emb_expanded.repeat(1, num_trigs, 1, 1)

        distance_embeddings = self._compute_distance_embeddings(top_trig_indices, top_arg_spans)

        cls_repeat = (cls_projected.unsqueeze(dim=1).unsqueeze(dim=2).
                      repeat(1, num_trigs, num_args, 1))

        pair_embeddings_list = [trig_emb_tiled, arg_emb_tiled, distance_embeddings, cls_repeat]
        pair_embeddings = torch.cat(pair_embeddings_list, dim=3)

        if trig_emb_extras:
            trig_extras_expanded = torch.cat(trig_emb_extras, dim=-1).unsqueeze(2)
            trig_extras_tiled = trig_extras_expanded.repeat(1, 1, num_args, 1)
            pair_embeddings = torch.cat([pair_embeddings, trig_extras_tiled], dim=3)

        if arg_emb_extras:
            arg_extras_expanded = torch.cat(arg_emb_extras, dim=-1).unsqueeze(1)
            arg_extras_tiled = arg_extras_expanded.repeat(1, num_trigs, 1, 1)
            pair_embeddings = torch.cat([pair_embeddings, arg_extras_tiled], dim=3)

        if self._shared_attention_context:
            attended_context = self._get_shared_attention_context(pair_embeddings, text_emb, text_mask)
            pair_embeddings = torch.cat([pair_embeddings, attended_context], dim=3)

        return pair_embeddings

    def _compute_distance_embeddings(self, top_trig_indices, top_arg_spans):
        top_trig_ixs = top_trig_indices.unsqueeze(2)
        arg_span_starts = top_arg_spans[:, :, 0].unsqueeze(1)
        arg_span_ends = top_arg_spans[:, :, 1].unsqueeze(1)
        dist_from_start = top_trig_ixs - arg_span_starts
        dist_from_end = top_trig_ixs - arg_span_ends
        # Distance from trigger to arg.
        dist = torch.min(dist_from_start.abs(), dist_from_end.abs())
        # When the trigger is inside the arg span, also set the distance to zero.
        trigger_inside = (top_trig_ixs >= arg_span_starts) & (top_trig_ixs <= arg_span_ends)
        dist[trigger_inside] = 0
        dist_buckets = util.bucket_values(dist, self._num_distance_buckets)
        dist_emb = self._distance_embedding(dist_buckets)
        trigger_before_feature = (top_trig_ixs < arg_span_starts).float().unsqueeze(-1)
        trigger_inside_feature = trigger_inside.float().unsqueeze(-1)
        res = torch.cat([dist_emb, trigger_before_feature, trigger_inside_feature], dim=-1)

        return res

    def _get_shared_attention_context(self, pair_embeddings, text_emb, text_mask):
        batch_size, n_triggers, n_args, emb_dim = pair_embeddings.size()
        pair_emb_flat = pair_embeddings.view([batch_size, -1, emb_dim])
        attn_unnorm = self._shared_attention_context_module(pair_emb_flat, text_emb)
        attn_weights = util.masked_softmax(attn_unnorm, text_mask)
        context = util.weighted_sum(text_emb, attn_weights)
        context = context.view(batch_size, n_triggers, n_args, -1)

        return context

    def _get_context(self, span_starts, span_ends, text_emb):
        """
        Given span start and end (inclusive), get the context on either side.
        """
        # The text_emb are already zero-padded on the right, which is correct.
        assert span_starts.size() == span_ends.size()
        batch_size, seq_length, emb_size = text_emb.size()
        num_candidates = span_starts.size(1)
        padding = torch.zeros(batch_size, self._context_window, emb_size, device=text_emb.device)
        # [batch_size, seq_length + 2 x context_window, emb_size]
        padded_emb = torch.cat([padding, text_emb, padding], dim=1)

        pad_batch = []
        for batch_ix, (start_ixs, end_ixs) in enumerate(zip(span_starts, span_ends)):
            pad_entry = []
            for start_ix, end_ix in zip(start_ixs, end_ixs):
                # The starts are inclusive, ends are exclusive.
                left_start = start_ix
                left_end = start_ix + self._context_window
                right_start = end_ix + self._context_window + 1
                right_end = end_ix + 2 * self._context_window + 1
                left_pad = padded_emb[batch_ix, left_start:left_end]
                right_pad = padded_emb[batch_ix, right_start:right_end]
                pad = torch.cat([left_pad, right_pad], dim=0).view(-1).unsqueeze(0)
                pad_entry.append(pad)

            pad_entry = torch.cat(pad_entry, dim=0).unsqueeze(0)
            pad_batch.append(pad_entry)

        pad_batch = torch.cat(pad_batch, dim=0)

        return pad_batch

    def _compute_argument_scores(self, pairwise_embeddings, top_trig_scores, top_arg_scores,
                                 top_arg_mask, prepend_zeros=True):
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

        # Softmax correction to compare arguments.
        if self._softmax_correction:
            the_temp = torch.exp(self._softmax_log_temp)
            the_multiplier = torch.exp(self._softmax_log_multiplier)
            softmax_scores = util.masked_softmax(argument_scores / the_temp, mask=top_arg_mask, dim=2)
            argument_scores = argument_scores + the_multiplier * softmax_scores

        shape = [argument_scores.size(0), argument_scores.size(1), argument_scores.size(2), 1]
        dummy_scores = argument_scores.new_zeros(*shape)

        if prepend_zeros:
            argument_scores = torch.cat([dummy_scores, argument_scores], -1)
        return argument_scores

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
                     top_trig_masks.bool(), top_arg_masks.bool())

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
        mask_flat = trigger_mask.view(-1).bool()

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
