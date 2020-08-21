import logging
import itertools
from typing import Any, Dict, List, Optional, Callable

import torch
from torch.nn import functional as F
from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.modules import TimeDistributed
from allennlp.modules.token_embedders import Embedding

from dygie.training.event_metrics import EventMetrics
from dygie.models.shared import fields_to_batches
from dygie.models.entity_beam_pruner import make_pruner
from dygie.data.dataset_readers import document

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


# TODO(dwadden) Different sentences should have different number of relation candidates depending on
# length.
class EventExtractor(Model):
    """
    Event extraction for DyGIE.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 make_feedforward: Callable,
                 token_emb_dim: int,   # Triggers are represented via token embeddings.
                 span_emb_dim: int,    # Arguments are represented via span embeddings.
                 feature_size: int,
                 trigger_spans_per_word: float,
                 argument_spans_per_word: float,
                 loss_weights: Dict[str, float],
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(EventExtractor, self).__init__(vocab, regularizer)

        self._trigger_namespaces = [entry for entry in vocab.get_namespaces()
                                    if "trigger_labels" in entry]
        self._argument_namespaces = [entry for entry in vocab.get_namespaces()
                                     if "argument_labels" in entry]

        self._n_trigger_labels = {name: vocab.get_vocab_size(name)
                                  for name in self._trigger_namespaces}
        self._n_argument_labels = {name: vocab.get_vocab_size(name)
                                   for name in self._argument_namespaces}

        # Make sure the null trigger label is always 0.
        for namespace in self._trigger_namespaces:
            null_label = vocab.get_token_index("", namespace)
            assert null_label == 0  # If not, the dummy class won't correspond to the null label.

        # Create trigger scorers and pruners.
        self._trigger_scorers = torch.nn.ModuleDict()
        self._trigger_pruners = torch.nn.ModuleDict()
        for trigger_namespace in self._trigger_namespaces:
            # The trigger pruner.
            trigger_candidate_feedforward = make_feedforward(input_dim=token_emb_dim)
            self._trigger_pruners[trigger_namespace] = make_pruner(trigger_candidate_feedforward)
            # The trigger scorer.
            trigger_feedforward = make_feedforward(input_dim=token_emb_dim)
            self._trigger_scorers[namespace] = torch.nn.Sequential(
                TimeDistributed(trigger_feedforward),
                TimeDistributed(torch.nn.Linear(trigger_feedforward.get_output_dim(),
                                                self._n_trigger_labels[trigger_namespace] - 1)))

        # Creater argument scorers and pruners.
        self._mention_pruners = torch.nn.ModuleDict()
        self._argument_feedforwards = torch.nn.ModuleDict()
        self._argument_scorers = torch.nn.ModuleDict()
        for argument_namespace in self._argument_namespaces:
            # The argument pruner.
            mention_feedforward = make_feedforward(input_dim=span_emb_dim)
            self._mention_pruners[argument_namespace] = make_pruner(mention_feedforward)
            # The argument scorer. The `+ 2` is there because I include indicator features for
            # whether the trigger is before or inside the arg span.

            # TODO(dwadden) Here
            argument_feedforward_dim = token_emb_dim + span_emb_dim + feature_size + 2
            argument_feedforward = make_feedforward(input_dim=argument_feedforward_dim)
            self._argument_feedforwards[argument_namespace] = argument_feedforward
            self._argument_scorers[argument_namespace] = torch.nn.Linear(
                argument_feedforward.get_output_dim(), self._n_argument_labels[argument_namespace])

        # Weight on trigger labeling and argument labeling.
        self._loss_weights = loss_weights

        # Distance embeddings.
        self._num_distance_buckets = 10  # Just use 10 which is the default.
        self._distance_embedding = Embedding(embedding_dim=feature_size,
                                             num_embeddings=self._num_distance_buckets)

        self._trigger_spans_per_word = trigger_spans_per_word
        self._argument_spans_per_word = argument_spans_per_word

        # Metrics
        # Make a metric for each dataset (not each namespace).
        namespaces = self._trigger_namespaces + self._argument_namespaces
        datasets = set([x.split("__")[0] for x in namespaces])
        self._metrics = {dataset: EventMetrics() for dataset in datasets}

        self._active_namespaces = {"trigger": None, "argument": None}
        self._active_dataset = None

        # Trigger and argument loss.
        self._trigger_loss = torch.nn.CrossEntropyLoss(reduction="sum")
        self._argument_loss = torch.nn.CrossEntropyLoss(reduction="sum", ignore_index=-1)

    ####################

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
                ner_labels,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """
        The trigger embeddings are just the contextualized token embeddings, and the trigger mask is
        the text mask. For the arguments, we consider all the spans.
        """
        self._active_dataset = metadata.dataset
        self._active_namespaces = {"trigger": f"{self._active_dataset}__trigger_labels",
                                   "argument": f"{self._active_dataset}__argument_labels"}

        # Compute trigger scores.
        trigger_scores = self._compute_trigger_scores(
            trigger_embeddings, trigger_mask)

        # Get trigger candidates for event argument labeling.
        num_trigs_to_keep = torch.floor(
            sentence_lengths.float() * self._trigger_spans_per_word).long()
        num_trigs_to_keep = torch.max(num_trigs_to_keep,
                                      torch.ones_like(num_trigs_to_keep))
        num_trigs_to_keep = torch.min(num_trigs_to_keep,
                                      15 * torch.ones_like(num_trigs_to_keep))

        trigger_pruner = self._trigger_pruners[self._active_namespaces["trigger"]]
        (top_trig_embeddings, top_trig_mask,
         top_trig_indices, top_trig_scores, num_trigs_kept) = trigger_pruner(
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
        mention_pruner = self._mention_pruners[self._active_namespaces["argument"]]
        gold_labels = None
        (top_arg_embeddings, top_arg_mask,
         top_arg_indices, top_arg_scores, num_arg_spans_kept) = mention_pruner(
             span_embeddings, span_mask, num_arg_spans_to_keep, gold_labels)

        top_arg_mask = top_arg_mask.unsqueeze(-1)
        top_arg_spans = util.batched_index_select(spans,
                                                  top_arg_indices)

        # Compute trigger / argument pair embeddings.
        trig_arg_embeddings = self._compute_trig_arg_embeddings(
            top_trig_embeddings, top_arg_embeddings, top_trig_indices, top_arg_spans)
        argument_scores = self._compute_argument_scores(
            trig_arg_embeddings, top_trig_scores, top_arg_scores, top_arg_mask)

        # Assemble inputs to do prediction.
        output_dict = {"top_trigger_indices": top_trig_indices,
                       "top_argument_spans": top_arg_spans,
                       "trigger_scores": trigger_scores,
                       "argument_scores": argument_scores,
                       "num_triggers_kept": num_trigs_kept,
                       "num_argument_spans_kept": num_arg_spans_kept,
                       "sentence_lengths": sentence_lengths}

        prediction_dicts, predictions = self.predict(output_dict, metadata)

        output_dict = {"predictions": predictions}

        # Evaluate loss and F1 if labels were provided.
        if trigger_labels is not None and argument_labels is not None:
            # Compute the loss for both triggers and arguments.
            trigger_loss = self._get_trigger_loss(trigger_scores, trigger_labels, trigger_mask)

            gold_arguments = self._get_pruned_gold_arguments(
                argument_labels, top_trig_indices, top_arg_indices, top_trig_mask, top_arg_mask)

            argument_loss = self._get_argument_loss(argument_scores, gold_arguments)

            # Compute F1.
            assert len(prediction_dicts) == len(metadata)  # Make sure length of predictions is right.

            # Compute metrics for this label namespace.
            metrics = self._metrics[self._active_dataset]
            metrics(prediction_dicts, metadata)

            loss = (self._loss_weights["trigger"] * trigger_loss +
                    self._loss_weights["arguments"] * argument_loss)

            output_dict["loss"] = loss

        return output_dict

    ####################

    # Embeddings

    def _compute_trig_arg_embeddings(self,
                                     top_trig_embeddings,
                                     top_arg_embeddings,
                                     top_trig_indices,
                                     top_arg_spans):
        """
        Create trigger / argument pair embeddings, consisting of:
        - The embeddings of the trigger and argument pair.
        - Optionally, the embeddings of the trigger and argument labels.
        - Optionally, embeddings of the words surrounding the trigger and argument.
        """
        num_trigs = top_trig_embeddings.size(1)
        num_args = top_arg_embeddings.size(1)

        trig_emb_expanded = top_trig_embeddings.unsqueeze(2)
        trig_emb_tiled = trig_emb_expanded.repeat(1, 1, num_args, 1)

        arg_emb_expanded = top_arg_embeddings.unsqueeze(1)
        arg_emb_tiled = arg_emb_expanded.repeat(1, num_trigs, 1, 1)

        distance_embeddings = self._compute_distance_embeddings(top_trig_indices, top_arg_spans)

        pair_embeddings_list = [trig_emb_tiled, arg_emb_tiled, distance_embeddings]
        pair_embeddings = torch.cat(pair_embeddings_list, dim=3)

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

    ####################

    # Scorers

    def _compute_trigger_scores(self, trigger_embeddings, trigger_mask):
        """
        Compute trigger scores for all tokens.
        """
        trigger_scorer = self._trigger_scorers[self._active_namespaces["trigger"]]
        trigger_scores = trigger_scorer(trigger_embeddings)
        # Give large negative scores to masked-out elements.
        mask = trigger_mask.unsqueeze(-1)
        trigger_scores = util.replace_masked_values(trigger_scores, mask.bool(), -1e20)
        dummy_dims = [trigger_scores.size(0), trigger_scores.size(1), 1]
        dummy_scores = trigger_scores.new_zeros(*dummy_dims)
        trigger_scores = torch.cat((dummy_scores, trigger_scores), -1)
        # Give large negative scores to the masked-out values.
        return trigger_scores

    def _compute_argument_scores(self, pairwise_embeddings, top_trig_scores, top_arg_scores,
                                 top_arg_mask, prepend_zeros=True):
        batch_size = pairwise_embeddings.size(0)
        max_num_trigs = pairwise_embeddings.size(1)
        max_num_args = pairwise_embeddings.size(2)
        argument_feedforward = self._argument_feedforwards[self._active_namespaces["argument"]]

        feature_dim = argument_feedforward.input_dim
        embeddings_flat = pairwise_embeddings.view(-1, feature_dim)

        arguments_projected_flat = argument_feedforward(embeddings_flat)

        argument_scorer = self._argument_scorers[self._active_namespaces["argument"]]
        argument_scores_flat = argument_scorer(arguments_projected_flat)

        argument_scores = argument_scores_flat.view(batch_size, max_num_trigs, max_num_args, -1)

        # Add the mention scores for each of the candidates.

        argument_scores += (top_trig_scores.unsqueeze(-1) +
                            top_arg_scores.transpose(1, 2).unsqueeze(-1))

        shape = [argument_scores.size(0), argument_scores.size(1), argument_scores.size(2), 1]
        dummy_scores = argument_scores.new_zeros(*shape)

        if prepend_zeros:
            argument_scores = torch.cat([dummy_scores, argument_scores], -1)
        return argument_scores

    ####################

    # Predictions / decoding.

    def predict(self, output_dict, document):
        """
        Take the output and convert it into a list of dicts. Each entry is a sentence. Each key is a
        pair of span indices for that sentence, and each value is the relation label on that span
        pair.
        """
        outputs = fields_to_batches({k: v.detach().cpu() for k, v in output_dict.items()})

        prediction_dicts = []
        predictions = []

        # Collect predictions for each sentence in minibatch.
        for output, sentence in zip(outputs, document):
            decoded_trig = self._decode_trigger(output)
            decoded_args = self._decode_arguments(output, decoded_trig)
            predicted_events = self._assemble_predictions(decoded_trig, decoded_args, sentence)
            prediction_dicts.append({"trigger_dict": decoded_trig, "argument_dict": decoded_args})
            predictions.append(predicted_events)

        return prediction_dicts, predictions

    def _decode_trigger(self, output):
        trigger_scores = output["trigger_scores"]
        predicted_scores_raw, predicted_triggers = trigger_scores.max(dim=1)
        softmax_scores = F.softmax(trigger_scores, dim=1)
        predicted_scores_softmax, _ = softmax_scores.max(dim=1)
        trigger_dict = {}
        # TODO(dwadden) Can speed this up with array ops.
        for i in range(output["sentence_lengths"]):
            trig_label = predicted_triggers[i].item()
            if trig_label > 0:
                predicted_label = self.vocab.get_token_from_index(
                    trig_label, namespace=self._active_namespaces["trigger"])
                trigger_dict[i] = (predicted_label,
                                   predicted_scores_raw[i].item(),
                                   predicted_scores_softmax[i].item())

        return trigger_dict

    def _decode_arguments(self, output, decoded_trig):
        # TODO(dwadden) Vectorize.
        argument_dict = {}
        argument_scores = output["argument_scores"]
        predicted_scores_raw, predicted_arguments = argument_scores.max(dim=-1)
        # The null argument has label -1.
        predicted_arguments -= 1
        softmax_scores = F.softmax(argument_scores, dim=-1)
        predicted_scores_softmax, _ = softmax_scores.max(dim=-1)

        for i, j in itertools.product(range(output["num_triggers_kept"]),
                                      range(output["num_argument_spans_kept"])):
            trig_ix = output["top_trigger_indices"][i].item()
            arg_span = tuple(output["top_argument_spans"][j].tolist())
            arg_label = predicted_arguments[i, j].item()
            # Only include the argument if its putative trigger is predicted as a real trigger.
            if arg_label >= 0 and trig_ix in decoded_trig:
                arg_score_raw = predicted_scores_raw[i, j].item()
                arg_score_softmax = predicted_scores_softmax[i, j].item()
                label_name = self.vocab.get_token_from_index(
                    arg_label, namespace=self._active_namespaces["argument"])
                argument_dict[(trig_ix, arg_span)] = (label_name, arg_score_raw, arg_score_softmax)

        return argument_dict

    def _assemble_predictions(self, trigger_dict, argument_dict, sentence):
        events_json = []
        for trigger_ix, trigger_label in trigger_dict.items():
            this_event = []
            this_event.append([trigger_ix] + list(trigger_label))
            event_arguments = {k: v for k, v in argument_dict.items() if k[0] == trigger_ix}
            this_event_args = []
            for k, v in event_arguments.items():
                entry = list(k[1]) + list(v)
                this_event_args.append(entry)
            this_event_args = sorted(this_event_args, key=lambda entry: entry[0])
            this_event.extend(this_event_args)
            events_json.append(this_event)

        events = document.PredictedEvents(events_json, sentence, sentence_offsets=True)

        return events

    ####################

    # Loss function and evaluation metrics.

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
        n_trigger_labels = self._n_trigger_labels[self._active_namespaces["trigger"]]
        trigger_scores_flat = trigger_scores.view(-1, n_trigger_labels)
        trigger_labels_flat = trigger_labels.view(-1)
        mask_flat = trigger_mask.view(-1).bool()

        loss = self._trigger_loss(trigger_scores_flat[mask_flat], trigger_labels_flat[mask_flat])
        return loss

    def _get_argument_loss(self, argument_scores, argument_labels):
        """
        Compute cross-entropy loss on argument labels.
        """
        n_argument_labels = self._n_argument_labels[self._active_namespaces["argument"]]
        # Need to add one for the null class.
        scores_flat = argument_scores.view(-1, n_argument_labels + 1)
        # Need to add 1 so that the null label is 0, to line up with indices into prediction matrix.
        labels_flat = argument_labels.view(-1)
        # Compute cross-entropy loss.
        loss = self._argument_loss(scores_flat, labels_flat)
        return loss

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        res = {}
        for namespace, metrics in self._metrics.items():
            f1_metrics = metrics.get_metric(reset)
            f1_metrics = {f"{namespace}_{k}": v for k, v in f1_metrics.items()}
            res.update(f1_metrics)

        prod = itertools.product(["trig_id", "trig_class", "arg_id", "arg_class"],
                                 ["precision", "recall", "f1"])
        names = [f"{task}_{metric}" for task, metric in prod]

        res_avg = {}
        for name in names:
            values = [res[key] for key in res if name in key]
            res_avg[f"MEAN__{name}"] = sum(values) / len(values) if values else 0
            res.update(res_avg)

        return res
