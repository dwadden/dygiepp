from overrides import overrides
from collections import Counter

from allennlp.training.metrics.metric import Metric

from dygie.training.f1 import compute_f1


def _invert_arguments(arguments, triggers):
    """
    For scoring the argument, we don't need the trigger spans to match exactly. We just need the
    trigger label corresponding to the predicted trigger span to be correct.
    """
    # Can't use a dict because multiple triggers could share the same argument.
    inverted = set()
    for k, v in arguments.items():
        if k[0] in triggers:  # If it's not, the trigger this arg points to is null. TODO(dwadden) check.
            trigger_label = triggers[k[0]]
            to_append = (k[1], trigger_label, v)
            inverted.add(to_append)

    return inverted


class EventMetrics(Metric):
    """
    Computes precision, recall, and micro-averaged F1 for triggers and arguments.
    """
    def __init__(self):
        self.reset()

    @overrides
    def __call__(self, predicted_events_list, metadata_list):
        for predicted_events, metadata in zip(predicted_events_list, metadata_list):
            # Trigger scoring.
            predicted_triggers = predicted_events["trigger_dict"]
            gold_triggers = metadata["trigger_dict"]
            self._score_triggers(predicted_triggers, gold_triggers)

            # Argument scoring.
            predicted_arguments = _invert_arguments(predicted_events["argument_dict"], predicted_triggers)
            gold_arguments = _invert_arguments(metadata["argument_dict"], gold_triggers)
            self._score_arguments(predicted_arguments, gold_arguments)

    def _score_triggers(self, predicted_triggers, gold_triggers):
        self._gold_triggers += len(gold_triggers)
        self._predicted_triggers += len(predicted_triggers)
        for token_ix, label in predicted_triggers.items():
            # Check whether the offsets match, and whether the labels match.
            if token_ix in gold_triggers:
                self._matched_trigger_ids += 1
                if gold_triggers[token_ix] == label:
                    self._matched_trigger_classes += 1

    def _score_arguments(self, predicted_arguments, gold_arguments):
        self._gold_arguments += len(gold_arguments)
        self._predicted_arguments += len(predicted_arguments)
        for prediction in predicted_arguments:
            ix, trigger, arg = prediction
            gold_id_matches = {entry for entry in gold_arguments
                               if entry[0] == ix
                               and entry[1] == trigger}
            if gold_id_matches:
                self._matched_argument_ids += 1
                gold_class_matches = {entry for entry in gold_id_matches if entry[2] == arg}
                if gold_class_matches:
                    self._matched_argument_classes += 1

    @overrides
    def get_metric(self, reset=False):
        res = {}

        # Triggers
        res["trig_id_precision"], res["trig_id_recall"], res["trig_id_f1"] = compute_f1(
            self._predicted_triggers, self._gold_triggers, self._matched_trigger_ids)
        res["trig_class_precision"], res["trig_class_recall"], res["trig_class_f1"] = compute_f1(
            self._predicted_triggers, self._gold_triggers, self._matched_trigger_classes)

        # Arguments
        res["arg_id_precision"], res["arg_id_recall"], res["arg_id_f1"] = compute_f1(
            self._predicted_arguments, self._gold_arguments, self._matched_argument_ids)
        res["arg_class_precision"], res["arg_class_recall"], res["arg_class_f1"] = compute_f1(
            self._predicted_arguments, self._gold_arguments, self._matched_argument_classes)

        # Reset counts if at end of epoch.
        if reset:
            self.reset()

        return res

    @overrides
    def reset(self):
        self._gold_triggers = 0
        self._predicted_triggers = 0
        self._matched_trigger_ids = 0
        self._matched_trigger_classes = 0
        self._gold_arguments = 0
        self._predicted_arguments = 0
        self._matched_argument_ids = 0
        self._matched_argument_classes = 0


class ArgumentStats(Metric):
    """
    Compute the fraction of predicted event arguments that are associated with multiple triggers.
    """
    def __init__(self):
        self.reset()

    @overrides
    def __call__(self, predicted_events_list):
        for predicted_events in predicted_events_list:
            predicted_arguments = _invert_arguments(predicted_events["argument_dict"],
                                                    predicted_events["trigger_dict"])
            # Count how many times each span appears as an argument.
            span_counts = Counter()
            for prediction in predicted_arguments:
                span_counts[prediction[0]] += 1
            # Count how many spans appear more than once.
            repeated = {k: v for k, v in span_counts.items() if v > 1}
            self._total_arguments += len(span_counts)
            self._repeated_arguments += len(repeated)

    @overrides
    def get_metric(self, reset=False):
        # Fraction of event arguments associated with multiple triggers.
        args_multiple = (self._repeated_arguments / self._total_arguments
                         if self._total_arguments
                         else 0)

        if reset:
            self.reset()

        res = dict(args_multiple=args_multiple)
        return res

    @overrides
    def reset(self):
        self._total_arguments = 0
        self._repeated_arguments = 0
