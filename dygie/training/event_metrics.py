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


# TODO(dwadden) Clean this up.
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
            gold_triggers = metadata.events.trigger_dict
            self._score_triggers(predicted_triggers, gold_triggers)

            # Argument scoring.
            predicted_arguments = predicted_events["argument_dict"]
            gold_arguments = metadata.events.argument_dict
            self._score_arguments(
                predicted_triggers, gold_triggers, predicted_arguments, gold_arguments)

    def _score_triggers(self, predicted_triggers, gold_triggers):
        self._gold_triggers += len(gold_triggers)
        self._predicted_triggers += len(predicted_triggers)
        for token_ix, pred in predicted_triggers.items():
            label = pred[0]
            # Check whether the offsets match, and whether the labels match.
            if token_ix in gold_triggers:
                self._matched_trigger_ids += 1
                if gold_triggers[token_ix] == label:
                    self._matched_trigger_classes += 1

    def _score_arguments(self, predicted_triggers, gold_triggers, predicted_arguments, gold_arguments):
        # Note that the index of the trigger doesn't actually need to be correct to get full credit;
        # the event type and event role need to be correct (see Sec. 3 of paper).
        def format(arg_dict, trigger_dict, prediction=False):
            # Make it a list of [index, event_type, arg_label].
            res = []
            for (trigger_ix, arg_ix), label in arg_dict.items():
                # If it doesn't match a trigger, don't predict it (enforced in decoding).
                if trigger_ix not in trigger_dict:
                    continue
                event_type = trigger_dict[trigger_ix]
                # TODO(dwadden) This is clunky; it's because predictions have confidence scores.
                if prediction:
                    event_type = event_type[0]
                    label = label[0]
                res.append((arg_ix, event_type, label))
            return res

        formatted_gold_arguments = format(gold_arguments, gold_triggers, prediction=False)
        formatted_predicted_arguments = format(predicted_arguments, predicted_triggers, prediction=True)

        self._gold_arguments += len(formatted_gold_arguments)
        self._predicted_arguments += len(formatted_predicted_arguments)

        # Go through each predicted arg and look for a match.
        for entry in formatted_predicted_arguments:
            # No credit if not associated with a predicted trigger.
            class_match = int(any([entry == gold for gold in formatted_gold_arguments]))
            id_match = int(any([entry[:2] == gold[:2] for gold in formatted_gold_arguments]))

            self._matched_argument_classes += class_match
            self._matched_argument_ids += id_match


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
