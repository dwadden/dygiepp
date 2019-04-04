"""
Metrics that require reasoning over outputs from multiple modules - for example, both NER and
events.
"""

from overrides import overrides

from allennlp.training.metrics.metric import Metric

# TODO(dwadden) Write a unit test at some point.


class JointMetrics(Metric):
    """
    Right now, counts how many event predictions are valid. I.e. the trigger / argument and argument
    / ner labels are compatible.
    """
    def __init__(self, valid_events):
        self._valid_events = valid_events
        self.reset()

    @overrides
    def __call__(self, predicted_ner_list, predicted_events_list):
        """
        Loop over all event argument predictions. Check whether the event trigger and the argument
        ner tag agree with the role of the argument.
        """
        for predicted_ners, predicted_events in zip(predicted_ner_list, predicted_events_list):
            trigger_dict = predicted_events["trigger_dict"]
            for (trigger_ix, arg_span), predicted_arg in predicted_events["argument_dict"].items():
                self._total_trigger_arg_pairs += 1
                self._total_ner_arg_pairs += 1
                predicted_trigger = trigger_dict[trigger_ix] if trigger_ix in trigger_dict else ""
                # Check if predicted trigger, arg pair is allowed.
                if (predicted_trigger, predicted_arg) in self._valid_events["trigger_to_arg"]:
                    self._valid_trigger_arg_pairs += 1
                # Check if the predicted ner, arg pair is allowed.
                predicted_ner = predicted_ners[arg_span] if arg_span in predicted_ners else ""
                if (predicted_ner, predicted_arg) in self._valid_events["ner_to_arg"]:
                    self._valid_ner_arg_pairs += 1

    @overrides
    def get_metric(self, reset=False):
        frac_valid_trigger_arg_pairs = (self._valid_trigger_arg_pairs / self._total_trigger_arg_pairs
                                        if self._total_trigger_arg_pairs > 0
                                        else 0)
        frac_valid_ner_arg_pairs = (self._valid_ner_arg_pairs / self._total_ner_arg_pairs
                                    if self._total_ner_arg_pairs > 0
                                    else 0)

        # Reset counts if requested.
        if reset:
            self.reset()

        res = dict(frac_trig_arg=frac_valid_trigger_arg_pairs,
                   frac_ner_arg=frac_valid_ner_arg_pairs)
        return res

    def reset(self):
        self._valid_trigger_arg_pairs = 0
        self._total_trigger_arg_pairs = 0
        self._valid_ner_arg_pairs = 0
        self._total_ner_arg_pairs = 0
