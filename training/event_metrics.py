from overrides import overrides

from allennlp.training.metrics.metric import Metric


class EventMetrics(Metric):
    """
    Computes precision, recall, and micro-averaged F1 for triggers and arguments.
    """
    def __init__(self):
        self.reset()

    @overrides
    def __call__(self, predicted_events_list, metadata_list):
        for predicted_triggers, metadata in zip(predicted_events_list, metadata_list):
            # Trigger scoring.
            gold_triggers = metadata["trigger_dict"]
            self._total_gold_triggers += len(gold_triggers)
            self._total_predicted_triggers += len(predicted_triggers)
            for token_ix, label in predicted_triggers.items():
                if token_ix in gold_triggers and gold_triggers[token_ix] == label:
                    self._total_matched_triggers += 1

    @overrides
    def get_metric(self, reset=False):
        # Triggers
        trigger_precision = (self._total_matched_triggers / self._total_predicted_triggers
                             if self._total_predicted_triggers > 0
                             else 0)
        trigger_recall = (self._total_matched_triggers / self._total_gold_triggers
                          if self._total_gold_triggers > 0
                          else 0)
        trigger_f1 = (2 * trigger_precision * trigger_recall / (trigger_precision + trigger_recall)
                      if trigger_precision + trigger_recall > 0
                      else 0)

        # Reset counts if at end of epoch.
        if reset:
            self.reset()

        res = dict(trig_precision=trigger_precision,
                   trig_recall=trigger_recall,
                   trig_f1=trigger_f1)
        return res

    @overrides
    def reset(self):
        self._total_gold_triggers = 0
        self._total_predicted_triggers = 0
        self._total_matched_triggers = 0
