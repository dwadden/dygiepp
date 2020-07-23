from overrides import overrides

from allennlp.training.metrics.metric import Metric

from dygie.training.f1 import compute_f1


class RelationMetrics(Metric):
    """
    Computes precision, recall, and micro-averaged F1 from a list of predicted and gold spans.
    """
    def __init__(self):
        self.reset()

    # TODO(dwadden) This requires decoding because the dataset reader gets rid of gold spans wider
    # than the span width. So, I can't just compare the tensor of gold labels to the tensor of
    # predicted labels.
    @overrides
    def __call__(self, predicted_relation_list, metadata_list):
        for predicted_relations, metadata in zip(predicted_relation_list, metadata_list):
            gold_relations = metadata.relation_dict
            self._total_gold += len(gold_relations)
            self._total_predicted += len(predicted_relations)
            for (span_1, span_2), label in predicted_relations.items():
                ix = (span_1, span_2)
                if ix in gold_relations and gold_relations[ix] == label:
                    self._total_matched += 1

    @overrides
    def get_metric(self, reset=False):
        precision, recall, f1 = compute_f1(self._total_predicted, self._total_gold, self._total_matched)

        # Reset counts if at end of epoch.
        if reset:
            self.reset()

        return precision, recall, f1

    @overrides
    def reset(self):
        self._total_gold = 0
        self._total_predicted = 0
        self._total_matched = 0
