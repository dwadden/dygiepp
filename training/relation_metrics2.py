from overrides import overrides

from allennlp.training.metrics.metric import Metric


class RelationMetrics(Metric):
    """
    Computes precision, recall, and micro-averaged F1 from a list of predicted and gold spans.
    """
    def __init__(self):
        self.reset()

    @overrides
    def __call__(self, predicted_relation_list, metadata_list):
        for predicted, metadata in zip(predicted_relation_list, metadata_list):
            gold = metadata["relation_dict"]
            self._n_relevant += len(gold)
            self._n_retrieved += len(predicted)
            for k, v in predicted:
                if k in gold and gold[k] == v:
                    self._n_match += 1

    @overrides
    def get_metric(self, reset=False):
        precision = self._n_match / self._n_retrieved if self._n_retrieved > 0 else 0
        recall = self._n_match / self._n_relevant if self._n_relevant > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        if reset:
            print("Resetting")
            self.reset()

        return precision, recall, f1

    @overrides
    def reset(self):
        self._n_relevant = 0
        self._n_retrieved = 0
        self._n_match = 0


class CandidateRecall(Metric):
    """
    Computes relation candidate recall.
    """
    def __init__(self):
        self.reset()

    def __call__(self, predicted_relation_list, metadata_list):
        pass

    @overrides
    def get_metric(self, reset=False):
        return 0

    @overrides
    def reset(self):
        pass
