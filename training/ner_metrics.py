from overrides import overrides
from typing import Optional

import torch

from allennlp.training.metrics.metric import Metric
from allennlp.training.metrics.f1_measure import F1Measure

class NERMetrics(Metric):
    """
    Computes precision, recall, and micro-averaged F1 from a list of predicted and gold labels.
    """
    def __init__(self, number_of_classes: int, none_label: int=0):
        self.number_of_classes = number_of_classes
        self.none_label = none_label
        self._single_class_ner_metrics = [F1Measure(i) for i in range(self.number_of_classes) if i != self.none_label]
        self.reset()

    @overrides
    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        for metric in self._single_class_ner_metrics:
            metric(predictions, gold_labels, mask)

    @overrides
    def get_metric(self, reset=False):
        """
        Returns
        -------
        A tuple of the following metrics based on the accumulated count statistics:
        precision : float
        recall : float
        f1-measure : float
        """
        self._true_positives = sum(metric._true_positives for metric in self._single_class_ner_metrics)
        self._false_positives = sum(metric._false_positives for metric in self._single_class_ner_metrics)
        self._true_negatives = sum(metric._true_negatives for metric in self._single_class_ner_metrics)
        self._false_negatives = sum(metric._false_negatives for metric in self._single_class_ner_metrics)

        precision = float(self._true_positives) / float(self._true_positives + self._false_positives + 1e-13)
        recall = float(self._true_positives) / float(self._true_positives + self._false_negatives + 1e-13)
        f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))

        # Reset counts if at end of epoch.
        if reset:
            self.reset()

        return precision, recall, f1_measure

    @overrides
    def reset(self):
        self._true_positives = 0
        self._false_positives = 0
        self._true_negatives = 0
        self._false_negatives = 0
        for metric in self._single_class_ner_metrics:
            metric.reset()
