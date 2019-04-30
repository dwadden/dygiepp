from overrides import overrides
from typing import Optional

import torch

from allennlp.training.metrics.metric import Metric

class NERMetrics(Metric):
    """
    Computes precision, recall, and micro-averaged F1 from a list of predicted and gold labels.
    """
    def __init__(self, number_of_classes: int, none_label: int=0):
        self.number_of_classes = number_of_classes
        self.none_label = none_label
        self.reset()

    @overrides
    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        predictions = predictions.cpu()
        gold_labels = gold_labels.cpu()
        mask = mask.cpu()
        for i in range(self.number_of_classes):
            if i == self.none_label:
                continue
            self._true_positives += ((predictions==i)*(gold_labels==i)*mask.byte()).sum()
            self._false_positives += ((predictions==i)*(gold_labels!=i)*mask.byte()).sum()
            self._true_negatives += ((predictions!=i)*(gold_labels!=i)*mask.byte()).sum()
            self._false_negatives += ((predictions!=i)*(gold_labels==i)*mask.byte()).sum()

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
        precision = float(self._true_positives) / (float(self._true_positives + self._false_positives) + 1e-13)
        recall = float(self._true_positives) / (float(self._true_positives + self._false_negatives) + 1e-13)
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
