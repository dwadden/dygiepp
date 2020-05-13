import logging
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

import torch
from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules import TimeDistributed
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from dygie.training.ner_metrics import NERMetrics
from dygie.models import shared

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class NERTagger(Model):
    """
    Named entity recognition module of DyGIE model.

    Parameters
    ----------
    mention_feedforward : ``FeedForward``
        This feedforward network is applied to the span representations which is then scored
        by a linear layer.
    feature_size: ``int``
        The embedding size for all the embedded features, such as distances or span widths.
    lexical_dropout: ``int``
        The probability of dropping out dimensions of the embedded text.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 mention_feedforward: FeedForward,
                 feature_size: int,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(NERTagger, self).__init__(vocab, regularizer)

        # Number of classes determine the output dimension of the final layer
        self._n_labels = vocab.get_vocab_size('ner_labels')

        # TODO(dwadden) think of a better way to enforce this.
        # Null label is needed to keep track of when calculating the metrics
        null_label = vocab.get_token_index("", "ner_labels")
        assert null_label == 0  # If not, the dummy class won't correspond to the null label.

        # The output dim is 1 less than the number of labels because we don't score the null label;
        # we just give it a score of 0 by default.
        self._ner_scorer = torch.nn.Sequential(
            TimeDistributed(mention_feedforward),
            TimeDistributed(torch.nn.Linear(
                mention_feedforward.get_output_dim(),
                self._n_labels - 1)))

        self._ner_metrics = NERMetrics(self._n_labels, null_label)

        self._loss = torch.nn.CrossEntropyLoss(reduction="sum")

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                spans: torch.IntTensor,
                span_mask: torch.IntTensor,
                span_embeddings: torch.IntTensor,
                sentence_lengths: torch.Tensor,
                ner_labels: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        """
        TODO(dwadden) Write documentation.
        """

        # Shape: (Batch size, Number of Spans, Span Embedding Size)
        # span_embeddings
        ner_scores = self._ner_scorer(span_embeddings)
        # Give large negative scores to masked-out elements.
        mask = span_mask.unsqueeze(-1)
        ner_scores = util.replace_masked_values(ner_scores, mask, -1e20)
        # The dummy_scores are the score for the null label.
        dummy_dims = [ner_scores.size(0), ner_scores.size(1), 1]
        dummy_scores = ner_scores.new_zeros(*dummy_dims)
        ner_scores = torch.cat((dummy_scores, ner_scores), -1)

        # Mask out irrelevant labels.
        ner_scores = self._mask_irrelevant_labels(ner_scores, metadata)

        _, predicted_ner = ner_scores.max(2)

        output_dict = {"spans": spans,
                       "span_mask": span_mask,
                       "ner_scores": ner_scores,
                       "metadata": metadata,
                       "predicted_ner": predicted_ner}

        if ner_labels is not None:
            self._ner_metrics(predicted_ner, ner_labels, span_mask)

            ner_scores_flat = ner_scores.view(-1, self._n_labels)
            ner_labels_flat = ner_labels.view(-1)
            mask_flat = span_mask.view(-1).bool()

            loss = self._loss(ner_scores_flat[mask_flat], ner_labels_flat[mask_flat])
            output_dict["loss"] = loss

        if metadata is not None:
            output_dict["document"] = [x["sentence"] for x in metadata]

        return output_dict

    def _mask_irrelevant_labels(self, ner_scores, metadata):
        """
        Mask out the NER scores for classes from different datasets.
        """
        if not shared.is_seed_datasets(metadata):
            return ner_scores

        datasets = pd.Series([x["doc_key"].split(":")[0] for x in metadata]).values
        datasets = np.expand_dims(datasets, 1)
        # The null label is the first class.
        indices = pd.Series(self.vocab.get_index_to_token_vocabulary("ner_labels"))
        indices = indices.str.split(":").str[0].values
        indices = np.expand_dims(indices, 0)
        # When mask == 1, we keep the score. When mask == 0, we ignore it.
        mask = datasets == indices
        mask[:, 0] = True  # Always keep the null label.
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)
        mask = mask.unsqueeze(1).to(ner_scores.device)
        masked_scores = util.replace_masked_values(ner_scores, mask, -1e20)
        return masked_scores


    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]):
        predicted_ner_batch = output_dict["predicted_ner"].detach().cpu()
        ner_scores_batch = output_dict["ner_scores"].detach().cpu()
        spans_batch = output_dict["spans"].detach().cpu()
        span_mask_batch = output_dict["span_mask"].detach().cpu().bool()

        if shared.is_seed_datasets(output_dict["metadata"]):
            decoded_ner, decoded_ner_dict = self._decode_regular(
                predicted_ner_batch, spans_batch, span_mask_batch)
        else:
            # If we're doing multi-label decoding, make sure we get the same
            # spans as before (just a check since I'm doing this quickly).
            decoded_ner, decoded_ner_dict = self._decode_cord(
                ner_scores_batch, spans_batch, span_mask_batch)
            check_ner, check_ner_dict = self._decode_regular(
                predicted_ner_batch, spans_batch, span_mask_batch)

            for decoded_dict, check_dict in zip(decoded_ner_dict, check_ner_dict):
                if decoded_dict.keys() != check_dict.keys():
                    raise Exception("Check on multi-label decoding failed.")

        output_dict["decoded_ner"] = decoded_ner
        output_dict["decoded_ner_dict"] = decoded_ner_dict

        return output_dict

    def _decode_regular(self, predicted_ner_batch, spans_batch, span_mask_batch):
        res_list = []
        res_dict = []
        for spans, span_mask, predicted_NERs in zip(spans_batch, span_mask_batch, predicted_ner_batch):
            entry_list = []
            entry_dict = {}
            for span, ner in zip(spans[span_mask], predicted_NERs[span_mask]):
                ner = ner.item()
                if ner > 0:
                    the_span = (span[0].item(), span[1].item())
                    the_label = self.vocab.get_token_from_index(ner, "ner_labels")
                    entry_list.append((the_span[0], the_span[1], the_label))
                    entry_dict[the_span] = the_label
            res_list.append(entry_list)
            res_dict.append(entry_dict)

        return res_list, res_dict

    def _decode_cord(self, ner_scores_batch, spans_batch, span_mask_batch):
        indices = pd.Series(self.vocab.get_index_to_token_vocabulary("ner_labels"))
        datasets = indices.str.split(":").str[0].values
        uniq_datasets = np.unique(datasets)

        res_list = []
        res_dict = []
        for spans, span_mask, ner_scores in zip(spans_batch, span_mask_batch, ner_scores_batch):
            threshold = ner_scores[0, 0].item()
            ner_np = ner_scores.numpy()[span_mask.numpy()]

            scores_dataset = {}
            for dataset in uniq_datasets:
                if dataset == "":
                    continue
                scores_dataset[dataset] = {}
                col_ix = datasets == dataset
                labels = indices[col_ix].values
                ner_dataset = ner_np[:, col_ix]
                scores_dataset[dataset]["max_scores"] = ner_dataset.max(axis=1)
                max_indices = ner_dataset.argmax(axis=1)
                scores_dataset[dataset]["max_labels"] = labels[max_indices]

            entry_list = []
            entry_dict = {}
            for i, span in enumerate(spans[span_mask]):
                the_span = (span[0].item(), span[1].item())
                span_labels = []

                for dataset in uniq_datasets:
                    if dataset == "":
                        continue
                    this_score = scores_dataset[dataset]["max_scores"][i]
                    this_label = scores_dataset[dataset]["max_labels"][i]

                    if this_score > threshold:
                        span_labels.append({"label": this_label, "score": this_score})

                if span_labels:
                    entry_list.append((the_span[0], the_span[1], span_labels))
                    entry_dict[the_span] = span_labels

            res_list.append(entry_list)
            res_dict.append(entry_dict)

        return res_list, res_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        ner_precision, ner_recall, ner_f1 = self._ner_metrics.get_metric(reset)
        return {"ner_precision": ner_precision,
                "ner_recall": ner_recall,
                "ner_f1": ner_f1}
