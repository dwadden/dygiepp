import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.modules import FeedForward
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder, Pruner
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor, EndpointSpanExtractor
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import ConllCorefScores, F1Measure

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
    spans_per_word: float, required.
        A multiplier between zero and one which controls what percentage of candidate mention
        spans we retain with respect to the number of words in the document.
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
                 spans_per_word: float,
                 # initializer: InitializerApplicator = InitializerApplicator(), # TODO(dwadden add this).
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(NERTagger, self).__init__(vocab, regularizer)

        #This should be passed as one of the parameters
        self.number_of_ner_classes = vocab.get_vocab_size('ner_labels')

        # TODO(dwadden) Do we want TimeDistributed for this one? Ulme - Yes, I think we do
        #feedforward_scorer = torch.nn.Sequential(
        #    TimeDistributed(mention_feedforward),
        #    TimeDistributed(torch.nn.Linear(mention_feedforward.get_output_dim(), 1))
        #)
        #self._mention_pruner = Pruner(feedforward_scorer)

        self.final_network = torch.nn.Sequential(
            TimeDistributed(mention_feedforward),
            TimeDistributed(torch.nn.Linear(
                                mention_feedforward.get_output_dim(),
                                self.number_of_ner_classes - 1)
            )
        )

        self.loss_function = torch.nn.CrossEntropyLoss()

        # TODO(dwadden) Add this.
        #initializer(self)

        #self._conll_coref_scores = conllcorefscores()
        self._ner_metrics = [F1Measure(i) for i in range(self.number_of_ner_classes)]

    @overrides
    def forward(self,  # type: ignore
                spans: torch.IntTensor,
                span_mask: torch.IntTensor,
                span_embeddings: torch.IntTensor,
                sentence_lengths: torch.Tensor,
                max_sentence_length: int,
                ner_labels: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        """
        TODO(dwadden) Write documentation.
        """

        #Shape: (Batch size, Number of Spans, Span Embedding Size)
        #span_embeddings

        num_spans = spans.size(1)

        #num_spans_to_keep = int(math.floor(self._spans_per_word * max_sentence_length))

        # Prune based on mention scores.

        #PRUNE
        #(top_span_embeddings, top_span_mask,
        # top_span_indices, top_span_mention_scores) = self._mention_pruner(span_embeddings,
        #                                                                   span_mask,
        #                                                                   num_spans_to_keep)
        #PRUNE
        #top_span_mask = top_span_mask.unsqueeze(-1)

        # Shape: (batch_size * num_spans_to_keep)
        # torch.index_select only accepts 1D indices, but here
        # we need to select spans for each element in the batch.
        # This reformats the indices to take into account their
        # index into the batch. We precompute this here to make
        # the multiple calls to util.batched_index_select below more efficient.

        #PRUNE
        #flat_top_span_indices = util.flatten_and_batch_shift_indices(top_span_indices, num_spans)

        # Compute final predictions for which spans to consider as mentions.
        # Shape: (batch_size, num_spans_to_keep, 2)

        #PRUNE
        #top_spans = util.batched_index_select(spans,
        #                                      top_span_indices,
        #                                      flat_top_span_indices)

        # Compute indices for antecedent spans to consider.
        #max_antecedents = min(self._max_antecedents, num_spans_to_keep)

        # Now that we have our variables in terms of num_spans_to_keep, we need to
        # compare span pairs to decide each span's antecedent. Each span can only
        # have prior spans as antecedents, and we only consider up to max_antecedents
        # prior spans. So the first thing we do is construct a matrix mapping a span's
        #  index to the indices of its allowed antecedents. Note that this is independent
        #  of the batch dimension - it's just a function of the span's position in
        # top_spans. The spans are in document order, so we can just use the relative
        # index of the spans to know which other spans are allowed antecedents.

        # Once we have this matrix, we reformat our variables again to get embeddings
        # for all valid antecedents for each span. This gives us variables with shapes
        #  like (batch_size, num_spans_to_keep, max_antecedents, embedding_size), which
        #  we can use to make coreference decisions between valid span pairs.

        # Shapes:
        # (num_spans_to_keep, max_antecedents),
        # (1, max_antecedents),
        # (1, num_spans_to_keep, max_antecedents)

        #valid_antecedent_indices, valid_antecedent_offsets, valid_antecedent_log_mask = \
        #    self._generate_valid_antecedents(num_spans_to_keep, max_antecedents, util.get_device_of(text_mask))
        # Select tensors relating to the antecedent spans.
        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        #candidate_antecedent_embeddings = util.flattened_index_select(top_span_embeddings,
        #                                                              valid_antecedent_indices)

        # Shape: (batch_size, num_spans_to_keep, max_antecedents)
        #candidate_antecedent_mention_scores = util.flattened_index_select(top_span_mention_scores,
        #                                                                  valid_antecedent_indices).squeeze(-1)
        # Compute antecedent scores.
        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        #span_pair_embeddings = self._compute_span_pair_embeddings(top_span_embeddings,
        #                                                          candidate_antecedent_embeddings,
        #                                                          valid_antecedent_offsets)
        # Shape: (batch_size, num_spans_to_keep, 1 + max_antecedents)
        #coreference_scores = self._compute_coreference_scores(span_pair_embeddings,
        #                                                      top_span_mention_scores,
        #                                                      candidate_antecedent_mention_scores,
        #                                                      valid_antecedent_log_mask)

        ner_scores = self.final_network(span_embeddings)
        dummy_dims = [ner_scores.size(0), ner_scores.size(1), 1]
        dummy_scores = ner_scores.new_zeros(*dummy_dims)
        ner_scores = torch.cat((dummy_scores, ner_scores), -1)

        #ner_scores = self._compute_ner_scores(span_embeddings,
        #                                      top_span_mention_scores)

        # We now have, for each span which survived the pruning stage,
        # a predicted antecedent. This implies a clustering if we group
        # mentions which refer to each other in a chain.
        # Shape: (batch_size, num_spans_to_keep)
        _, predicted_ner = ner_scores.max(2)
        # Subtract one here because index 0 is the "no antecedent" class,
        # so this makes the indices line up with actual spans if the prediction
        # is greater than -1.
        #predicted_ner -= 1

        top_spans = spans
        output_dict = {"top_spans": top_spans,
                       "predicted_ner": predicted_ner}

        if ner_labels is not None:
            print('----------------ner labels----------')
            print(ner_labels.min())
            print(ner_labels.max())
            print('----------------ner labels----------')
            #import ipdb; ipdb.set_trace()
            #ner_labels = ner_labels.resize(2415)


            # Find the gold labels for the spans which we kept.
            #pruned_gold_labels = util.batched_index_select(ner_labels.unsqueeze(-1),
            #                                               top_span_indices,
            #                                               flat_top_span_indices)

            # Compute labels.
            # Shape: (batch_size, num_spans_to_keep, max_antecedents + 1)
            #gold_antecedent_labels = self._compute_antecedent_gold_labels(pruned_gold_labels,
            #                                                              antecedent_labels)
            # Now, compute the loss using the negative marginal log-likelihood.
            # This is equal to the log of the sum of the probabilities of all antecedent predictions
            # that would be consistent with the data, in the sense that we are minimising, for a
            # given span, the negative marginal log likelihood of all antecedents which are in the
            # same gold cluster as the span we are currently considering. Each span i predicts a
            # single antecedent j, but there might be several prior mentions k in the same
            # coreference cluster that would be valid antecedents. Our loss is the sum of the
            # probability assigned to all valid antecedents. This is a valid objective for
            # clustering as we don't mind which antecedent is predicted, so long as they are in
            #  the same coreference cluster.

            #ner_log_probs = util.masked_log_softmax(ner_scores, top_span_mask)
            #correct_antecedent_log_probs = ner_log_probs# + gold_antecedent_labels.log()
            #negative_marginal_log_likelihood = -util.logsumexp(correct_antecedent_log_probs).sum()

            #self._conll_coref_scores(top_spans, valid_antecedent_indices, predicted_antecedents, metadata)

            for metric in self._ner_metrics:
                metric(ner_scores, ner_labels, span_mask)
            loss = util.sequence_cross_entropy_with_logits(ner_scores, ner_labels, span_mask)
            output_dict["loss"] = loss

        if metadata is not None:
            output_dict["document"] = [x["sentence"] for x in metadata]
        return output_dict


    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]):
        pass
        """
        Converts the list of spans and predicted antecedent indices into clusters
        of spans for each element in the batch.

        Parameters
        ----------
        output_dict : ``Dict[str, torch.Tensor]``, required.
            The result of calling :func:`forward` on an instance or batch of instances.

        Returns
        -------
        The same output dictionary, but with an additional ``clusters`` key:

        clusters : ``List[List[List[Tuple[int, int]]]]``
            A nested list, representing, for each instance in the batch, the list of clusters,
            which are in turn comprised of a list of (start, end) inclusive spans into the
            original document.
        """

        # A tensor of shape (batch_size, num_spans_to_keep, 2), representing
        # the start and end indices of each span.
        batch_top_spans = output_dict["top_spans"].detach().cpu()

        # A tensor of shape (batch_size, num_spans_to_keep) representing, for each span,
        # the index into ``antecedent_indices`` which specifies the antecedent span. Additionally,
        # the index can be -1, specifying that the span has no predicted antecedent.
        batch_predicted_antecedents = output_dict["predicted_antecedents"].detach().cpu()

        # A tensor of shape (num_spans_to_keep, max_antecedents), representing the indices
        # of the predicted antecedents with respect to the 2nd dimension of ``batch_top_spans``
        # for each antecedent we considered.
        antecedent_indices = output_dict["antecedent_indices"].detach().cpu()
        batch_clusters: List[List[List[Tuple[int, int]]]] = []

        # Calling zip() on two tensors results in an iterator over their
        # first dimension. This is iterating over instances in the batch.
        for top_spans, predicted_antecedents in zip(batch_top_spans, batch_predicted_antecedents):
            spans_to_cluster_ids: Dict[Tuple[int, int], int] = {}
            clusters: List[List[Tuple[int, int]]] = []

            for i, (span, predicted_antecedent) in enumerate(zip(top_spans, predicted_antecedents)):
                if predicted_antecedent < 0:
                    # We don't care about spans which are
                    # not co-referent with anything.
                    continue

                # Find the right cluster to update with this span.
                # To do this, we find the row in ``antecedent_indices``
                # corresponding to this span we are considering.
                # The predicted antecedent is then an index into this list
                # of indices, denoting the span from ``top_spans`` which is the
                # most likely antecedent.
                predicted_index = antecedent_indices[i, predicted_antecedent]

                antecedent_span = (top_spans[predicted_index, 0].item(),
                                   top_spans[predicted_index, 1].item())

                # Check if we've seen the span before.
                if antecedent_span in spans_to_cluster_ids:
                    predicted_cluster_id: int = spans_to_cluster_ids[antecedent_span]
                else:
                    # We start a new cluster.
                    predicted_cluster_id = len(clusters)
                    # Append a new cluster containing only this span.
                    clusters.append([antecedent_span])
                    # Record the new id of this span.
                    spans_to_cluster_ids[antecedent_span] = predicted_cluster_id

                # Now add the span we are currently considering.
                span_start, span_end = span[0].item(), span[1].item()
                clusters[predicted_cluster_id].append((span_start, span_end))
                spans_to_cluster_ids[(span_start, span_end)] = predicted_cluster_id
            batch_clusters.append(clusters)

        output_dict["clusters"] = batch_clusters
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = [metric.get_metric(reset) for metric in self._ner_metrics]
        print('---------metrics---------')
        print(metrics)
        print('---------metrics---------')
        return {"ner_precision": sum(el[0] for el in metrics)/len(metrics),
                "ner_recall": sum(el[1] for el in metrics)/len(metrics),
                "ner_f1": sum(el[2] for el in metrics)/len(metrics)}
