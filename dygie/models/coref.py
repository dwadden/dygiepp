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
from allennlp.modules import TimeDistributed
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import MentionRecall, ConllCorefScores

from dygie.models import shared
from dygie.models.entity_beam_pruner import Pruner

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class CorefResolver(Model):
    """
    TODO(dwadden) document correctly.

    Parameters
    ----------
    mention_feedforward : ``FeedForward``
        This feedforward network is applied to the span representations which is then scored
        by a linear layer.
    antecedent_feedforward: ``FeedForward``
        This feedforward network is applied to pairs of span representation, along with any
        pairwise features, which is then scored by a linear layer.
    feature_size: ``int``
        The embedding size for all the embedded features, such as distances or span widths.
    spans_per_word: float, required.
        A multiplier between zero and one which controls what percentage of candidate mention
        spans we retain with respect to the number of words in the document.
    max_antecedents: int, required.
        For each mention which survives the pruning stage, we consider this many antecedents.
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
                 antecedent_feedforward: FeedForward,
                 feature_size: int,
                 spans_per_word: float,
                 span_emb_dim: int,
                 max_antecedents: int,
                 coref_prop: int = 0,
                 coref_prop_dropout_f: float = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator(), # TODO(dwadden add this).
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(CorefResolver, self).__init__(vocab, regularizer)

        self._antecedent_feedforward = TimeDistributed(antecedent_feedforward)
        feedforward_scorer = torch.nn.Sequential(
            TimeDistributed(mention_feedforward),
            TimeDistributed(torch.nn.Linear(mention_feedforward.get_output_dim(), 1)))
        self._mention_pruner = Pruner(feedforward_scorer)
        self._antecedent_scorer = TimeDistributed(torch.nn.Linear(antecedent_feedforward.get_output_dim(), 1))

        # 10 possible distance buckets.
        self._num_distance_buckets = 10
        self._distance_embedding = Embedding(self._num_distance_buckets, feature_size)

        self._spans_per_word = spans_per_word
        self._max_antecedents = max_antecedents

        self._mention_recall = MentionRecall()
        self._conll_coref_scores = ConllCorefScores()

        self.coref_prop = coref_prop
        self._f_network = FeedForward(input_dim=2*span_emb_dim,
                                      num_layers=1,
                                      hidden_dims=span_emb_dim,
                                      activations=torch.nn.Sigmoid(),
                                      dropout=coref_prop_dropout_f)

        #self._f_network2 = FeedForward(input_dim=2*span_emb_dim,
        #                              num_layers=1,
        #                              hidden_dims=1,
        #                              activations=torch.nn.Sigmoid(),
        #                              dropout=coref_prop_dropout_f)
        self.antecedent_softmax = torch.nn.Softmax(dim=-1)
        initializer(self)

    def update_spans(self, output_dict, span_embeddings_batched, indices):
        new_span_embeddings_batched = span_embeddings_batched.clone()
        offsets = {}
        for key in indices:
            offset = 0
            while indices[key][offset] == 0:
                offset += 1
            offsets[key] = offset
        for doc_key in output_dict:
            span_ix = output_dict[doc_key]["span_ix"]
            top_span_embeddings = output_dict[doc_key]["top_span_embeddings"]
            for ix, el in enumerate(output_dict[doc_key]["top_span_indices"].view(-1)):
                new_span_embeddings_batched[span_ix[el] / span_embeddings_batched.shape[1] + offsets[doc_key], span_ix[el] % span_embeddings_batched.shape[1]] = top_span_embeddings[0, ix]

        return new_span_embeddings_batched

    def coref_propagation(self, output_dict):
        for doc_key in output_dict:
            output_dict[doc_key] = self.coref_propagation_doc(output_dict[doc_key])
        return output_dict

    def coref_propagation_doc(self, output_dict):
        coreference_scores = output_dict["coreference_scores"]
        top_span_embeddings = output_dict["top_span_embeddings"]
        antecedent_indices = output_dict["antecedent_indices"]
        for t in range(self.coref_prop):
            assert coreference_scores.shape[1] == antecedent_indices.shape[0]
            assert coreference_scores.shape[2] - 1 == antecedent_indices.shape[1]
            assert top_span_embeddings.shape[1] == coreference_scores.shape[1]
            assert antecedent_indices.max() <= top_span_embeddings.shape[1]

            antecedent_distribution = self.antecedent_softmax(coreference_scores)[:, :, 1:]
            top_span_emb_repeated = top_span_embeddings.repeat(antecedent_distribution.shape[2],1,1)
            if antecedent_indices.shape[0]==antecedent_indices.shape[1]:
                selected_top_span_embs = util.batched_index_select(top_span_emb_repeated, antecedent_indices).unsqueeze(0)
                entity_embs = (selected_top_span_embs.permute([3,0,1,2]) * antecedent_distribution).permute([1, 2, 3, 0]).sum(dim=2)
            else:
                ant_var1 = antecedent_indices.unsqueeze(0).unsqueeze(-1).repeat(1,1,1,top_span_embeddings.shape[-1])
                top_var1 = top_span_embeddings.unsqueeze(1).repeat(1,antecedent_distribution.shape[1],1,1)
                entity_embs = (torch.gather(top_var1, 2, ant_var1).permute([3,0,1,2]) * antecedent_distribution).permute([1, 2, 3, 0]).sum(dim=2)

            #entity_embs = F.dropout(entity_embs)

            f_network_input = torch.cat([top_span_embeddings, entity_embs], dim=-1)
            f_weights = self._f_network(f_network_input)
            top_span_embeddings = f_weights * top_span_embeddings + (1.0 - f_weights) * entity_embs

            #f_weights2 = self._f_network2(f_network_input)
            #top_span_embeddings = f_weights2 * top_span_embeddings + (1.0 - f_weights2) * entity_embs
            coreference_scores = self.get_coref_scores(top_span_embeddings, self._mention_pruner._scorer(top_span_embeddings), output_dict["antecedent_indices"], output_dict["valid_antecedent_offsets"], output_dict["valid_antecedent_log_mask"])

        output_dict["coreference_scores"] = coreference_scores
        output_dict["top_span_embeddings"] = top_span_embeddings
        return output_dict

    #@overrides
    #def forward(self,  # type: ignore
    def compute_representations(self,  # type: ignore
                spans_batched: torch.IntTensor,
                span_mask_batched,
                span_embeddings_batched,  # TODO(dwadden) add type.
                sentence_lengths,
                coref_labels_batched: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """
        Run the forward pass. Since we can only have coreferences between spans in the same
        document, we loop over the documents in the batch. This function assumes that the inputs are
        in order, but may go across documents.
        """
        output_docs = {}
        doc_keys = [entry["doc_key"] for entry in metadata]
        uniq_keys = []
        for entry in doc_keys:
            if entry not in uniq_keys:
                uniq_keys.append(entry)

        indices = {}
        for key in uniq_keys:
            ix_list = [1 if entry == key else 0 for entry in doc_keys]
            indices[key] = ix_list
            doc_metadata = [entry for entry in metadata if entry["doc_key"] == key]
            ix = torch.tensor(ix_list, dtype=torch.bool)
            if sentence_lengths[ix].sum().item() > 1:
                output_docs[key] = self._compute_representations_doc(
                    spans_batched[ix], span_mask_batched[ix], span_embeddings_batched[ix],
                    sentence_lengths[ix], ix, coref_labels_batched[ix], doc_metadata)
        return output_docs, indices

    def predict_labels(self, output_docs, metadata):
        for key in output_docs:
            output_docs[key] = self.predict_labels_doc(output_docs[key])
        return self.collect_losses(output_docs)

    def collect_losses(self, output_docs):
        uniq_keys = [el for el in output_docs]
        losses = torch.cat([entry["loss"].unsqueeze(0) for entry in output_docs.values()])
        loss = torch.sum(losses)

        # At train time, return a separate output dict for each document.
        if self.training:
            output = {"loss": loss,
                      "doc": output_docs}
        # At test time, we evaluate a whole document at a time. Just return the results for that
        # document.
        else:
            assert len(uniq_keys) == 1
            key = uniq_keys[0]
            output = output_docs[key]
            output["loss"] = loss
        return output

    def _compute_representations_doc(self,  # type: ignore
                     spans_batched: torch.IntTensor,
                     span_mask_batched,
                     span_embeddings_batched,  # TODO(dwadden) add type.
                     sentence_lengths,
                     ix,
                     coref_labels_batched: torch.IntTensor = None,
                     metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Run the forward pass for a single document.

        Important: This function assumes that sentences are going to be passed in in sorted order,
        from the same document.
        """
        # TODO(dwadden) How to handle case where only one span from a cluster makes it into the
        # minibatch? Should I get rid of the cluster?
        # TODO(dwadden) Write quick unit tests for correctness, time permitting.
        span_ix = span_mask_batched.view(-1).nonzero().squeeze()  # Indices of the spans to keep.
        spans, span_embeddings = self._flatten_spans(
            spans_batched, span_ix, span_embeddings_batched, sentence_lengths)
        coref_labels = self._flatten_coref_labels(coref_labels_batched, span_ix)

        document_length = sentence_lengths.sum().item()
        num_spans = spans.size(1)

        # Prune based on mention scores. Make sure we keep at least 1.
        num_spans_to_keep = max(2, int(math.ceil(self._spans_per_word * document_length)))

        # Since there's only one minibatch, there aren't any masked spans for us. The span mask is
        # always 1.
        span_mask = torch.ones(num_spans, device=spans_batched.device).unsqueeze(0)
        (top_span_embeddings, top_span_mask,
         top_span_indices, top_span_mention_scores, num_items_kept) = self._mention_pruner(
             span_embeddings, span_mask, num_spans_to_keep)
        top_span_mask = top_span_mask.unsqueeze(-1)
        # Shape: (batch_size * num_spans_to_keep)
        flat_top_span_indices = util.flatten_and_batch_shift_indices(top_span_indices, num_spans)

        # Compute final predictions for which spans to consider as mentions.
        # Shape: (batch_size, num_spans_to_keep, 2)
        top_spans = util.batched_index_select(spans,
                                              top_span_indices,
                                              flat_top_span_indices)

        # Compute indices for antecedent spans to consider.
        max_antecedents = min(self._max_antecedents, num_spans_to_keep)

        # Shapes:
        # (num_spans_to_keep, max_antecedents),
        # (1, max_antecedents),
        # (1, num_spans_to_keep, max_antecedents)
        valid_antecedent_indices, valid_antecedent_offsets, valid_antecedent_log_mask = \
            self._generate_valid_antecedents(num_spans_to_keep, max_antecedents, util.get_device_of(span_embeddings))

        coreference_scores = self.get_coref_scores(top_span_embeddings, top_span_mention_scores,
            valid_antecedent_indices, valid_antecedent_offsets, valid_antecedent_log_mask)

        output_dict = {"top_spans": top_spans,
                       "antecedent_indices": valid_antecedent_indices,
                       "valid_antecedent_log_mask": valid_antecedent_log_mask,
                       "valid_antecedent_offsets": valid_antecedent_offsets,
                       "top_span_indices": top_span_indices,
                       "top_span_mask": top_span_mask,
                       "top_span_embeddings": top_span_embeddings,
                       "flat_top_span_indices": flat_top_span_indices,
                       "coref_labels": coref_labels,
                       "coreference_scores": coreference_scores,
                       "sentence_lengths": sentence_lengths,
                       "span_ix": span_ix,
                       "metadata": metadata}

        return output_dict

    # TODO(Ulme) Split up method here?

    def get_coref_scores(self,
                         top_span_embeddings,
                         top_span_mention_scores,
                         valid_antecedent_indices,
                         valid_antecedent_offsets,
                         valid_antecedent_log_mask):
        candidate_antecedent_embeddings = util.flattened_index_select(top_span_embeddings,
                                                                      valid_antecedent_indices)
        # Shape: (batch_size, num_spans_to_keep, max_antecedents)
        candidate_antecedent_mention_scores = util.flattened_index_select(top_span_mention_scores,
                                                                          valid_antecedent_indices).squeeze(-1)
        # Compute antecedent scores.
        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        span_pair_embeddings = self._compute_span_pair_embeddings(top_span_embeddings,
                                                                  candidate_antecedent_embeddings,
                                                                  valid_antecedent_offsets)
        # Shape: (batch_size, num_spans_to_keep, 1 + max_antecedents)
        coreference_scores = self._compute_coreference_scores(span_pair_embeddings,
                                                              top_span_mention_scores,
                                                              candidate_antecedent_mention_scores,
                                                              valid_antecedent_log_mask)
        return coreference_scores

    def predict_labels_doc(self, output_dict):
        # Shape: (batch_size, num_spans_to_keep)
        coref_labels = output_dict["coref_labels"]
        coreference_scores = output_dict["coreference_scores"]
        _, predicted_antecedents = coreference_scores.max(2)
        # Subtract one here because index 0 is the "no antecedent" class,
        # so this makes the indices line up with actual spans if the prediction
        # is greater than -1.
        predicted_antecedents -= 1

        output_dict["predicted_antecedents"] = predicted_antecedents

        top_span_indices = output_dict["top_span_indices"]
        flat_top_span_indices = output_dict["flat_top_span_indices"]
        valid_antecedent_indices = output_dict["antecedent_indices"]
        valid_antecedent_log_mask = output_dict["valid_antecedent_log_mask"]
        top_spans = output_dict["top_spans"]
        top_span_mask = output_dict["top_span_mask"]
        metadata = output_dict["metadata"]
        sentence_lengths = output_dict["sentence_lengths"]

        if coref_labels is not None:
            # Find the gold labels for the spans which we kept.
            pruned_gold_labels = util.batched_index_select(coref_labels.unsqueeze(-1),
                                                           top_span_indices,
                                                           flat_top_span_indices)

            antecedent_labels = util.flattened_index_select(pruned_gold_labels,
                                                            valid_antecedent_indices).squeeze(-1)
            # There's an integer wrap-around happening here. It occurs in the original code.
            antecedent_labels += valid_antecedent_log_mask.long()

            # Compute labels.
            # Shape: (batch_size, num_spans_to_keep, max_antecedents + 1)
            gold_antecedent_labels = self._compute_antecedent_gold_labels(pruned_gold_labels,
                                                                          antecedent_labels)
            # Now, compute the loss using the negative marginal log-likelihood.
            coreference_log_probs = util.masked_log_softmax(coreference_scores, top_span_mask)
            correct_antecedent_log_probs = coreference_log_probs + gold_antecedent_labels.log()
            negative_marginal_log_likelihood = -util.logsumexp(correct_antecedent_log_probs).sum()

            # Need to get cluster data in same form as for original AllenNLP coref code so that the
            # evaluation code works.
            evaluation_metadata = self._make_evaluation_metadata(metadata, sentence_lengths)

            self._mention_recall(top_spans, evaluation_metadata)
            self._conll_coref_scores(
                top_spans, valid_antecedent_indices, predicted_antecedents, evaluation_metadata)

            output_dict["loss"] = negative_marginal_log_likelihood

        if metadata is not None:
            output_dict["document"] = [x["sentence"] for x in metadata]
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]):
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
        mention_recall = self._mention_recall.get_metric(reset)
        coref_precision, coref_recall, coref_f1 = self._conll_coref_scores.get_metric(reset)

        return {"coref_precision": coref_precision,
                "coref_recall": coref_recall,
                "coref_f1": coref_f1,
                "coref_mention_recall": mention_recall}

    @staticmethod
    def _generate_valid_antecedents(num_spans_to_keep: int,
                                    max_antecedents: int,
                                    device: int) -> Tuple[torch.IntTensor,
                                                          torch.IntTensor,
                                                          torch.FloatTensor]:
        """
        This method generates possible antecedents per span which survived the pruning
        stage. This procedure is `generic across the batch`. The reason this is the case is
        that each span in a batch can be coreferent with any previous span, but here we
        are computing the possible `indices` of these spans. So, regardless of the batch,
        the 1st span _cannot_ have any antecedents, because there are none to select from.
        Similarly, each element can only predict previous spans, so this returns a matrix
        of shape (num_spans_to_keep, max_antecedents), where the (i,j)-th index is equal to
        (i - 1) - j if j <= i, or zero otherwise.

        Parameters
        ----------
        num_spans_to_keep : ``int``, required.
            The number of spans that were kept while pruning.
        max_antecedents : ``int``, required.
            The maximum number of antecedent spans to consider for every span.
        device: ``int``, required.
            The CUDA device to use.

        Returns
        -------
        valid_antecedent_indices : ``torch.IntTensor``
            The indices of every antecedent to consider with respect to the top k spans.
            Has shape ``(num_spans_to_keep, max_antecedents)``.
        valid_antecedent_offsets : ``torch.IntTensor``
            The distance between the span and each of its antecedents in terms of the number
            of considered spans (i.e not the word distance between the spans).
            Has shape ``(1, max_antecedents)``.
        valid_antecedent_log_mask : ``torch.FloatTensor``
            The logged mask representing whether each antecedent span is valid. Required since
            different spans have different numbers of valid antecedents. For example, the first
            span in the document should have no valid antecedents.
            Has shape ``(1, num_spans_to_keep, max_antecedents)``.
        """
        # Shape: (num_spans_to_keep, 1)
        target_indices = util.get_range_vector(num_spans_to_keep, device).unsqueeze(1)

        # Shape: (1, max_antecedents)
        valid_antecedent_offsets = (util.get_range_vector(max_antecedents, device) + 1).unsqueeze(0)

        # This is a broadcasted subtraction.
        # Shape: (num_spans_to_keep, max_antecedents)
        raw_antecedent_indices = target_indices - valid_antecedent_offsets

        # Shape: (1, num_spans_to_keep, max_antecedents)
        valid_antecedent_log_mask = (raw_antecedent_indices >= 0).float().unsqueeze(0).log()

        # Shape: (num_spans_to_keep, max_antecedents)
        valid_antecedent_indices = F.relu(raw_antecedent_indices.float()).long()
        return valid_antecedent_indices, valid_antecedent_offsets, valid_antecedent_log_mask

    def _compute_span_pair_embeddings(self,
                                      top_span_embeddings: torch.FloatTensor,
                                      antecedent_embeddings: torch.FloatTensor,
                                      antecedent_offsets: torch.FloatTensor):
        """
        Computes an embedding representation of pairs of spans for the pairwise scoring function
        to consider. This includes both the original span representations, the element-wise
        similarity of the span representations, and an embedding representation of the distance
        between the two spans.

        Parameters
        ----------
        top_span_embeddings : ``torch.FloatTensor``, required.
            Embedding representations of the top spans. Has shape
            (batch_size, num_spans_to_keep, embedding_size).
        antecedent_embeddings : ``torch.FloatTensor``, required.
            Embedding representations of the antecedent spans we are considering
            for each top span. Has shape
            (batch_size, num_spans_to_keep, max_antecedents, embedding_size).
        antecedent_offsets : ``torch.IntTensor``, required.
            The offsets between each top span and its antecedent spans in terms
            of spans we are considering. Has shape (1, max_antecedents).

        Returns
        -------
        span_pair_embeddings : ``torch.FloatTensor``
            Embedding representation of the pair of spans to consider. Has shape
            (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        """
        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        target_embeddings = top_span_embeddings.unsqueeze(2).expand_as(antecedent_embeddings)

        # Shape: (1, max_antecedents, embedding_size)
        antecedent_distance_embeddings = self._distance_embedding(
                util.bucket_values(antecedent_offsets,
                                   num_total_buckets=self._num_distance_buckets))

        # Shape: (1, 1, max_antecedents, embedding_size)
        antecedent_distance_embeddings = antecedent_distance_embeddings.unsqueeze(0)

        expanded_distance_embeddings_shape = (antecedent_embeddings.size(0),
                                              antecedent_embeddings.size(1),
                                              antecedent_embeddings.size(2),
                                              antecedent_distance_embeddings.size(-1))
        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        antecedent_distance_embeddings = antecedent_distance_embeddings.expand(*expanded_distance_embeddings_shape)

        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        span_pair_embeddings = torch.cat([target_embeddings,
                                          antecedent_embeddings,
                                          antecedent_embeddings * target_embeddings,
                                          antecedent_distance_embeddings], -1)
        return span_pair_embeddings

    @staticmethod
    def _compute_antecedent_gold_labels(top_coref_labels: torch.IntTensor,
                                        antecedent_labels: torch.IntTensor):
        """
        Generates a binary indicator for every pair of spans. This label is one if and
        only if the pair of spans belong to the same cluster. The labels are augmented
        with a dummy antecedent at the zeroth position, which represents the prediction
        that a span does not have any antecedent.

        Parameters
        ----------
        top_coref_labels : ``torch.IntTensor``, required.
            The cluster id label for every span. The id is arbitrary,
            as we just care about the clustering. Has shape (batch_size, num_spans_to_keep).
        antecedent_labels : ``torch.IntTensor``, required.
            The cluster id label for every antecedent span. The id is arbitrary,
            as we just care about the clustering. Has shape
            (batch_size, num_spans_to_keep, max_antecedents).

        Returns
        -------
        pairwise_labels_with_dummy_label : ``torch.FloatTensor``
            A binary tensor representing whether a given pair of spans belong to
            the same cluster in the gold clustering.
            Has shape (batch_size, num_spans_to_keep, max_antecedents + 1).

        """
        # Shape: (batch_size, num_spans_to_keep, max_antecedents)
        target_labels = top_coref_labels.expand_as(antecedent_labels)
        same_cluster_indicator = (target_labels == antecedent_labels).float()
        non_dummy_indicator = (target_labels >= 0).float()
        pairwise_labels = same_cluster_indicator * non_dummy_indicator

        # Shape: (batch_size, num_spans_to_keep, 1)
        dummy_labels = (1 - pairwise_labels).prod(-1, keepdim=True)

        # Shape: (batch_size, num_spans_to_keep, max_antecedents + 1)
        pairwise_labels_with_dummy_label = torch.cat([dummy_labels, pairwise_labels], -1)
        return pairwise_labels_with_dummy_label

    def _compute_coreference_scores(self,
                                    pairwise_embeddings: torch.FloatTensor,
                                    top_span_mention_scores: torch.FloatTensor,
                                    antecedent_mention_scores: torch.FloatTensor,
                                    antecedent_log_mask: torch.FloatTensor) -> torch.FloatTensor:
        """
        Computes scores for every pair of spans. Additionally, a dummy label is included,
        representing the decision that the span is not coreferent with anything. For the dummy
        label, the score is always zero. For the true antecedent spans, the score consists of
        the pairwise antecedent score and the unary mention scores for the span and its
        antecedent. The factoring allows the model to blame many of the absent links on bad
        spans, enabling the pruning strategy used in the forward pass.

        Parameters
        ----------
        pairwise_embeddings: ``torch.FloatTensor``, required.
            Embedding representations of pairs of spans. Has shape
            (batch_size, num_spans_to_keep, max_antecedents, encoding_dim)
        top_span_mention_scores: ``torch.FloatTensor``, required.
            Mention scores for every span. Has shape
            (batch_size, num_spans_to_keep, max_antecedents).
        antecedent_mention_scores: ``torch.FloatTensor``, required.
            Mention scores for every antecedent. Has shape
            (batch_size, num_spans_to_keep, max_antecedents).
        antecedent_log_mask: ``torch.FloatTensor``, required.
            The log of the mask for valid antecedents.

        Returns
        -------
        coreference_scores: ``torch.FloatTensor``
            A tensor of shape (batch_size, num_spans_to_keep, max_antecedents + 1),
            representing the unormalised score for each (span, antecedent) pair
            we considered.

        """
        # Shape: (batch_size, num_spans_to_keep, max_antecedents)
        antecedent_scores = self._antecedent_scorer(
                self._antecedent_feedforward(pairwise_embeddings)).squeeze(-1)
        antecedent_scores += top_span_mention_scores + antecedent_mention_scores
        antecedent_scores += antecedent_log_mask

        # Shape: (batch_size, num_spans_to_keep, 1)
        shape = [antecedent_scores.size(0), antecedent_scores.size(1), 1]
        dummy_scores = antecedent_scores.new_zeros(*shape)

        # Shape: (batch_size, num_spans_to_keep, max_antecedents + 1)
        coreference_scores = torch.cat([dummy_scores, antecedent_scores], -1)
        return coreference_scores

    def _flatten_spans(self, spans_batched, span_ix, span_embeddings_batched, sentence_lengths):
        """
        Spans are input with each minibatch as a sentence. For coref, it's easier to flatten them out
        and consider all sentences together as a document.
        """
        # Get feature size and indices of good spans
        feature_size = self._mention_pruner._scorer[0]._module.input_dim

        # Change the span offsets to document-level, flatten, and keep good ones.
        sentence_offset = shared.cumsum_shifted(sentence_lengths).unsqueeze(1).unsqueeze(2)
        spans_offset = spans_batched + sentence_offset
        spans_flat = spans_offset.view(-1, 2)
        spans_flat = spans_flat[span_ix].unsqueeze(0)

        # Flatten the span embeddings and keep the good ones.
        emb_flat = span_embeddings_batched.view(-1, feature_size)
        span_embeddings_flat = emb_flat[span_ix].unsqueeze(0)

        return spans_flat, span_embeddings_flat

    @staticmethod
    def _flatten_coref_labels(coref_labels_batched, span_ix):
        "Flatten the coref labels."
        labels_flat = coref_labels_batched.view(-1)[span_ix]
        labels_flat = labels_flat.unsqueeze(0)
        return labels_flat

    @staticmethod
    def _make_evaluation_metadata(metadata, sentence_lengths):
        """
        Get cluster metadata in form to feed into evaluation scripts. For each entry in minibatch,
        return a dict with a metadata field, which is a list whose entries are lists specifying the
        spans involved in a given cluster.
        For coreference evaluation, we need to make the span indices with respect to the entire
        "document" (i.e. all sentences in minibatch), rather than with respect to each sentence.
        """
        # TODO(dwadden) Write tests to make sure sentence starts match lengths of sentences in
        # metadata.
        # As elsewhere, we assume the batch size will always be 1.
        cluster_dict = {}
        sentence_offset = shared.cumsum_shifted(sentence_lengths).tolist()
        for entry, sentence_start in zip(metadata, sentence_offset):
            for span, cluster_id in entry["cluster_dict"].items():
                span_offset = (span[0] + sentence_start, span[1] + sentence_start)
                if cluster_id in cluster_dict:
                    cluster_dict[cluster_id].append(span_offset)
                else:
                    cluster_dict[cluster_id] = [span_offset]

        # The `values` method returns an iterator, and I need a list.
        clusters = [val for val in cluster_dict.values()]
        return [dict(clusters=clusters)]
