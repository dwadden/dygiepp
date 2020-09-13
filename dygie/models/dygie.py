import logging
from typing import Dict, List, Optional, Union
import copy

import torch
import torch.nn.functional as F
from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.common.params import Params
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, FeedForward, TimeDistributed
from allennlp.modules.span_extractors import EndpointSpanExtractor
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator

# Import submodules.
from dygie.models.coref import CorefResolver
from dygie.models.ner import NERTagger
from dygie.models.relation import RelationExtractor
from dygie.models.events import EventExtractor
from dygie.data.dataset_readers import document

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("dygie")
class DyGIE(Model):
    """
    TODO(dwadden) document me.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``text`` ``TextField`` we get as input to the model.
    context_layer : ``Seq2SeqEncoder``
        This layer incorporates contextual information for each word in the document.
    feature_size: ``int``
        The embedding size for all the embedded features, such as distances or span widths.
    submodule_params: ``TODO(dwadden)``
        A nested dictionary specifying parameters to be passed on to initialize submodules.
    max_span_width: ``int``
        The maximum width of candidate spans.
    target_task: ``str``:
        The task used to make early stopping decisions.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    module_initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the individual modules.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    display_metrics: ``List[str]``. A list of the metrics that should be printed out during model
        training.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 modules,  # TODO(dwadden) Add type.
                 feature_size: int,
                 max_span_width: int,
                 target_task: str,
                 feedforward_params: Dict[str, Union[int, float]],
                 loss_weights: Dict[str, float],
                 initializer: InitializerApplicator = InitializerApplicator(),
                 module_initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 display_metrics: List[str] = None) -> None:
        super(DyGIE, self).__init__(vocab, regularizer)

        ####################

        # Create span extractor.
        self._endpoint_span_extractor = EndpointSpanExtractor(
            embedder.get_output_dim(),
            combination="x,y",
            num_width_embeddings=max_span_width,
            span_width_embedding_dim=feature_size,
            bucket_widths=False)

        ####################

        # Set parameters.
        self._embedder = embedder
        self._loss_weights = loss_weights
        self._max_span_width = max_span_width
        self._display_metrics = self._get_display_metrics(target_task)
        token_emb_dim = self._embedder.get_output_dim()
        span_emb_dim = self._endpoint_span_extractor.get_output_dim()

        ####################

        # Create submodules.

        modules = Params(modules)

        # Helper function to create feedforward networks.
        def make_feedforward(input_dim):
            return FeedForward(input_dim=input_dim,
                               num_layers=feedforward_params["num_layers"],
                               hidden_dims=feedforward_params["hidden_dims"],
                               activations=torch.nn.ReLU(),
                               dropout=feedforward_params["dropout"])

        # Submodules

        self._ner = NERTagger.from_params(vocab=vocab,
                                          make_feedforward=make_feedforward,
                                          span_emb_dim=span_emb_dim,
                                          feature_size=feature_size,
                                          params=modules.pop("ner"))

        self._coref = CorefResolver.from_params(vocab=vocab,
                                                make_feedforward=make_feedforward,
                                                span_emb_dim=span_emb_dim,
                                                feature_size=feature_size,
                                                params=modules.pop("coref"))

        self._relation = RelationExtractor.from_params(vocab=vocab,
                                                       make_feedforward=make_feedforward,
                                                       span_emb_dim=span_emb_dim,
                                                       feature_size=feature_size,
                                                       params=modules.pop("relation"))

        self._events = EventExtractor.from_params(vocab=vocab,
                                                  make_feedforward=make_feedforward,
                                                  token_emb_dim=token_emb_dim,
                                                  span_emb_dim=span_emb_dim,
                                                  feature_size=feature_size,
                                                  params=modules.pop("events"))

        ####################

        # Initialize text embedder and all submodules
        for module in [self._ner, self._coref, self._relation, self._events]:
            module_initializer(module)

        initializer(self)

    @staticmethod
    def _get_display_metrics(target_task):
        """
        The `target` is the name of the task used to make early stopping decisions. Show metrics
        related to this task.
        """
        lookup = {
            "ner": [f"MEAN__{name}" for name in
                    ["ner_precision", "ner_recall", "ner_f1"]],
            "relation": [f"MEAN__{name}" for name in
                         ["relation_precision", "relation_recall", "relation_f1"]],
            "coref": ["coref_precision", "coref_recall", "coref_f1", "coref_mention_recall"],
            "events": [f"MEAN__{name}" for name in
                       ["trig_class_f1", "arg_class_f1"]]}
        if target_task not in lookup:
            raise ValueError(f"Invalied value {target_task} has been given as the target task.")
        return lookup[target_task]

    @staticmethod
    def _debatch(x):
        # TODO(dwadden) Get rid of this when I find a better way to do it.
        return x if x is None else x.squeeze(0)

    @overrides
    def forward(self,
                text,
                spans,
                metadata,
                ner_labels=None,
                coref_labels=None,
                relation_labels=None,
                trigger_labels=None,
                argument_labels=None):
        """
        TODO(dwadden) change this.
        """
        # In AllenNLP, AdjacencyFields are passed in as floats. This fixes it.
        if relation_labels is not None:
            relation_labels = relation_labels.long()
        if argument_labels is not None:
            argument_labels = argument_labels.long()

        # TODO(dwadden) Multi-document minibatching isn't supported yet. For now, get rid of the
        # extra dimension in the input tensors. Will return to this once the model runs.
        if len(metadata) > 1:
            raise NotImplementedError("Multi-document minibatching not yet supported.")

        metadata = metadata[0]
        spans = self._debatch(spans)  # (n_sents, max_n_spans, 2)
        ner_labels = self._debatch(ner_labels)  # (n_sents, max_n_spans)
        coref_labels = self._debatch(coref_labels)  #  (n_sents, max_n_spans)
        relation_labels = self._debatch(relation_labels)  # (n_sents, max_n_spans, max_n_spans)
        trigger_labels = self._debatch(trigger_labels)  # TODO(dwadden)
        argument_labels = self._debatch(argument_labels)  # TODO(dwadden)

        # Encode using BERT, then debatch.
        # Since the data are batched, we use `num_wrapping_dims=1` to unwrap the document dimension.
        # (1, n_sents, max_sententence_length, embedding_dim)

        # TODO(dwadden) Deal with the case where the input is longer than 512.
        text_embeddings = self._embedder(text, num_wrapping_dims=1)
        # (n_sents, max_n_wordpieces, embedding_dim)
        text_embeddings = self._debatch(text_embeddings)

        # (n_sents, max_sentence_length)
        text_mask = self._debatch(util.get_text_field_mask(text, num_wrapping_dims=1).float())
        sentence_lengths = text_mask.sum(dim=1).long()  # (n_sents)

        span_mask = (spans[:, :, 0] >= 0).float()  # (n_sents, max_n_spans)
        # SpanFields return -1 when they are used as padding. As we do some comparisons based on
        # span widths when we attend over the span representations that we generate from these
        # indices, we need them to be <= 0. This is only relevant in edge cases where the number of
        # spans we consider after the pruning stage is >= the total number of spans, because in this
        # case, it is possible we might consider a masked span.
        spans = F.relu(spans.float()).long()  # (n_sents, max_n_spans, 2)

        # Shape: (batch_size, num_spans, 2 * encoding_dim + feature_size)
        span_embeddings = self._endpoint_span_extractor(text_embeddings, spans)

        # Make calls out to the modules to get results.
        output_coref = {'loss': 0}
        output_ner = {'loss': 0}
        output_relation = {'loss': 0}
        output_events = {'loss': 0}

        # Prune and compute span representations for coreference module
        if self._loss_weights["coref"] > 0 or self._coref.coref_prop > 0:
            output_coref, coref_indices = self._coref.compute_representations(
                spans, span_mask, span_embeddings, sentence_lengths, coref_labels, metadata)

        # Propagation of global information to enhance the span embeddings
        if self._coref.coref_prop > 0:
            output_coref = self._coref.coref_propagation(output_coref)
            span_embeddings = self._coref.update_spans(
                output_coref, span_embeddings, coref_indices)

        # Make predictions and compute losses for each module
        if self._loss_weights['ner'] > 0:
            output_ner = self._ner(
                spans, span_mask, span_embeddings, sentence_lengths, ner_labels, metadata)

        if self._loss_weights['coref'] > 0:
            output_coref = self._coref.predict_labels(output_coref, metadata)

        if self._loss_weights['relation'] > 0:
            output_relation = self._relation(
                spans, span_mask, span_embeddings, sentence_lengths, relation_labels, metadata)

        if self._loss_weights['events'] > 0:
            # The `text_embeddings` serve as representations for event triggers.
            output_events = self._events(
                text_mask, text_embeddings, spans, span_mask, span_embeddings,
                sentence_lengths, trigger_labels, argument_labels,
                ner_labels, metadata)

        # Use `get` since there are some cases where the output dict won't have a loss - for
        # instance, when doing prediction.
        loss = (self._loss_weights['coref'] * output_coref.get("loss", 0) +
                self._loss_weights['ner'] * output_ner.get("loss", 0) +
                self._loss_weights['relation'] * output_relation.get("loss", 0) +
                self._loss_weights['events'] * output_events.get("loss", 0))

        # Multiply the loss by the weight multiplier for this document.
        weight = metadata.weight if metadata.weight is not None else 1.0
        loss *= torch.tensor(weight)

        output_dict = dict(coref=output_coref,
                           relation=output_relation,
                           ner=output_ner,
                           events=output_events)
        output_dict['loss'] = loss

        output_dict["metadata"] = metadata

        return output_dict

    def update_span_embeddings(self, span_embeddings, span_mask, top_span_embeddings,
                               top_span_mask, top_span_indices):
        # TODO(Ulme) Speed this up by tensorizing

        new_span_embeddings = span_embeddings.clone()
        for sample_nr in range(len(top_span_mask)):
            for top_span_nr, span_nr in enumerate(top_span_indices[sample_nr]):
                if top_span_mask[sample_nr, top_span_nr] == 0 or span_mask[sample_nr, span_nr] == 0:
                    break
                new_span_embeddings[sample_nr,
                                    span_nr] = top_span_embeddings[sample_nr, top_span_nr]
        return new_span_embeddings

    @overrides
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]):
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

        doc = copy.deepcopy(output_dict["metadata"])

        if self._loss_weights["coref"] > 0:
            # TODO(dwadden) Will need to get rid of the [0] when batch training is enabled.
            decoded_coref = self._coref.make_output_human_readable(output_dict["coref"])["predicted_clusters"][0]
            sentences = doc.sentences
            sentence_starts = [sent.sentence_start for sent in sentences]
            predicted_clusters = [document.Cluster(entry, i, sentences, sentence_starts)
                                  for i, entry in enumerate(decoded_coref)]
            doc.predicted_clusters = predicted_clusters
            # TODO(dwadden) update the sentences with cluster information.

        if self._loss_weights["ner"] > 0:
            for predictions, sentence in zip(output_dict["ner"]["predictions"], doc):
                sentence.predicted_ner = predictions

        if self._loss_weights["relation"] > 0:
            for predictions, sentence in zip(output_dict["relation"]["predictions"], doc):
                sentence.predicted_relations = predictions

        if self._loss_weights["events"] > 0:
            for predictions, sentence in zip(output_dict["events"]["predictions"], doc):
                sentence.predicted_events = predictions

        return doc

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """
        Get all metrics from all modules. For the ones that shouldn't be displayed, prefix their
        keys with an underscore.
        """
        metrics_coref = self._coref.get_metrics(reset=reset)
        metrics_ner = self._ner.get_metrics(reset=reset)
        metrics_relation = self._relation.get_metrics(reset=reset)
        metrics_events = self._events.get_metrics(reset=reset)

        # Make sure that there aren't any conflicting names.
        metric_names = (list(metrics_coref.keys()) + list(metrics_ner.keys()) +
                        list(metrics_relation.keys()) + list(metrics_events.keys()))
        assert len(set(metric_names)) == len(metric_names)
        all_metrics = dict(list(metrics_coref.items()) +
                           list(metrics_ner.items()) +
                           list(metrics_relation.items()) +
                           list(metrics_events.items()))

        # If no list of desired metrics given, display them all.
        if self._display_metrics is None:
            return all_metrics
        # Otherwise only display the selected ones.
        res = {}
        for k, v in all_metrics.items():
            if k in self._display_metrics:
                res[k] = v
            else:
                new_k = "_" + k
                res[new_k] = v
        return res
