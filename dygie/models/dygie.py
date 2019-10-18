from os import path
import logging
from typing import Dict, List, Optional
import copy

import torch
import torch.nn.functional as F
from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.common.params import Params
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, FeedForward
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor, EndpointSpanExtractor
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator

# Import submodules.
from dygie.models.coref import CorefResolver
from dygie.models.ner import NERTagger
from dygie.models.relation import RelationExtractor
from dygie.models.events import EventExtractor
from dygie.training.joint_metrics import JointMetrics

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
    lexical_dropout: ``int``
        The probability of dropping out dimensions of the embedded text.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    display_metrics: ``List[str]``. A list of the metrics that should be printed out during model
        training.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 context_layer: Seq2SeqEncoder,
                 modules,  # TODO(dwadden) Add type.
                 feature_size: int,
                 max_span_width: int,
                 loss_weights: Dict[str, int],
                 lexical_dropout: float = 0.2,
                 lstm_dropout: float = 0.4,
                 use_attentive_span_extractor: bool = False,
                 co_train: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 display_metrics: List[str] = None) -> None:
        super(DyGIE, self).__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._context_layer = context_layer

        self._loss_weights = loss_weights
        self._permanent_loss_weights = copy.deepcopy(self._loss_weights)

        # Need to add this line so things don't break. TODO(dwadden) sort out what's happening.
        modules = Params(modules)
        self._coref = CorefResolver.from_params(vocab=vocab,
                                                feature_size=feature_size,
                                                params=modules.pop("coref"))
        self._ner = NERTagger.from_params(vocab=vocab,
                                          feature_size=feature_size,
                                          params=modules.pop("ner"))
        self._relation = RelationExtractor.from_params(vocab=vocab,
                                                       feature_size=feature_size,
                                                       params=modules.pop("relation"))
        self._events = EventExtractor.from_params(vocab=vocab,
                                                  feature_size=feature_size,
                                                  params=modules.pop("events"))

        # Make endpoint span extractor.

        self._endpoint_span_extractor = EndpointSpanExtractor(context_layer.get_output_dim(),
                                                              combination="x,y",
                                                              num_width_embeddings=max_span_width,
                                                              span_width_embedding_dim=feature_size,
                                                              bucket_widths=False)
        if use_attentive_span_extractor:
            self._attentive_span_extractor = SelfAttentiveSpanExtractor(
                input_dim=text_field_embedder.get_output_dim())
        else:
            self._attentive_span_extractor = None

        self._max_span_width = max_span_width

        self._display_metrics = display_metrics

        if lexical_dropout > 0:
            self._lexical_dropout = torch.nn.Dropout(p=lexical_dropout)
        else:
            self._lexical_dropout = lambda x: x

        # Do co-training if we're training on ACE and ontonotes.
        self._co_train = co_train

        # Big gotcha: PyTorch doesn't add dropout to the LSTM's output layer. We need to do this
        # manually.
        if lstm_dropout > 0:
            self._lstm_dropout = torch.nn.Dropout(p=lstm_dropout)
        else:
            self._lstm_dropout = lambda x: x

        initializer(self)

    @overrides
    def forward(self,
                text,
                spans,
                ner_labels,
                coref_labels,
                relation_labels,
                trigger_labels,
                argument_labels,
                metadata):
        """
        TODO(dwadden) change this.
        """
        # For co-training on Ontonotes, need to change the loss weights depending on the data coming
        # in. This is a hack but it will do for now.
        if self._co_train:
            if self.training:
                dataset = [entry["dataset"] for entry in metadata]
                assert len(set(dataset)) == 1
                dataset = dataset[0]
                assert dataset in ["ace", "ontonotes"]
                if dataset == "ontonotes":
                    self._loss_weights = dict(coref=1, ner=0, relation=0, events=0)
                else:
                    self._loss_weights = self._permanent_loss_weights
            # This assumes that there won't be any co-training data in the dev and test sets, and that
            # coref propagation will still happen even when the coref weight is set to 0.
            else:
                self._loss_weights = self._permanent_loss_weights

        # In AllenNLP, AdjacencyFields are passed in as floats. This fixes it.
        relation_labels = relation_labels.long()
        argument_labels = argument_labels.long()

        # If we're doing Bert, get the sentence class token as part of the text embedding. This will
        # break if we use Bert together with other embeddings, but that won't happen much.
        if "bert-offsets" in text:
            offsets = text["bert-offsets"]
            sent_ix = torch.zeros(offsets.size(0), device=offsets.device, dtype=torch.long).unsqueeze(1)
            padded_offsets = torch.cat([sent_ix, offsets], dim=1)
            text["bert-offsets"] = padded_offsets
            padded_embeddings = self._text_field_embedder(text)
            cls_embeddings = padded_embeddings[:, 0, :]
            text_embeddings = padded_embeddings[:, 1:, :]
        else:
            text_embeddings = self._text_field_embedder(text)
            cls_embeddings = torch.zeros([text_embeddings.size(0), text_embeddings.size(2)],
                                         device=text_embeddings.device)

        text_embeddings = self._lexical_dropout(text_embeddings)

        # Shape: (batch_size, max_sentence_length)
        text_mask = util.get_text_field_mask(text).float()
        sentence_group_lengths = text_mask.sum(dim=1).long()

        sentence_lengths = 0*text_mask.sum(dim=1).long()
        for i in range(len(metadata)):
            sentence_lengths[i] = metadata[i]["end_ix"] - metadata[i]["start_ix"]
            for k in range(sentence_lengths[i], sentence_group_lengths[i]):
                text_mask[i][k] = 0

        max_sentence_length = sentence_lengths.max().item()

        # TODO(Ulme) Speed this up by tensorizing
        new_text_embeddings = torch.zeros([text_embeddings.shape[0], max_sentence_length, text_embeddings.shape[2]], device=text_embeddings.device)
        for i in range(len(new_text_embeddings)):
            new_text_embeddings[i][0:metadata[i]["end_ix"] - metadata[i]["start_ix"]] = text_embeddings[i][metadata[i]["start_ix"]:metadata[i]["end_ix"]]

        #max_sent_len = max(sentence_lengths)
        #the_list = [list(k+metadata[i]["start_ix"] if k < max_sent_len else 0 for k in range(text_embeddings.shape[1])) for i in range(len(metadata))]
        #import ipdb; ipdb.set_trace()
        #text_embeddings = torch.gather(text_embeddings, 1, torch.tensor(the_list, device=text_embeddings.device).unsqueeze(2).repeat(1, 1, text_embeddings.shape[2]))
        text_embeddings = new_text_embeddings

        # Only keep the text embeddings that correspond to actual tokens.
        # text_embeddings = text_embeddings[:, :max_sentence_length, :].contiguous()
        text_mask = text_mask[:, :max_sentence_length].contiguous()

        # Shape: (batch_size, max_sentence_length, encoding_dim)
        contextualized_embeddings = self._lstm_dropout(self._context_layer(text_embeddings, text_mask))
        assert spans.max() < contextualized_embeddings.shape[1]

        if self._attentive_span_extractor is not None:
            # Shape: (batch_size, num_spans, emebedding_size)
            attended_span_embeddings = self._attentive_span_extractor(text_embeddings, spans)

        # Shape: (batch_size, num_spans)
        span_mask = (spans[:, :, 0] >= 0).float()
        # SpanFields return -1 when they are used as padding. As we do
        # some comparisons based on span widths when we attend over the
        # span representations that we generate from these indices, we
        # need them to be <= 0. This is only relevant in edge cases where
        # the number of spans we consider after the pruning stage is >= the
        # total number of spans, because in this case, it is possible we might
        # consider a masked span.
        # Shape: (batch_size, num_spans, 2)
        spans = F.relu(spans.float()).long()

        # Shape: (batch_size, num_spans, 2 * encoding_dim + feature_size)
        endpoint_span_embeddings = self._endpoint_span_extractor(contextualized_embeddings, spans)

        if self._attentive_span_extractor is not None:
            # Shape: (batch_size, num_spans, emebedding_size + 2 * encoding_dim + feature_size)
            span_embeddings = torch.cat([endpoint_span_embeddings, attended_span_embeddings], -1)
        else:
            span_embeddings = endpoint_span_embeddings

        # TODO(Ulme) try normalizing span embeddeings
        #span_embeddings = span_embeddings.abs().sum(dim=-1).unsqueeze(-1)

        # Make calls out to the modules to get results.
        output_coref = {'loss': 0}
        output_ner = {'loss': 0}
        output_relation = {'loss': 0}
        output_events = {'loss': 0}

        # Prune and compute span representations for coreference module
        if self._loss_weights["coref"] > 0 or self._coref.coref_prop > 0:
            output_coref, coref_indices = self._coref.compute_representations(
                spans, span_mask, span_embeddings, sentence_lengths, coref_labels, metadata)

        # Prune and compute span representations for relation module
        if self._loss_weights["relation"] > 0 or self._relation.rel_prop > 0:
            output_relation = self._relation.compute_representations(
                spans, span_mask, span_embeddings, sentence_lengths, relation_labels, metadata)

        # Propagation of global information to enhance the span embeddings
        if self._coref.coref_prop > 0:
            # TODO(Ulme) Implement Coref Propagation
            output_coref = self._coref.coref_propagation(output_coref)
            span_embeddings = self._coref.update_spans(output_coref, span_embeddings, coref_indices)

        if self._relation.rel_prop > 0:
            output_relation = self._relation.relation_propagation(output_relation)
            span_embeddings = self.update_span_embeddings(span_embeddings, span_mask,
                output_relation["top_span_embeddings"], output_relation["top_span_mask"],
                output_relation["top_span_indices"])

        # Make predictions and compute losses for each module
        if self._loss_weights['ner'] > 0:
            output_ner = self._ner(
                spans, span_mask, span_embeddings, sentence_lengths, ner_labels, metadata)

        if self._loss_weights['coref'] > 0:
            output_coref = self._coref.predict_labels(output_coref, metadata)

        if self._loss_weights['relation'] > 0:
            output_relation = self._relation.predict_labels(relation_labels, output_relation, metadata)

        if self._loss_weights['events'] > 0:
            # Make the trigger embeddings the same size as the argument embeddings to make
            # propagation easier.
            if self._events._span_prop._n_span_prop > 0:
                trigger_embeddings = contextualized_embeddings.repeat(1, 1, 2)
                trigger_widths = torch.zeros([trigger_embeddings.size(0), trigger_embeddings.size(1)],
                                             device=trigger_embeddings.device, dtype=torch.long)
                trigger_width_embs = self._endpoint_span_extractor._span_width_embedding(trigger_widths)
                trigger_width_embs = trigger_width_embs.detach()
                trigger_embeddings = torch.cat([trigger_embeddings, trigger_width_embs], dim=-1)
            else:
                trigger_embeddings = contextualized_embeddings

            output_events = self._events(
                text_mask, trigger_embeddings, spans, span_mask, span_embeddings, cls_embeddings,
                sentence_lengths, output_ner, trigger_labels, argument_labels,
                ner_labels, metadata)

        if "loss" not in output_coref:
            output_coref["loss"] = 0
        if "loss" not in output_relation:
            output_relation["loss"] = 0

        # TODO(dwadden) just did this part.
        loss = (self._loss_weights['coref'] * output_coref['loss'] +
                self._loss_weights['ner'] * output_ner['loss'] +
                self._loss_weights['relation'] * output_relation['loss'] +
                self._loss_weights['events'] * output_events['loss'])

        output_dict = dict(coref=output_coref,
                           relation=output_relation,
                           ner=output_ner,
                           events=output_events)
        output_dict['loss'] = loss

        # Check to see if event predictions are globally compatible (argument labels are compatible
        # with NER tags and trigger tags).
        # if self._loss_weights["ner"] > 0 and self._loss_weights["events"] > 0:
        #     decoded_ner = self._ner.decode(output_dict["ner"])
        #     decoded_events = self._events.decode(output_dict["events"])
        #     self._joint_metrics(decoded_ner, decoded_events)

        return output_dict

    def update_span_embeddings(self, span_embeddings, span_mask, top_span_embeddings, top_span_mask, top_span_indices):
        # TODO(Ulme) Speed this up by tensorizing

        new_span_embeddings = span_embeddings.clone()
        for sample_nr in range(len(top_span_mask)):
            for top_span_nr, span_nr in enumerate(top_span_indices[sample_nr]):
                if top_span_mask[sample_nr, top_span_nr] == 0 or span_mask[sample_nr, span_nr] == 0:
                    break
                new_span_embeddings[sample_nr, span_nr] = top_span_embeddings[sample_nr, top_span_nr]
        return new_span_embeddings

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
        # TODO(dwadden) which things are already decoded?
        res = {}
        if self._loss_weights["coref"] > 0:
            res["coref"] = self._coref.decode(output_dict["coref"])
        if self._loss_weights["ner"] > 0:
            res["ner"] = self._ner.decode(output_dict["ner"])
        if self._loss_weights["relation"] > 0:
            res["relation"] = self._relation.decode(output_dict["relation"])
        if self._loss_weights["events"] > 0:
            res["events"] = output_dict["events"]

        return res

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """
        Get all metrics from all modules. For the ones that shouldn't be displayed, prefix their
        keys with an underscore.
        """
        metrics_coref = self._coref.get_metrics(reset=reset)
        metrics_ner = self._ner.get_metrics(reset=reset)
        metrics_relation = self._relation.get_metrics(reset=reset)
        metrics_events = self._events.get_metrics(reset=reset)
        if self._loss_weights["ner"] > 0 and self._loss_weights["events"] > 0:
            metrics_joint = self._joint_metrics.get_metric(reset=reset)
        else:
            metrics_joint = {}

        # Make sure that there aren't any conflicting names.
        metric_names = (list(metrics_coref.keys()) + list(metrics_ner.keys()) +
                        list(metrics_relation.keys()) + list(metrics_events.keys()))
        assert len(set(metric_names)) == len(metric_names)
        all_metrics = dict(list(metrics_coref.items()) +
                           list(metrics_ner.items()) +
                           list(metrics_relation.items()) +
                           list(metrics_events.items()) +
                           list(metrics_joint.items()))

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
