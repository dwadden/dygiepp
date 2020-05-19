from typing import List
import numpy as np
from copy import deepcopy

from overrides import overrides
import numpy
import json

from allennlp.common.util import JsonDict
from allennlp.nn import util
from allennlp.data.dataset import Batch
from allennlp.data import DatasetReader
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from allennlp.data.fields import AdjacencyField
from allennlp.data.instance import Instance

from dygie.interpret.dygie import DygieInterpreter

import sys
from IPython.core.ultratb import FormattedTB
sys.excepthook = FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)


@Predictor.register("dygie")
class DyGIEPredictor(Predictor):
    """
    Predictor for DyGIE model.

    If model was trained on coref, prediction is done on a whole document at
    once. This risks overflowing memory on large documents.
    If the model was trained without coref, prediction is done by sentence.
    """
    def __init__(
            self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        # After the code was finished, I realized that to do prediction we need
        # to load in entire documents as a single instance. I added a
        # `predict_hack` flag to `ie_json.py`. When set to True, it yields full
        # documents instead of sentences.
        self._dataset_reader._predict_hack = True
        self._decode_fields = dict(coref="clusters",
                                   ner="decoded_ner",
                                   relation="decoded_relations",
                                   events="decoded_events")
        self._decode_names = dict(coref="predicted_clusters",
                                  ner="predicted_ner",
                                  relation="predicted_relations",
                                  events="predicted_events")
        # Model to interpret the predictions.
        self._interpreter = DygieInterpreter(self)

    def predict(self, document):
        return self.predict_json({"document": document})

    def predict_tokenized(self, tokenized_document: List[str]) -> JsonDict:
        instance = self._words_list_to_instance(tokenized_document)
        return self.predict_instance(instance)

    @overrides
    def dump_line(self, outputs):
        # Need to override to tell Python how to deal with Numpy ints.
        return json.dumps(outputs, default=int) + "\n"

    @overrides
    def predict_instance(self, instance):
        """
        An instance is an entire document, represented as a list of sentences.
        """
        model = self._model
        cuda_device = model._get_prediction_device()

        doc_keys = [entry["metadata"]["doc_key"] for entry in instance]
        assert len(set(doc_keys)) == 1
        doc_key = doc_keys[0]

        sentence_lengths = [len(entry["metadata"]["sentence"]) for entry in instance]
        sentence_starts = np.cumsum(sentence_lengths)
        sentence_starts = np.roll(sentence_starts, 1)
        sentence_starts[0] = 0

        decoded_instance = {x: [] for x in self._decode_fields}

        # If we're doing coref, predict on the whole document together. This may
        # run out of memory. Otherwise just predict a sentence at a time.
        if self._model._loss_weights["coref"]:
            batches = [Batch(instance)]
        else:
            batches = [Batch([sentence]) for sentence in instance]

        for dataset in batches:
            dataset.index_instances(model.vocab)
            model_input = util.move_to_device(dataset.as_tensor_dict(), cuda_device)
            pred = model(**model_input)
            decoded = model.decode(pred)

            for k, v in self._decode_fields.items():
                if k in decoded:
                    decoded_instance[k].extend(decoded[k][v])

        predictions = {}
        predictions["doc_key"] = doc_key
        predictions["sentences"] = [x["metadata"]["sentence"] for x in instance]
        for k, v in decoded_instance.items():
            # If we didn't train on this task, don't predict on it.
            if self._model._loss_weights[k] == 0:
                continue
            predictions[self._decode_names[k]] = self._cleanup(
                k, v, sentence_starts)

        labeled_instances = self.predictions_to_labeled_instances(instance, predictions)

        interpretations = self._interpreter.saliency_interpret_from_labeled_instances(
            labeled_instances)

        import ipdb; ipdb.set_trace()

        return predictions

    def predictions_to_labeled_instances(self, instances, predictions):
        """
        Convert predictions to labeled instances.
        """
        # Need to subtract off the sentence starts.
        sentence_lengths = [len(entry["metadata"]["sentence"]) for entry in instances]
        sentence_starts = np.cumsum(sentence_lengths)
        sentence_starts = np.roll(sentence_starts, 1)
        sentence_starts[0] = 0

        # The result.
        result_instances = []
        assert len(instances) == len(predictions["predicted_relations"])
        zipped = zip(instances, predictions["predicted_relations"], sentence_starts)
        for instance, relations, sentence_start in zipped:
            # The span field.
            span_field = instance['spans']

            for relation in relations:
                start1, end1, start2, end2 = relation[:4] - sentence_start
                relation_label = relation[4]

                found = {"span1": False, "span2": False}
                span_ix1 = span_ix2 = None
                for i, span in enumerate(span_field):
                    if start1 == span.span_start and end1 == span.span_end:
                        span_ix1 = i
                        found["span1"] = True
                    if start2 == span.span_start and end2 == span.span_end:
                        span_ix2 = i
                        found["span2"] = True
                if not (found["span1"] and found["span2"]):
                    raise Exception("Couldn't find predicted spans.")

                relation_indices = [(span_ix1, span_ix2)]
                relation_label_field = AdjacencyField(
                    indices=relation_indices,
                    sequence_field=span_field,
                    labels=[relation_label],
                    label_namespace="relation_labels")

                new_instance = deepcopy(instance)
                new_instance.add_field("relation_labels", relation_label_field, self._model.vocab)

                result_instances.append(new_instance)

        return result_instances

    @staticmethod
    def _check_lengths(d):
        keys = list(d.keys())
        # Dict fields that won't have the same length as the # of sentences in the doc.
        keys_to_remove = ["doc_key", "clusters", "predicted_clusters"]
        for key in keys_to_remove:
            if key in keys:
                keys.remove(key)
        lengths = [len(d[k]) for k in keys]
        assert len(set(lengths)) == 1

    def _cleanup(self, k, decoded, sentence_starts):
        dispatch = {"coref": self._cleanup_coref,
                    "ner": self._cleanup_ner,
                    "relation": self._cleanup_relation,
                    "events": self._cleanup_event}  # TODO(dwadden) make this nicer later if worth it.
        return dispatch[k](decoded, sentence_starts)

    @staticmethod
    def _cleanup_coref(decoded, sentence_starts):
        "Convert from nested list of tuples to nested list of lists."
        # The coref code assumes batch sizes other than 1. We only have 1.
        assert len(decoded) == 1
        decoded = decoded[0]
        res = []
        for cluster in decoded:
            cleaned = [list(x) for x in cluster]  # Convert from tuple to list.
            res.append(cleaned)
        return res

    @staticmethod
    def _cleanup_ner(decoded, sentence_starts):
        assert len(decoded) == len(sentence_starts)
        res = []
        for sentence, sentence_start in zip(decoded, sentence_starts):
            res_sentence = []
            for tag in sentence:
                new_tag = [tag[0] + sentence_start, tag[1] + sentence_start, tag[2]]
                res_sentence.append(new_tag)
            res.append(res_sentence)
        return res

    @staticmethod
    def _cleanup_relation(decoded, sentence_starts):
        "Add sentence offsets to relation results."
        def fmt_score(x):
            return round(float(x), 4)

        assert len(decoded) == len(sentence_starts)  # Length check.
        res = []
        for sentence, sentence_start in zip(decoded, sentence_starts):
            res_sentence = []
            for rel in sentence:
                # Output the spans, the label, and the scores.
                cleaned = ([x + sentence_start for x in rel[:4]] +
                           [rel[4], fmt_score(rel[5]), fmt_score(rel[6])])
                res_sentence.append(cleaned)
            res.append(res_sentence)
        return res

    @staticmethod
    def _cleanup_event(decoded, sentence_starts):
        assert len(decoded) == len(sentence_starts)  # Length check.
        res = []
        for sentence, sentence_start in zip(decoded, sentence_starts):
            trigger_dict = sentence["trigger_dict"]
            argument_dict = sentence["argument_dict_with_scores"]
            this_sentence = []
            for trigger_ix, trigger_label in trigger_dict.items():
                this_event = []
                this_event.append([trigger_ix + sentence_start, trigger_label])
                event_arguments = {k: v for k, v in argument_dict.items() if k[0] == trigger_ix}
                this_event_args = []
                for k, v in event_arguments.items():
                    entry = [x + sentence_start for x in k[1]] + list(v)
                    this_event_args.append(entry)
                this_event_args = sorted(this_event_args, key=lambda entry: entry[0])
                this_event.extend(this_event_args)
                this_sentence.append(this_event)
            res.append(this_sentence)

        return res

    @overrides
    def get_gradients(self, instances):
        """
        Gets the gradients of the loss with respect to the model inputs.

        Parameters
        ----------
        instances: List[Instance]

        Returns
        -------
        Tuple[Dict[str, Any], Dict[str, Any]]
        The first item is a Dict of gradient entries for each input.
        The keys have the form  ``{grad_input_1: ..., grad_input_2: ... }``
        up to the number of inputs given. The second item is the model's output.

        Notes
        -----
        Takes a ``JsonDict`` representing the inputs of the model and converts
        them to :class:`~allennlp.data.instance.Instance`s, sends these through
        the model :func:`forward` function after registering hooks on the embedding
        layer of the model. Calls :func:`backward` on the loss and then removes the
        hooks.
        """
        embedding_gradients = []
        hooks = self._register_embedding_gradient_hooks(embedding_gradients)

        dataset = Batch(instances)
        dataset.index_instances(self._model.vocab)

        # NOTE(dwadden) the problem happens here. For some reason, the data doesn't end up on the correct device.

        outputs = self._model.decode(self._model.forward(**dataset.as_tensor_dict()))

        import ipdb; ipdb.set_trace()

        loss = outputs['loss']
        self._model.zero_grad()
        loss.backward()

        for hook in hooks:
            hook.remove()

        grad_dict = dict()
        for idx, grad in enumerate(embedding_gradients):
            key = 'grad_input_' + str(idx + 1)
            grad_dict[key] = grad.detach().cpu().numpy()

        return grad_dict, outputs
