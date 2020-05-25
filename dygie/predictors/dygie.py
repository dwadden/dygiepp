from typing import List
import numpy as np

from overrides import overrides
import numpy
import json

from allennlp.common.util import JsonDict
from allennlp.nn import util
from allennlp.data.dataset import Batch
from allennlp.data import DatasetReader
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor


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

        # Include the gold fields corresponding to the predicted fields.
        for field in ["ner", "relation", "events"]:
            assert field not in predictions
            predicted_field = f"predicted_{field}"
            if f"predicted_{field}" in predictions:
                values = [inst["metadata"][field] for inst in instance]
                # If we have data, store it.
                if any([x != [] for x in values]):
                    assert len(values) == len(predictions[predicted_field])
                    predictions[field] = values

        return predictions

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
        assert len(decoded) == len(sentence_starts)  # Length check.
        res = []
        for sentence, sentence_start in zip(decoded, sentence_starts):
            res_sentence = []
            for rel in sentence:
                cleaned = [x + sentence_start for x in rel[:4]] + [rel[4]]
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
