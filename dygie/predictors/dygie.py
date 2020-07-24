from typing import List
import numpy as np

from overrides import overrides
import numpy
import json

from allennlp.common.util import JsonDict
from allennlp.nn import util
from allennlp.data import Batch
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

        dataset = Batch([instance])
        dataset.index_instances(model.vocab)
        model_input = util.move_to_device(dataset.as_tensor_dict(), cuda_device)
        prediction = model.make_output_human_readable(model(**model_input))

        return prediction.to_json()

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
