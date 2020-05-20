from typing import List
import numpy as np
from copy import deepcopy
import torch

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

from dygie.interpret.dygie import DyGIEInterpreter


class DyGIEPredictionException(Exception):
    pass


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
        self._interpreter = DyGIEInterpreter(self)

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

        # Keep track of BERT offsets for gradient attribution.
        # TODO(dwadden) When we switch to AllenNLP 1.0 this shouldn't be necessary. Apparently the
        # solution is in this PR: https://github.com/allenai/allennlp/pull/4179/files.
        # Also relevant stuff here:
        # https://github.com/allenai/allennlp/blob/master/allennlp/modules/token_embedders/pretrained_transformer_mismatched_embedder.py
        bert_offsets_all = []

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
            bert_offsets_all.append(pred["text"]["bert-offsets"].detach().cpu().numpy()[0])
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
        if len(labeled_instances) != len(bert_offsets_all):
            raise DyGIEPredictionException("There's confusion about the number of sentences.")

        # Loop over the sentences, getting interpretations for each relation.
        all_interpretations = []
        for labeled_sentence, bert_offsets in zip(labeled_instances, bert_offsets_all):
            interpretations = self._interpreter.saliency_interpret_from_labeled_instances(
                labeled_sentence)
            interpretations = self._aggregate_for_bert_offsets(interpretations, bert_offsets)
            all_interpretations.append(interpretations)

        zipped = zip(predictions["sentences"], predictions["predicted_relations"], all_interpretations)
        for sentence, predicted_relations, interpretations in zipped:
            if len(predicted_relations) != len(interpretations):
                raise DyGIEPredictionException("Length mistmatch pase")
            for predicted_relation, interpretation in zip(predicted_relations, interpretations):
                rounded = [round(x, 4) for x in interpretation]
                predicted_relation.append(rounded)

        return predictions

    @staticmethod
    def _aggregate_for_bert_offsets(interpretations, bert_offsets):
        """
        For gradient-based attribution models, there's on gradient per BERT wordpiece. We need to
        convert this to gradients per token by summing over the wordpieces. The `bert_offsets`
        give the mapping from tokens to wordpieces.
        """
        def aggregate_one(scores):
            "Aggregate scores for a single input."
            # Note: The entry in `scores` is on the CLS token, and the last is on the SEP token.
            if max(bert_offsets) > len(scores) - 2:
                raise DyGIEPredictionException("Something's weird with the offsets.")

            aggregated_scores = []
            offset_end = len(scores) - 1
            offsets_with_end = bert_offsets.tolist() + [offset_end]
            for i in range(len(offsets_with_end) - 1):
                start = offsets_with_end[i]
                end = offsets_with_end[i + 1]
                token_scores = scores[start:end]
                this_score = np.mean(token_scores)
                aggregated_scores.append(this_score)

            if len(aggregated_scores) != len(bert_offsets):
                raise DyGIEPredictionException("Length mismatch in aggregate scores.")

            return aggregated_scores

        # Make sure the keys are in sorted order.
        aggregated = []
        keys = sorted(interpretations.keys(), key=lambda x: int(x.split("_")[1]))
        for key in keys:
            interp = interpretations[key]
            if set(interp.keys()) != set(["grad_input_1"]):
                raise DyGIEPredictionException("Unexpected outputs.")
            interp = interp["grad_input_1"]
            to_append = aggregate_one(interp)
            aggregated.append(to_append)

        return aggregated


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
            # Accumulate the results for a single instance.
            results_inst = []
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

                results_inst.append(new_instance)

            # Append the results for this instance to the full list of results.
            result_instances.append(results_inst)

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
        "Get the gradients with respect to a single predicted relation."
        embedding_gradients = []
        hooks = self._register_embedding_gradient_hooks(embedding_gradients)

        dataset = Batch(instances)
        dataset.index_instances(self._model.vocab)

        # NOTE(dwadden) In v1.0, the predictor has a `device` attribute. This isn't available yet,
        # so I'll hack it.
        device = next(self._model._text_field_embedder.parameters()).device.index
        dataset_tensor_dict = util.move_to_device(dataset.as_tensor_dict(), device)
        outputs = self._model.decode(self._model.forward(**dataset_tensor_dict))

        relation_outputs = outputs["relation"]
        relation_scores = relation_outputs["relation_scores"]
        if relation_scores.size(0) != 1:
            raise DyGIEPredictionException("Only expected a single instance.")
        relation_scores = relation_scores[0]

        # If the gold entities are in the beam, then this pair contributes to the loss. Otherwise,
        # we don't get any information.
        gold_indices = self._get_index_in_relation_scores(instances, relation_outputs)
        if gold_indices is None:
            loss = torch.tensor(0, dtype=torch.float, requires_grad=True)
        else:
            gold_label = instances[0]["relation_labels"].labels[0]
            # Need to add one because of null class.
            gold_number = self._model.vocab.get_token_index(gold_label, "relation_labels") + 1
            # This is the gold label number.
            gold_number = (torch.tensor([gold_number], requires_grad=False, dtype=int).
                           to(relation_scores.device))
            # Grab the scores that are relevant to this number.
            relevant_scores = relation_scores[gold_indices[0], gold_indices[1]].unsqueeze(0)
            # Compute the cross-entropy loss.
            loss = self._model._relation._loss(relevant_scores, gold_number)

        self._model.zero_grad()
        loss.backward()

        for hook in hooks:
            hook.remove()

        grad_dict = dict()
        for idx, grad in enumerate(embedding_gradients):
            key = 'grad_input_' + str(idx + 1)
            grad_dict[key] = grad.detach().cpu().numpy()

        return grad_dict, outputs

    def _get_index_in_relation_scores(self, instances, relation_outputs):
        """
        Get the indices of the predicted span pair in the marix of relation scores. They may not
        always be there (since we're sweeping the parameters during integrated gradients). When
        either one is not there, return None.
        """
        if len(instances) != 1:
            raise DyGIEPredictionException("Expected a single instance.")

        gold_indices = instances[0]["relation_labels"].indices[0]
        top_span_indices = relation_outputs["top_span_indices"].detach().cpu()[0]
        ixs = {0: torch.where(gold_indices[0] == top_span_indices),
               1: torch.where(gold_indices[1] == top_span_indices)}

        for k in ixs:
            ix = ixs[k]
            if len(ix) != 1:
                raise DyGIEPredictionException("Expected a tuple with one item.")
            ix = ix[0]
            if len(ix) == 0:
                # Didn't find it. Return None.
                return None
            if len(ix) > 1:
                raise DyGIEPredictionException("There shouldn't be multiple occurrences.")
            ix = ix.item()
            ixs[k] = ix

        return (ixs[0], ixs[1])
