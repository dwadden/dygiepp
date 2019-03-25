#! /usr/bin/env python

"""
Make predictions of trained model, output as json like input. Not easy to do this in the current
AllenNLP predictor framework, so here's a short script to do it.

usage: predict.py [archive-file] [test-file] [output-file]
"""

# TODO(dwadden) This breaks right now on relation prediction because json can't do dicts whose keys
# are tuples.

import json
from sys import argv

import numpy as np

from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor
from allennlp.common.util import import_submodules
from allennlp.data import DatasetReader
from allennlp.data.dataset import Batch

from dygie.data.iterators.document_iterator import DocumentIterator


decode_fields = dict(coref="clusters",
                     ner="decoded_ner",
                     relation="decoded_relations",
                     events="decoded_events")

decode_names = dict(coref="clusters",
                    ner="ner",
                    relation="relations",
                    events="events")


def cleanup(k, decoded, sentence_starts):
    dispatch = {"coref": cleanup_coref,
                "ner": cleanup_ner,
                "relation": cleanup_relation,
                "events": lambda x, y: x}  # TODO(dwadden) make this nicer later if worth it.
    return dispatch[k](decoded, sentence_starts)


def cleanup_coref(decoded, sentence_starts):
    "Convert from nested list of tuples to nested list of lists."
    # The coref code assumes batch sizes other than 1. We only have 1.
    assert len(decoded) == 1
    decoded = decoded[0]
    res = []
    for cluster in decoded:
        cleaned = [list(x) for x in cluster]  # Convert from tuple to list.
        res.append(cleaned)
    return res


def cleanup_ner(decoded, sentence_starts):
    assert len(decoded) == len(sentence_starts)
    res = []
    for sentence, sentence_start in zip(decoded, sentence_starts):
        res_sentence = []
        for tag in sentence:
            new_tag = [tag[0] + sentence_start, tag[1] + sentence_start, tag[2]]
            res_sentence.append(new_tag)
        res.append(res_sentence)
    return res


def cleanup_relation(decoded, sentence_starts):
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


def predict(archive_file, test_file, output_file):
    import_submodules("dygie")
    archive = load_archive(archive_file)
    model = archive.model
    model.eval()
    config = archive.config.duplicate()
    dataset_reader_params = config["dataset_reader"]
    dataset_reader = DatasetReader.from_params(dataset_reader_params)
    instances = dataset_reader.read(test_file)
    batch = Batch(instances)
    batch.index_instances(model.vocab)
    iterator = DocumentIterator()
    with open(output_file, "w") as f:
        for doc in iterator(batch.instances, num_epochs=1):
            sentence_lengths = [len(entry["sentence"]) for entry in doc["metadata"]]
            sentence_starts = np.cumsum(sentence_lengths)
            sentence_starts = np.roll(sentence_starts, 1)
            sentence_starts[0] = 0
            pred = model(**doc)
            decoded = model.decode(pred)
            res = {}
            for k, v in decoded.items():
                res[decode_names[k]] = cleanup(k, v[decode_fields[k]], sentence_starts)
            res["sentences"] = [entry["sentence"] for entry in doc["metadata"]]
            res["doc_key"] = doc["metadata"][0]["doc_key"]
            encoded = json.dumps(res, default=int)
            f.write(encoded + "\n")


def main():
    archive_file = argv[1]
    test_file = argv[2]
    output_file = argv[3]
    predict(archive_file, test_file, output_file)


if __name__ == '__main__':
    main()
