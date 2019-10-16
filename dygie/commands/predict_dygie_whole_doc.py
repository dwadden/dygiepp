#! /usr/bin/env python

"""
Make predictions of trained model, output as json like input. Not easy to do this in the current
AllenNLP predictor framework, so here's a short script to do it.

usage: predict_dygie_whole_doc.py [archive-file] [test-file] [output-file]

This is a variant of `predict_dygie` that was used for the AI2 hackathon to make full-document
predictions.
"""

# TODO(dwadden) This breaks right now on relation prediction because json can't do dicts whose keys
# are tuples.

import json
from sys import argv
import warnings
from os import path

import numpy as np

from allennlp.models.archival import load_archive
from allennlp.common.util import import_submodules
from allennlp.data import DatasetReader
from allennlp.data.dataset import Batch
from allennlp.nn import util as nn_util

from dygie.data.iterators.document_iterator import DocumentIterator
from dygie.data.iterators.batch_iterator import BatchIterator


decode_fields = dict(coref="clusters",
                     ner="decoded_ner",
                     relation="decoded_relations")

decode_names = dict(coref="predicted_clusters",
                    ner="predicted_ner",
                    relation="predicted_relations")


def cleanup(k, decoded, sentence_starts):
    dispatch = {"coref": cleanup_coref,
                "ner": cleanup_ner,
                "relation": cleanup_relation}
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


def cleanup_event(decoded, sentence_starts):
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


def load_json(test_file):
    res = []
    with open(test_file, "r") as f:
        for line in f:
            res.append(json.loads(line))

    return res


def check_lengths(d):
    "Make sure all entries in dict have same length."
    keys = list(d.keys())
    for key in ["doc_key", "clusters", "predicted_clusters", "section_starts"]:
        if key in keys:
            keys.remove(key)
    lengths = [len(d[k]) for k in keys]
    assert len(set(lengths)) == 1


def predict(model, dataset_reader, test_file, output_file, cuda_device):
    gold_test_data = load_json(test_file)
    instances = dataset_reader.read(test_file)
    batch = Batch(instances)
    batch.index_instances(model.vocab)
    iterator = BatchIterator()
    iterator._batch_size = 5
    # For long documents, loop over batches of sentences. Keep track of the
    # total length and append onto the end of the predictions for each sentence
    # batch.
    assert len(gold_test_data) == 1
    gold_data = gold_test_data[0]
    predictions = {}
    total_length = 0
    for sents in iterator(batch.instances, num_epochs=1, shuffle=False):
        sents = nn_util.move_to_device(sents, cuda_device)  # Put on GPU.
        sentence_lengths = [len(entry["sentence"]) for entry in sents["metadata"]]
        sentence_starts = np.cumsum(sentence_lengths) + total_length
        sentence_starts = np.roll(sentence_starts, 1)
        sentence_starts[0] = total_length
        pred = model(**sents)
        decoded = model.decode(pred)
        if total_length == 0:
            for k, v in decoded.items():
                predictions[decode_names[k]] = cleanup(k, v[decode_fields[k]], sentence_starts)
        else:
            for k, v in decoded.items():
                predictions[decode_names[k]] += cleanup(k, v[decode_fields[k]], sentence_starts)
        total_length += sum(sentence_lengths)

    res = {}
    res.update(gold_data)
    res.update(predictions)
    check_lengths(res)
    encoded = json.dumps(res, default=int)
    with open(output_file, "w") as f:
        f.write(encoded + "\n")


def predict_list(archive_file, test_file_dir, test_file_list, output_dir,
                 cuda_device, log_file):
    import_submodules("dygie")
    archive = load_archive(archive_file, cuda_device)
    model = archive.model
    model.eval()
    config = archive.config.duplicate()
    dataset_reader_params = config["dataset_reader"]
    dataset_reader = DatasetReader.from_params(dataset_reader_params)

    test_files = []
    with open(test_file_list) as f:
        for line in f:
            test_files.append(line.strip())

    for name in test_files:
        test_file = path.join(test_file_dir, name)
        output_file = path.join(output_dir, name)
        try:
            predict(model, dataset_reader, test_file, output_file, cuda_device)
        except Exception:
            with open(log_file, "a") as log:
                print(f"{name}", file=log)



def main():
    archive_file = argv[1]
    test_file_dir = argv[2]
    test_file_list = argv[3]
    output_dir = argv[4]
    cuda_device = int(argv[5])
    log_file = argv[6]
    predict_list(archive_file, test_file_dir, test_file_list, output_dir,
                 cuda_device, log_file)


if __name__ == '__main__':
    main()
