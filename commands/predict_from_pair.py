#! /usr/bin/env python

from sys import argv
import os
from os import path
import itertools
import numpy as np
import json

import torch

from allennlp.data import Vocabulary

from dygie.models.shared import fields_to_batches
from dygie.commands import predict_dygie as pdy


def decode_trigger(output, vocab):
    trigger_dict = {}
    for i in range(output["sentence_lengths"]):
        trig_label = output["predicted_triggers"][i].item()
        if trig_label > 0:
            trigger_dict[i] = vocab.get_token_from_index(trig_label, namespace="trigger_labels")

    return trigger_dict


def decode_arguments(output, decoded_trig, vocab):
    argument_dict = {}
    argument_dict_with_scores = {}
    the_highest, _ = output["argument_scores"].max(dim=0)
    for i, j in itertools.product(range(output["num_triggers_kept"]),
                                  range(output["num_argument_spans_kept"])):
        trig_ix = output["top_trigger_indices"][i].item()
        arg_span = tuple(output["top_argument_spans"][j].tolist())
        arg_label = output["predicted_arguments"][i, j].item()
        # Only include the argument if its putative trigger is predicted as a real trigger.
        if arg_label >= 0 and trig_ix in decoded_trig:
            arg_score = output["argument_scores"][i, j, arg_label + 1].item()
            label_name = vocab.get_token_from_index(arg_label, namespace="argument_labels")
            argument_dict[(trig_ix, arg_span)] = label_name
            # Keep around a version with the predicted labels and their scores, for debugging
            # purposes.
            argument_dict_with_scores[(trig_ix, arg_span)] = (label_name, arg_score)

    return argument_dict, argument_dict_with_scores


def decode(trig_dict, arg_dict, vocab):
    """
    Largely copy-pasted from what happens in dygie.
    """
    ignore = ["loss", "decoded_events"]
    trigs = fields_to_batches({k: v.detach().cpu() for k, v in trig_dict.items() if k not in ignore})
    args = fields_to_batches({k: v.detach().cpu() for k, v in arg_dict.items() if k not in ignore})

    res = []

    # Collect predictions for each sentence in minibatch.
    for trig, arg in zip(trigs, args):
        decoded_trig = decode_trigger(trig, vocab)
        decoded_args, decoded_args_with_scores = decode_arguments(arg, decoded_trig, vocab)
        entry = dict(trigger_dict=decoded_trig, argument_dict=decoded_args,
                     argument_dict_with_scores=decoded_args_with_scores)
        res.append(entry)

    return res


def get_pred_dicts(pred_dir):
    res = {}
    for name in os.listdir(pred_dir):
        doc_key = name.replace(".th", "")
        res[doc_key] = torch.load(path.join(pred_dir, name))

    return res


def predict_one(trig, arg, gold, vocab):
    sentence_lengths = [len(entry) for entry in gold["sentences"]]
    sentence_starts = np.cumsum(sentence_lengths)
    sentence_starts = np.roll(sentence_starts, 1)
    sentence_starts[0] = 0
    decoded = decode(trig["events"], arg["events"], vocab)
    cleaned = pdy.cleanup_event(decoded, sentence_starts)
    res = {}
    res.update(gold)
    res["predicted_events"] = cleaned
    pdy.check_lengths(res)
    encoded = json.dumps(res, default=int)
    return encoded


def get_gold_data(test_file):
    data = pdy.load_json(test_file)
    res = {x["doc_key"]: x for x in data}
    return res


def predict_from_pair(trigger_prediction_dir, arg_prediction_dir, vocab_dir, test_file, output_file):
    trig_preds = get_pred_dicts(trigger_prediction_dir)
    arg_preds = get_pred_dicts(arg_prediction_dir)
    vocab = Vocabulary.from_files(vocab_dir)
    gold = get_gold_data(test_file)
    assert set(arg_preds.keys()) == set(trig_preds.keys())
    with open(output_file, "w") as f:
        for doc in trig_preds:
            one_pred = predict_one(trig_preds[doc], arg_preds[doc], gold[doc], vocab)
            f.write(one_pred + "\n")


def main():
    trigger_prediction_dir = argv[1]
    arg_prediction_dir = argv[2]
    vocab_dir = argv[3]
    test_file = argv[4]
    output_file = argv[5]
    predict_from_pair(trigger_prediction_dir, arg_prediction_dir, vocab_dir, test_file, output_file)


if __name__ == '__main__':
    main()
