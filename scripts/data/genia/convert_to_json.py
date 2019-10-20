"""
Format GENIA data set from
https://gitlab.com/sutd_nlp/overlapping_mentions/tree/master/data/GENIA into the json form used by
DyGIE++.
"""

import json
import os
from os import path
import pandas as pd


def save_list(xs, name):
    "Save a list as text, one entry per line."
    with open(name, "w") as f:
        for x in xs:
            f.write(str(x) + "\n")


def make_sentences(lines):
    sentences = []
    sentence = []
    for i, line in enumerate(lines):
        if i and not i % 4:
            sentences.append(sentence)
            sentence = []
        sentence.append(line.strip())
    sentences.append(sentence)    # Get the last one.
    return sentences


def format_tag(tag, offset):
    ixs, name = tag.split(" ")
    # TODO(dwadden) this is a hack. Shouldn't be happening, but uncommon enough
    # that not worth fixing.
    if not ixs:
        return None
    start_ix, end_ix = ixs.split(",")
    start_ix = int(start_ix) + offset
    end_ix = int(end_ix) + offset - 1      # Our endpoints are inclusive, not exclusive.
    name = name.replace("G#", "")
    return [start_ix, end_ix, name]


def no_tags(line):
    return len(line) == 1 and line[0] == ''


def process_ner(line, offset):
    # If not NER tags, return an empty list.
    if not line:
        return []
    else:
        tags = line.split("|")
        formatted = [format_tag(tag, offset) for tag in tags]
        # TODO(dwadden) this continues the hack from above.
        return [entry for entry in formatted if entry is not None]


def sentence_to_json(sent, offset):
    """Get the tokens and NER tags. Ignore the POS tags."""
    # No doc keys in the statnlp paper. Start by ignoring.
    tokens = sent[0].split(" ")
    ner_tags = process_ner(sent[2], offset)
    res = tokens, ner_tags
    return res


def doc_to_json(sents, doc_id, fold):
    """A list of sentences (a document) to json."""
    # Append fold info to doc_key since one doc appears in both train and dev;
    # ditto dev and test.
    res = dict(clusters=[],
               sentences=[],
               ner=[],
               relations=[],
               doc_key=str(doc_id) + '_' + fold)
    offset = 0
    for sent in sents:
        tokens_sent, ner_tags_sent = sentence_to_json(sent, offset)
        res["clusters"].append([])
        res["sentences"].append(tokens_sent)
        res["ner"].append(ner_tags_sent)
        res["relations"].append([])
        # Start the next set of indices from this offset.
        offset += len(tokens_sent)
    return res


def get_unique_ner_labels(jsonified):
    "Get unique NER labels."
    labels = set()
    for sentence in jsonified["ner"]:
        for entry in sentence:
            labels.add(entry[2])
    return labels


def format_fold(fold, in_dir, out_dir):
    """Take data SUTD-formatted documents and convert to our JSON format."""
    out_name = path.join(out_dir, "{0}.json".format(fold))
    ner_labels = set()
    with open(out_name, "w") as f_out:
        order = pd.read_csv(path.join(in_dir, "{0}_order.csv".format(fold)), header=None, sep="\t")[0]
        already_written = set()
        for doc_id in order:
            # I need this check because there is a duplicate document in the train
            # set, which makes elmo barf. This isn't a problem with my code, it's
            # upstream somewhere.
            if doc_id in already_written:
                continue
            already_written.add(doc_id)
            with open(path.join(in_dir, fold, "{0}.data".format(doc_id)), "r") as f_in:
                lines = f_in.readlines()
                sents = make_sentences(lines)
                jsonified = doc_to_json(sents, doc_id, fold)
                ner_labels_article = get_unique_ner_labels(jsonified)
                ner_labels = ner_labels | ner_labels_article
                f_out.write(json.dumps(jsonified) + "\n")
    return ner_labels


def main():
    in_prefix = "./data/genia/raw-data/sutd-article"
    in_dir = f"{in_prefix}/split-corrected"
    out_dir = "./data/genia/processed-data/json-ner"
    os.makedirs(out_dir)
    folds = ["train", "dev", "test"]
    ner_labels = set()

    for fold in folds:
        msg = "Formatting fold {0}.".format(fold)
        print(msg)
        ner_labels_fold = format_fold(fold, in_dir, out_dir)
        ner_labels = ner_labels | ner_labels_fold

    save_list(sorted(ner_labels), path.join(in_prefix, "ner-labels.txt"))


if __name__ == "__main__":
    main()
