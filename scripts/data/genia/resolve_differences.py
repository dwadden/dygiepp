"""
My article version of GENIA didn't quite line up with the SUTD version. To fix
the alignment, go through my versions and their versions. Check that they're
close, then write their version to article.
"""

from os import path
import os
import pandas as pd
from shutil import copyfile
from Levenshtein.StringMatcher import StringMatcher


base_dir = "./data/genia/raw-data"
article_dir = f"{base_dir}/sutd-article/split"
sutd_dir = f"{base_dir}/sutd-original"
out_dir = f"{base_dir}/sutd-article/split-corrected"


def make_comparator():
    matcher = StringMatcher()

    def compare(str1, str2):
        matcher.set_seqs(str1, str2)
        return matcher.distance()
    return compare


def make_sentences(lines):
    sentences = []
    sentence = []
    for i, line in enumerate(lines):
        if i and not i % 4:
            sentences.append(sentence)
            sentence = []
        sentence.append(line)
    sentences.append(sentence)    # Get the last one.
    return sentences


def get_matching_sentences(sentences_article, sutd):
    num_sentences = len(sentences_article)
    sents_sutd = []
    for _ in range(4 * num_sentences):
        sents_sutd.append(sutd.readline())
    return make_sentences(sents_sutd)


def fix_fold(fold):
    comparator = make_comparator()
    sutd = open(path.join(sutd_dir, "{0}.data".format(fold)), "r")
    doc_order_file = path.join(article_dir, "{0}_order.csv".format(fold))
    doc_order = pd.read_csv(doc_order_file, header=None, sep="\t")[0]

    if not path.exists(path.join(out_dir, fold)):
        os.mkdir(path.join(out_dir, fold))

    doc_order_file_to_copy = path.join(out_dir, "{0}_order.csv".format(fold))
    copyfile(doc_order_file, doc_order_file_to_copy)

    for doc in doc_order:
        with open(path.join(article_dir, fold, "{0}.data".format(doc)), "r") as article:
            with open(path.join(out_dir, fold, "{0}.data".format(doc)), "w") as f_out:
                sents_article = make_sentences(article.readlines())
                sents_sutd = get_matching_sentences(sents_article, sutd)
                if len(sents_article) != len(sents_sutd):
                    raise Exception("Wrong length.")
                dists = [comparator(sent_article[0], sent_sutd[0])
                         for sent_article, sent_sutd in zip(sents_article, sents_sutd)]
                if max(dists) > 15:
                    raise Exception("There's a problem with {0}.".format(doc))
                for sent in sents_sutd:
                    for line in sent:
                        f_out.write(line)


def main():
    if not path.exists(out_dir):
        os.mkdir(out_dir)

    for fold in ["train", "dev", "test"]:
        fix_fold(fold)


if __name__ == '__main__':
    main()
