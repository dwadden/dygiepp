"""
Take the by-document genia data, and split into folds to match the SUTD folds.
"""


import pandas as pd
import os
from os import path

train_path = "/data/dave/proj/scierc_coref/data/genia/sutd/train.data"
dev_path = "/data/dave/proj/scierc_coref/data/genia/sutd/dev.data"
test_path = "/data/dave/proj/scierc_coref/data/genia/sutd/test.data"

with open(dev_path, "r") as f:
    first_dev = f.readline()

with open(test_path, "r") as f:
    first_test = f.readline()


wkdir = "/data/dave/proj/scierc_coref/data/genia/sutd-article/final"
order_file = "/data/dave/proj/scierc_coref/data/genia/sutd-article/doc_order.csv"
order = pd.read_table(order_file, header=None)[0]

out_dir = "/data/dave/proj/scierc_coref/data/genia/sutd-article/split"
for fold in ["train", "dev", "test"]:
    os.mkdir(path.join(out_dir, fold))

fold = "train"

articles_fold=dict(train=[], dev=[], test=[])

for article_id in order:
    with open(path.join(wkdir, "{0}.data".format(article_id))) as article:
        out_file = open(path.join(out_dir, fold, str(article_id) + ".data"), "w")
        articles_fold[fold].append(article_id)
        for line in article:
            if line == first_dev:
                fold = "dev"
                articles_fold[fold].append(article_id)
                out_file.close()
                out_file = open(path.join(out_dir, fold, str(article_id) + ".data"), "w")
            if line == first_test:
                fold = "test"
                articles_fold[fold].append(article_id)
                out_file.close()
                out_file = open(path.join(out_dir, fold, str(article_id) + ".data"), "w")
            out_file.write(line)

for fold in ["train", "dev", "test"]:
    fold_file = path.join("/data/dave/proj/scierc_coref/data/genia/sutd-article/split", fold + "_order.csv")
    to_write = pd.Series(articles_fold[fold])
    to_write.to_csv(fold_file, index=False)
