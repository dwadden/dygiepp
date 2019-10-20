"""
Take the by-document genia data, and split into folds to match the SUTD folds.
"""


import pandas as pd
import os
from os import path

base_dir = "./data/genia/raw-data"

train_path = f"{base_dir}/sutd-original/train.data"
dev_path = f"{base_dir}/sutd-original/dev.data"
test_path = f"{base_dir}/sutd-original/test.data"

with open(dev_path, "r") as f:
    first_dev = f.readline()

with open(test_path, "r") as f:
    first_test = f.readline()


wkdir = f"{base_dir}/sutd-article/correct-format"
order_file = f"{base_dir}/sutd-article/doc_order.csv"
order = pd.read_csv(order_file, header=None, sep="\t")[0]

out_dir = f"{base_dir}/sutd-article/split"
os.mkdir(out_dir)
for fold in ["train", "dev", "test"]:
    os.mkdir(path.join(out_dir, fold))

fold = "train"

articles_fold = dict(train=[], dev=[], test=[])

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
    fold_file = path.join(f"{base_dir}/sutd-article/split", fold + "_order.csv")
    to_write = pd.Series(articles_fold[fold])
    to_write.to_csv(fold_file, index=False, header=False)
