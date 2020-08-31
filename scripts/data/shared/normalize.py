import argparse
import json
import os

from dygie.data.dataset_readers.document import Document, Dataset


def load_jsonl(fname):
    return [json.loads(x) for x in open(fname)]


def save_jsonl(xs, fname):
    with open(fname, "w") as f:
        for x in xs:
            print(json.dumps(x), file=f)


def get_args():
    parser = argparse.ArgumentParser(
        description="Normalize a dataset by adding a `dataset` field and splitting long documents.")
    parser.add_argument("input_directory", type=str,
                        help="Directory with train, dev, and test files.")
    parser.add_argument("output_directory", type=str,
                        help="Directory where the output files will go.")
    parser.add_argument("--file_extension", type=str, default="jsonl",
                        help="File extension for data files.")
    parser.add_argument("--train_name", type=str, default="train",
                        help="Name of the file with the training split.")
    parser.add_argument("--dev_name", type=str, default="dev",
                        help="Name of the file with the dev split. For instance, `validation`.")
    parser.add_argument("--test_name", type=str, default="test",
                        help="Name of the file with the test split.")
    parser.add_argument("--max_tokens_per_doc", type=int, default=500,
                        help="Maximum tokens per document. Longer ones will be split. If set to 0, do not split documents.")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name.")
    return parser.parse_args()


class Normalizer:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def normalize(self):
        os.makedirs(self.output_directory, exist_ok=True)
        fold_names = [self.train_name, self.dev_name, self.test_name]
        for fold in fold_names:
            self.process_fold(fold)

    def process_fold(self, fold):
        fname = f"{self.input_directory}/{fold}.{self.file_extension}"
        dataset = Dataset.from_jsonl(fname)
        res = []

        for doc in dataset:
            res.extend(self.process_entry(doc))

        out_name = f"{self.output_directory}/{fold}.{self.file_extension}"
        save_jsonl(res, out_name)

    def process_entry(self, doc):
        doc.dataset = self.dataset
        if self.max_tokens_per_doc > 0:
            splits = doc.split(self.max_tokens_per_doc)
        else:
            splits = [doc]

        return [split.to_json() for split in splits]


def main():
    args = get_args()
    normalizer = Normalizer(**vars(args))
    normalizer.normalize()


if __name__ == "__main__":
    main()
