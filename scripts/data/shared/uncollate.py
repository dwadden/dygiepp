import argparse
import os
from collections import defaultdict

from dygie.data.dataset_readers import document

####################

# Inverse of `collate.py`. Takes collated data and arranges it back into its original documents.


class UnCollator:
    def __init__(self, corpus, order_like=None):
        self.corpus = corpus
        self.dataset = self._get_dataset(corpus)
        self.order = self._get_order(order_like)
        self.weight = self._get_weight(corpus)

    def _get_order(self, order_like):
        if order_like is None:
            return None
        else:
            orig_doc_keys = set()
            for doc in self.corpus:
                for sent in doc:
                    orig_doc_keys.add(sent.metadata["_orig_doc_key"])

            # Make sure the keys match.
            orig_order = [x.doc_key for x in order_like]
            if set(orig_doc_keys) != set(orig_order):
                raise ValueError("Doc keys don't match between corpus to decollate and corpus to use for ordering")

            return orig_order

    def _get_weight(self, corpus):
        """
        Get document weight. Right now, can only handle corpora where all documents have same
        weight.
        """
        weights = set([x.weight for x in self.corpus])
        if len(weights) > 1:
            raise ValueError("Cannot uncollate documents with different instance weights.")
        return sorted(weights)[0]

    @staticmethod
    def _get_dataset(corpus):
        datasets = [x.dataset for x in corpus]
        if len(set(datasets)) > 1:
            raise ValueError("Can only uncollate documents with the same `dataset` field.")

        return datasets[0]

    def uncollate(self):
        # Collect all the sentences for each document.
        doc_dict = defaultdict(list)
        for doc in self.corpus:
            for sent in doc:
                doc_key = sent.metadata["_orig_doc_key"]
                doc_dict[doc_key].append(sent)

        # Re-assemble the dataset.
        docs = []
        order = self.order if self.order is not None else sorted(doc_dict)
        for doc_key in order:
            doc = self._uncollate_doc(doc_key, doc_dict[doc_key])
            docs.append(doc)

        return document.Dataset(docs)

    def _uncollate_doc(self, doc_key, sents):
        # Uncollate the sentences in a single document
        sents = sorted(sents, key=lambda x: x.metadata["_orig_sent_ix"])
        if [x.metadata["_orig_sent_ix"] for x in sents] != [x for x in range(len(sents))]:
            raise Exception(f"Some sentences for {doc_key} are missing.")

        sentences = []
        sentence_ix = 0
        sentence_start = 0

        for sent in sents:
            sent.sentence_ix = sentence_ix
            sent.sentence_start = sentence_start
            # Remove unnecessary metadata fields.
            for field in ["_orig_sent_ix", "_orig_doc_key"]:
                del sent.metadata[field]
            sentences.append(sent)
            sentence_ix += 1
            sentence_start += len(sent)

        new_doc = document.Document(doc_key=doc_key,
                                    dataset=self.dataset,
                                    sentences=sentences,
                                    weight=self.weight)
        return new_doc


####################


def get_args(args=None):
    parser = argparse.ArgumentParser(
        description="Un-collated a previously collated a dataset.")
    parser.add_argument("input_directory", type=str,
                        help="Directory with train, dev, and test files.")
    parser.add_argument("output_directory", type=str,
                        help="Directory where the output files will go.")
    parser.add_argument("--order_like_directory", type=str, default=None,
                        help="If a directory is given, order the documents like in this directory.")
    parser.add_argument("--file_extension", type=str, default="jsonl",
                        help="File extension for data files.")
    parser.add_argument("--train_name", type=str, default="train",
                        help="Name of the file with the training split. Enter `skip` to skip this fold.")
    parser.add_argument("--dev_name", type=str, default="dev",
                        help="Name of the file with the dev split. For instance, `validation`, of `skip` to skip")
    parser.add_argument("--test_name", type=str, default="test",
                        help="Name of the file with the test split.")

    # If args are given, parse them; otherwise use command line.
    if args is not None:
        return parser.parse_args(args)
    else:
        return parser.parse_args()


class UnCollateRunner:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def run(self):
        os.makedirs(self.output_directory, exist_ok=True)
        fold_names = [self.train_name, self.dev_name, self.test_name]
        for fold in fold_names:
            if fold == "skip":
                continue
            else:
                self.process_fold(fold)

    def process_fold(self, fold):
        fname = f"{self.input_directory}/{fold}.{self.file_extension}"
        corpus = document.Dataset.from_jsonl(fname)
        if self.order_like_directory is not None:
            order_fname = f"{self.order_like_directory}/{fold}.{self.file_extension}"
            order_like = document.Dataset.from_jsonl(order_fname)
        else:
            order_like = None
        uncollator = UnCollator(
            corpus, order_like)
        res = uncollator.uncollate()
        out_name = f"{self.output_directory}/{fold}.{self.file_extension}"
        res.to_jsonl(out_name)


def main():
    args = get_args()
    runner = UnCollateRunner(**vars(args))
    runner.run()


if __name__ == "__main__":
    main()
