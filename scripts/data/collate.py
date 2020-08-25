import argparse
import os

from dygie.data.dataset_readers import document

####################

# Class to collate the documents.


class Collator:
    def __init__(self, corpus, max_spans_per_doc, max_sentences_per_doc, dataset):
        self.corpus = corpus
        self.max_spans_per_doc = max_spans_per_doc
        self.max_sentences_per_doc = max_sentences_per_doc
        self.dataset = self._get_dataset(dataset)
        self.weight = self._get_weight(corpus)
        self._remove_clusters()
        self._reset_batch()

    def _reset_batch(self):
        self.sents_batch = []
        self.sentence_ix = 0
        self.sentence_start = 0

    def collate(self):
        self._reset_batch()
        sents = self._sort_sentences()
        documents = []
        document_counter = 0

        for sent in sents:
            sent_spans = len(sent) ** 2
            # How many spans will there be if we add this sentence to the batch?
            candidate_n_spans = sent_spans * len(self.sents_batch) + 1
            # How many sentences?
            candidate_n_sents = len(self.sents_batch) + 1
            # If adding a sentence makes the document too big, start a new one.
            start_new_doc = ((candidate_n_spans > self.max_spans_per_doc) or
                             (candidate_n_sents > self.max_sentences_per_doc))
            # If it would put us over, finish this document and start a new one.
            if start_new_doc:
                new_doc = document.Document(doc_key=document_counter,
                                            dataset=self.dataset,
                                            sentences=self.sents_batch)
                documents.append(new_doc)
                document_counter += 1

                self._reset_batch()

            # Reset the index of the sentence in the document, and its starting token.
            sent.sentence_ix = self.sentence_ix
            sent.sentence_start = self.sentence_start
            self.sents_batch.append(sent)
            self.sentence_ix += 1
            self.sentence_start += len(sent)

        # At the end, get any docs that aren't left.
        new_doc = document.Document(doc_key=document_counter,
                                    dataset=self.dataset,
                                    sentences=self.sents_batch,
                                    weight=self.weight)
        documents.append(new_doc)
        self._reset_batch()

        return document.Dataset(documents)

    def _get_dataset(self, dataset):
        if dataset is not None:
            return dataset

        datasets = [x.dataset for x in self.corpus]

        if len(set(datasets)) != 1:
            raise ValueError("The documents in the corpus must be from a single dataset.")

        return datasets[0]

    def _get_weight(self, corpus):
        """
        Get document weight. Right now, can only handle corpora where all documents have same
        weight.
        """
        weights = set([x.weight for x in self.corpus])
        if len(weights) > 1:
            raise ValueError("Cannot collate documents with different instance weights.")
        return sorted(weights)[0]

    def _remove_clusters(self):
        "Can't collate data with coreference information. Remove it."
        for doc in self.corpus:
            doc.clusters = None
            doc.predicted_clusters = None
            for sent in doc:
                sent.cluster_dic = None

    def _sort_sentences(self):
        all_sents = []
        for doc in self.corpus:
            for i, sent in enumerate(doc):
                sent.metadata = {"_orig_doc_key": doc.doc_key,
                                 "_orig_sent_ix": i}
                all_sents.append(sent)

        return sorted(all_sents, key=lambda x: len(x))


####################


def get_args(args=None):
    parser = argparse.ArgumentParser(
        description="Collate a dataset. Re-organize into `documents` with sentences of similar length.")
    parser.add_argument("input_directory", type=str,
                        help="Directory with train, dev, and test files.")
    parser.add_argument("output_directory", type=str,
                        help="Directory where the output files will go.")
    parser.add_argument("--file_extension", type=str, default="jsonl",
                        help="File extension for data files.")
    parser.add_argument("--train_name", type=str, default="train",
                        help="Name of the file with the training split. To skip this fold, enter `skip`.")
    parser.add_argument("--dev_name", type=str, default="dev",
                        help="Name of the file with the dev split. For instance, `validation`. Enter `skip` to skip.")
    parser.add_argument("--test_name", type=str, default="test",
                        help="Name of the file with the test split. Enter `skip` to skip.")
    parser.add_argument("--max_spans_per_doc", type=int, default=50000,
                        help="Heuristic for max spans, as square of longest sentence length")
    parser.add_argument("--max_sentences_per_doc", type=int, default=16,
                        help="Maximum number of sentences allowed in a document.")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name.")

    # If args are given, parse them; otherwise use command line.
    if args is not None:
        return parser.parse_args(args)
    else:
        return parser.parse_args()


class CollateRunner:
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
        collator = Collator(
            corpus, self.max_spans_per_doc, self.max_sentences_per_doc, self.dataset)
        res = collator.collate()
        out_name = f"{self.output_directory}/{fold}.{self.file_extension}"
        res.to_jsonl(out_name)


def main():
    args = get_args()
    runner = CollateRunner(**vars(args))
    runner.run()


if __name__ == "__main__":
    main()
