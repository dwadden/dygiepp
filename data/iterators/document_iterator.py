from typing import Iterable
import logging
import numpy as np

from overrides import overrides

from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.dataset import Batch

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DataIterator.register("ie_document")
class DocumentIterator(DataIterator):
    """
    For multi-task IE, we want the training instances in a batch to be successive sentences from the
    same document. Otherwise the coreference labels don't make sense.

    At train time, we use minibatches of sentences from the same document. At evaluation time, read
    in an entire document as a minibatch.
    """
    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        # Make a list indicating whether each entry is the last sentence of the document.
        doc_keys = np.array([instance["metadata"]["doc_key"] for instance in instances])
        # If one document, just set the last sentence manually.
        if len(set(doc_keys)) == 1:
            last_sentences = np.repeat(False, len(doc_keys))
            last_sentences[-1] = True
        # Otherwise get last sentences by comparing document names.
        else:
            rolled = np.roll(doc_keys, -1)
            last_sentences = (doc_keys != rolled).tolist()

        batch = []
        for instance, last_sentence in zip(instances, last_sentences):
            batch.append(instance)
            if last_sentence:
                full_batch = batch
                batch = []
                yield Batch(full_batch)

    @overrides
    def get_num_batches(self, instances: Iterable[Instance]) -> int:
        """
        Get the number of batches.
        """
        n_docs = len(set([instance["metadata"]["doc_key"] for instance in instances]))
        return n_docs
