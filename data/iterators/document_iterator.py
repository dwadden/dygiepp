from collections import deque
from typing import Iterable, Deque
import logging
import random

from allennlp.common.util import lazy_groups_of
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.dataset import Batch

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DataIterator.register("document")
class DocumentIterator(DataIterator):
    """
    For multi-task IE, we want the training instances in a batch to be successive sentences from the
    same document. Otherwise the coreference labels don't make sense.

    At train time, we use minibatches of sentences from the same document. At evaluation time, read
    in an entire document as a minibatch.
    """
    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        # TODO(dwadden) Write minimal unit test.
        doc_key = instances[0]["metadata"]["doc_key"]
        this_batch = []

        total_length = 0

        for i, instance in enumerate(instances):
            # If this sentence is from the current document, append to instance list.
            if instance["metadata"]["doc_key"] == doc_key:
                this_batch.append(instance)
                # If we've got a full batch, yield it and reset the batch.
                # If batch_size is -1, then just do entire documents at a time. For evaluation.
                # Also, if we're on the final instance in the batch, yield.
                if ((self._batch_size >= 0 and len(this_batch) == self._batch_size) or
                    (i == len(instances) - 1)):
                    full_batch = this_batch
                    this_batch = []
                    total_length += len(full_batch)
                    yield Batch(full_batch)
            # If we've hit the start of a new document, yield the old one and create a new batch.
            else:
                full_batch = this_batch
                this_batch = [instance]
                doc_key = instance["metadata"]["doc_key"]
                # Check to make sure the batch has at least one entry.
                if full_batch:
                    total_length += len(full_batch)
                    yield Batch(full_batch)
