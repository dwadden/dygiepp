from collections import deque
from typing import Iterable, Deque
import logging
import numpy as np

from overrides import overrides

from allennlp.common.util import lazy_groups_of
from allennlp.data.instance import Instance
from allennlp.data.dataloader import DataLoader, PyTorchDataLoader
from allennlp.data.batch import Batch

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DataLoader.register("max_sentence_batch")
class BatchIterator(PyTorchDataLoader):
    """
    First arranges dataset by number of sentences
    """

    def _create_batches(self, instances: Iterable[Instance], sentence_limit: 12):
        # sort dataset by length
        instances.sort(key=lambda instance: instance["text"].sequence_length())
        curr_batch = []
        total_sentences_in_batch = 0

        for instance in instances:
            doc_length = instance["text"].sequence_length()
            # If we're under the limit, add to current batch
            if total_sentences_in_batch + doc_length <= sentence_limit:
                curr_batch.append(instance)
                total_sentences_in_batch += doc_length
            # Else, reset and yield the current batch.
            else:
                batch_to_yield = curr_batch
                curr_batch = [instance]
                total_sentences_in_batch = instance["text"].sequence_length()
                yield Batch(batch_to_yield)
        yield Batch(curr_batch) #yield last batch regardless of size
