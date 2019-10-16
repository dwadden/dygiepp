from collections import deque
from typing import Iterable, Deque
import logging
import numpy as np

from overrides import overrides

from allennlp.common.util import lazy_groups_of
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.dataset import Batch

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DataIterator.register("ie_batch")
class BatchIterator(DataIterator):
    """
    For multi-task IE, we want the training instances in a batch to be successive sentences from the
    same document. Otherwise the coreference labels don't make sense.

    At train time, if `shuffle` is True, shuffle the documents but not the instances within them.
    Then, do the same thing as AllenNLP BasicIterator.
    """
    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        # Shuffle the documents if requested.
        maybe_shuffled_instances = self._shuffle_documents(instances) if shuffle else instances

        for instance_list in self._memory_sized_lists(maybe_shuffled_instances):
            iterator = iter(instance_list)
            excess: Deque[Instance] = deque()
            # Then break each memory-sized list into batches.
            for batch_instances in lazy_groups_of(iterator, self._batch_size):
                for possibly_smaller_batches in self._ensure_batch_is_sufficiently_small(batch_instances, excess):
                    batch = Batch(possibly_smaller_batches)
                    yield batch
            if excess:
                yield Batch(excess)

    @staticmethod
    def _shuffle_documents(instances):
        """
        Randomly permute the documents for each batch
        """
        doc_keys = np.array([instance["metadata"]["doc_key"] for instance in instances])
        shuffled = np.random.permutation(np.unique(doc_keys))
        res = []
        for doc in shuffled:
            ixs = np.nonzero(doc_keys == doc)[0].tolist()
            doc_instances = [instances[ix] for ix in ixs]
            sentence_nums = [entry["metadata"]["sentence_num"] for entry in doc_instances]
            assert sentence_nums == list(range(len(doc_instances)))  # Make sure sentences are in order.
            res.extend(doc_instances)
        assert len(res) == len(instances)
        return res
