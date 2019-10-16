from collections import deque, defaultdict
import math
from typing import List, Dict, Iterable, Any, Set, Deque
import logging
import numpy as np

from overrides import overrides

from allennlp.common.util import lazy_groups_of
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.dataset import Batch

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def get_dataset_instances(instances, dataset):
    return [instance for instance in instances if instance["metadata"]["dataset"] == dataset]


# The idea for this iterator is borrowed from
# https://github.com/allenai/allennlp/blob/master/allennlp/tests/training/multi_task_trainer_test.py#L225
@DataIterator.register("ie_multitask")
class MultiTaskIterator(DataIterator):
    """
    To use when we're co-training on Ontonotes and ACE.
    """
    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        # Shuffle the documents if requested.
        ace_instances = self._shuffle_documents(get_dataset_instances(instances, "ace"))
        ontonotes_instances = self._shuffle_documents(get_dataset_instances(instances, "ontonotes"))
        n_ontonotes = math.floor(len(ace_instances) / 2)
        ontonotes_instances = ontonotes_instances[:n_ontonotes]

        all_instances = self._shuffle_documents(ace_instances + ontonotes_instances)
        hoppers: Dict[Any, List[Instance]] = defaultdict(list)

        for instance in all_instances:
            # Which hopper do we put this instance in?
            instance_type = (instance["metadata"]["dataset"]
                             if "dataset" in instance["metadata"]
                             else None)

            hoppers[instance_type].append(instance)

            # If the hopper is full, yield up the batch and clear it.
            if len(hoppers[instance_type]) >= self._batch_size:
                yield Batch(hoppers[instance_type])
                hoppers[instance_type].clear()

        # Deal with leftovers
        for remaining in hoppers.values():
            if remaining:
                yield Batch(remaining)

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

    def get_num_batches(self, instances):
        ace_instances = get_dataset_instances(instances, "ace")
        n_batches = len(ace_instances) + math.floor(len(ace_instances) / 2)
        return math.ceil(n_batches / self._batch_size)
