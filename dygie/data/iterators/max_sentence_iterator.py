from typing import Iterable
import logging

from allennlp.data.instance import Instance
from allennlp.data.dataloader import DataLoader, PyTorchDataLoader
from allennlp.data.batch import Batch

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


# TODO(dwadden / kenny): This actually doesn't work. Things have changed in AllenNLP V1 and we need
# to update the Sampler, not the DataLoader. More info here:
# https://guide.allennlp.org/reading-data#1.
@DataLoader.register("max_sentence")
class MaxSentenceIterator(PyTorchDataLoader):
    """
    First arranges dataset by number of sentences, then yields batches according to sentence limit.
    """
    def __init__(self, max_sentences, **kwargs):
        super().__init__(**kwargs)
        self._max_sentences = max_sentences

    def _create_batches(self, instances: Iterable[Instance]) -> Iterable[Batch]:
        # Sort dataset by length
        instances = sorted(instances, key=lambda instance: instance["text"].sequence_length())
        curr_batch = []
        total_sentences_in_batch = 0

        for instance in instances:
            doc_length = instance["text"].sequence_length()
            # If we're under the limit, add to current batch
            if total_sentences_in_batch + doc_length <= self._max_sentences:
                curr_batch.append(instance)
                total_sentences_in_batch += doc_length
            # Else, reset and yield the current batch.
            else:
                batch_to_yield = curr_batch
                curr_batch = [instance]
                total_sentences_in_batch = instance["text"].sequence_length()
                yield Batch(batch_to_yield)

        yield Batch(curr_batch)  # Yield last batch regardless of size
