"""
Dataset reader that takes scierc-formatted json, and converts to the form required by the coref
model for AllenNLP.
"""

import json

from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers import ConllCorefReader
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.fields import Field, ListField, TextField, SpanField, MetadataField, SequenceLabelField
from allennlp.data.tokenizers import Token


@DatasetReader.register("coref_json")
class CorefJsonReader(ConllCorefReader):
    @overrides
    def _read(self, file_path):
        with open(file_path, "r") as f:
            # Loop over the documents.
            for line in f:
                js = json.loads(line)
                sentences = js["sentences"]
                canonical_clusters = [[tuple(span) for span in group] for group in js["clusters"]]
                yield self.text_to_instance(sentences, canonical_clusters)
