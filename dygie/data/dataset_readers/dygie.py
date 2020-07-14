import logging
import collections
from typing import Any, Dict, List, Optional, Tuple, DefaultDict, Set, Union
import json
import itertools
import os
import pathlib
import pickle as pkl
import warnings

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (Field, ListField, TextField, SpanField, MetadataField,
                                  SequenceLabelField, AdjacencyField)
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.dataset_readers.dataset_utils import enumerate_spans
from allennlp_models.common.ontonotes import Ontonotes

from allennlp.data.fields.span_field import SpanField

from dygie.data.fields.adjacency_field_assym import AdjacencyFieldAssym
from dygie.data.dataset_readers.document import Document, Sentence

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# TODO(dwadden) Add types, unit-test, clean up.

class DyGIEDataException(Exception):
    pass

def make_cluster_dict(clusters: List[List[List[int]]]) -> Dict[SpanField, int]:
    """
    Returns a dict whose keys are spans, and values are the ID of the cluster of which the span is a
    member.
    """
    return {tuple(span): cluster_id for cluster_id, spans in enumerate(clusters) for span in spans}

def cluster_dict_sentence(cluster_dict: Dict[Tuple[int, int], int], sentence_start: int, sentence_end: int):
    """
    Split cluster dict into clusters in current sentence, and clusters that come later.
    """

    def within_sentence(span):
        return span[0] >= sentence_start and span[1] <= sentence_end

    # Get the within-sentence spans.
    cluster_sent = {span: cluster for span, cluster in cluster_dict.items() if within_sentence(span)}

    ## Create new cluster dict with the within-sentence clusters removed.
    new_cluster_dict = {span: cluster for span, cluster in cluster_dict.items()
                        if span not in cluster_sent}

    return cluster_sent, new_cluster_dict


def format_label_fields(cluster_tmp: Dict[Tuple[int,int], int])
    # Coref
    cluster_dict = MissingDict(-1,
        (
            (   (span_start-ss, span_end-ss), cluster_id)
            for ((span_start, span_end), cluster_id) in cluster_tmp.items()
        )
    )

    return cluster_dict


@DatasetReader.register("dygie")
class DyGIEReader(DatasetReader):
    """
    Reads a single JSON-formatted file. This is the same file format as used in the
    scierc, but is preprocessed
    """
    # The predict_hack flag was added post_hoc when I realized I, protocol=pkl.HIGHEST_PROTOCOL needed to
    # return full documents as a single batch when doing prediction.
    def __init__(self,
                 max_span_width: int,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self._max_span_width = max_span_width
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            # Loop over the documents.
            doc_text = json.loads(line)
            instance = self.text_to_instance(doc_text)
            yield instance

    @staticmethod
    def _process_relations(spans, sent):
        if sent.relations is None:
            return None, None

        n_spans = len(spans)
        span_tuples = [(span.span_start, span.span_end) for span in spans]
        candidate_indices = [(i, j) for i in range(n_spans) for j in range(n_spans)]

        relations = []
        relation_indices = []
        for i, j in candidate_indices:
            span_pair = (span_tuples[i], span_tuples[j])
            relation_label = sent.relation_dict.get(span_pair, "")
            if relation_label:
                relation_indices.append((i, j))
                relations.append(relation_label)

        return relations, relation_indices

    @staticmethod
    def _process_events(spans, sent):
        if sent.events is None:
            return None, None, None

        n_tokens = len(sent.text)

        trigger_labels = []
        for i in range(n_tokens):
            trigger_label = sent.trigger_dict.get(i, "")
            trigger_labels.append(trigger_label)

        n_spans = len(spans)
        arguments = []
        argument_indices = []
        candidate_indices = [(i, j) for i in range(n_tokens) for j in range(n_spans)]
        span_tuples = [(span.span_start, span.span_end) for span in spans]

        for i, j in candidate_indices:
            token_span_pair = (i, span_tuples[j])
            argument_label = sent.argument_dict.get(token_span_pair, "")
            if argument_label:
                argument_indices.append((i, j))
                arguments.append(argument_label)

        return trigger_labels, arguments, argument_indices

    def _process_sentence(self, sent: Sentence):
        # Get the sentence text and define the `text_field`.
        sentence_text = [self._normalize_word(word) for word in sent.text]
        text_field = TextField([Token(word) for word in sentence_text], self._token_indexers)

        # Enumerate spans. Store NER labels and create span field over all spans in the text.
        spans = []
        ner_labels = []

        for start, end in enumerate_spans(sentence_text, max_span_width=self._max_span_width):
            span_ix = (start, end)
            ner_label = sent.ner_dict.get(span_ix, "")
            ner_labels.append(ner_label)
            spans.append(SpanField(start, end, text_field))

        # Process relations.
        relation_labels, relation_indices = self._process_relations(spans, sent)

        # Store events triggers.
        trigger_labels, argument_labels, argument_indices = self._process_events(spans, sent)

        # Convert data to fields.
        dataset = sent.doc.dataset
        span_field = ListField(spans)
        fields = {}
        fields["text"] = text_field
        fields["spans"] = span_field
        if ner_labels is not None:
            fields["ner_labels"] = SequenceLabelField(
                ner_labels, span_field, label_namespace=f"{dataset}:ner_labels")
        if relation_labels is not None:
            fields["relation_labels"] = AdjacencyField(
                indices=relation_indices, sequence_field=span_field, labels=relation_labels,
                label_namespace=f"{dataset}:relation_labels")
        if trigger_labels is not None:
            fields["trigger_labels"] = SequenceLabelField(
                trigger_labels, text_field, label_namespace=f"{dataset}:trigger_labels")
            fields["argument_labels"] = AdjacencyFieldAssym(
                indices=argument_indices, row_field=text_field, col_field=span_field,
                labels=argument_labels, label_namespace=f"{dataset}:argument_labels")

        return fields

    def _process_sentence_fields(self, doc: Document):
        # Process each sentence.
        sentence_fields = [self._process_sentence(sent) for sent in doc.sentences]

        # Make sure that all sentences have the same set of keys.
        first_keys = set(sentence_fields[0].keys())
        for entry in sentence_fields:
            if set(entry.keys()) != first_keys:
                raise DyGIEDataException(f"Keys do not match across sentences for document {doc.doc_key}.")

        # For each field, store the data from all sentences together in a ListField.
        fields = {}
        keys = sentence_fields[0].keys()
        for key in keys:
            this_field = ListField([sent[key] for sent in sentence_fields])
            fields[key] = this_field

        return fields

    @overrides
    def text_to_instance(self, doc_text: Dict[str, Any]):
        """
        Convert a Document object into an instance.
        """
        doc = Document(doc_text)

        fields = self._process_sentence_fields(doc)
        fields["metadata"] = MetadataField(doc)

        # Generate fields for text spans, ner labels, coref labels.
        spans = []
        span_ner_labels = []
        span_coref_labels = []

        for start, end in enumerate_spans(sentence, max_span_width=self._max_span_width):
            span_ix = (start, end)
            span_ner_labels.append(ner_dict[span_ix])
            span_coref_labels.append(cluster_dict[span_ix])
            spans.append(SpanField(start, end, text_field))

        span_field = ListField(spans)
        coref_label_field = SequenceLabelField(span_coref_labels, span_field,
                                               label_namespace="coref_labels")


        return Instance(fields)

    @overrides
    def _instances_from_cache_file(self, cache_filename):
        with open(cache_filename, "rb") as f:
            for entry in pkl.load(f):
                yield entry

    @overrides
    def _instances_to_cache_file(self, cache_filename, instances):
        with open(cache_filename, "wb") as f:
            pkl.dump(instances, f, protocol=pkl.HIGHEST_PROTOCOL)


    @staticmethod
    def _normalize_word(word):
        if word == "/." or word == "/?":
            return word[1:]
        else:
            return word
