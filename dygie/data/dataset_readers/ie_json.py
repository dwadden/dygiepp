import logging
import collections
from typing import Any, Dict, List, Optional, Tuple, DefaultDict, Set, Union
import json
import itertools
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
from allennlp.data.dataset_readers.dataset_utils import Ontonotes, enumerate_spans


from allennlp.data.fields.span_field import SpanField

from dygie.data.fields.adjacency_field_assym import AdjacencyFieldAssym

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# TODO(dwadden) Add types, unit-test, clean up.

class MissingDict(dict):
    """
    If key isn't there, returns default value. Like defaultdict, but it doesn't store the missing
    keys that were queried.
    """
    def __init__(self, missing_val, generator=None) -> None:
        if generator:
            super().__init__(generator)
        else:
            super().__init__()
        self._missing_val = missing_val

    def __missing__(self, key):
        return self._missing_val

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


def format_label_fields(ner: List[List[Union[int,str]]],
                        relations: List[List[Union[int,str]]],
                        cluster_tmp: Dict[Tuple[int,int], int],
                        events: List[List[Union[int,str]]],
                        sentence_start: int) -> Tuple[Dict[Tuple[int,int],str],
                                                      Dict[Tuple[Tuple[int,int],Tuple[int,int]],str],
                                                      Dict[Tuple[int,int],int]]:
    """
    Format the label fields, making the following changes:
    1. Span indices should be with respect to sentence, not document.
    2. Return dicts whose keys are spans (or span pairs) and whose values are labels.
    """
    ss = sentence_start
    # NER
    ner_dict = MissingDict("",
        (
            ((span_start-ss, span_end-ss), named_entity)
            for (span_start, span_end, named_entity) in ner
        )
    )

    # Relations
    relation_dict = MissingDict("",
        (
            ((  (span1_start-ss, span1_end-ss),  (span2_start-ss, span2_end-ss)   ), relation)
            for (span1_start, span1_end, span2_start, span2_end, relation) in relations
        )
    )

    # Coref
    cluster_dict = MissingDict(-1,
        (
            (   (span_start-ss, span_end-ss), cluster_id)
            for ((span_start, span_end), cluster_id) in cluster_tmp.items()
        )
    )

    # Events. There are two structures. The `trigger_dict` is a mapping from span pairs to the
    # trigger labels. The `arg_dict` maps from (trigger_span, arg_span) pairs to trigger labels.
    trigger_dict = MissingDict("")
    arg_dict = MissingDict("")
    for event in events:
        the_trigger = event[0]
        the_args = event[1:]
        trigger_dict[the_trigger[0] - ss] = the_trigger[1]
        for the_arg in the_args:
            arg_dict[(the_trigger[0] - ss, (the_arg[0] - ss, the_arg[1] - ss))] = the_arg[2]

    return ner_dict, relation_dict, cluster_dict, trigger_dict, arg_dict


@DatasetReader.register("ie_json")
class IEJsonReader(DatasetReader):
    """
    Reads a single JSON-formatted file. This is the same file format as used in the
    scierc, but is preprocessed
    """
    # The predict_hack flag was added post_hoc when I realized I needed to
    # return full documents as a single batch when doing prediction.
    def __init__(self,
                 max_span_width: int,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 context_width: int = 1,
                 debug: bool = False,
                 lazy: bool = False,
                 predict_hack: bool = False) -> None:
        super().__init__(lazy)
        assert (context_width % 2 == 1) and (context_width > 0)
        self.k = int( (context_width - 1) / 2)
        self._max_span_width = max_span_width
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._debug = debug
        self._n_debug_docs = 10
        self._predict_hack = predict_hack

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as f:
            lines = f.readlines()
        # If we're debugging, only do the first 10 documents.
        if self._debug:
            lines = lines[:self._n_debug_docs]

        for line in lines:
            # Loop over the documents.
            sentence_start = 0
            js = json.loads(line)
            doc_key = js["doc_key"]
            dataset = js["dataset"] if "dataset" in js else None

            # If some fields are missing in the data set, fill them with empties.
            # TODO(dwadden) do this more cleanly once things are running.
            n_sentences = len(js["sentences"])
            # TODO(Ulme) Make it so that the
            js["sentence_groups"] = [[self._normalize_word(word) for sentence in js["sentences"][max(0, i-self.k):min(n_sentences, i + self.k + 1)] for word in sentence] for i in range(n_sentences)]
            js["sentence_start_index"] = [sum(len(js["sentences"][i-j-1]) for j in range(min(self.k, i))) if i > 0 else 0 for i in range(n_sentences)]
            js["sentence_end_index"] = [js["sentence_start_index"][i] + len(js["sentences"][i]) for i in range(n_sentences)]
            for sentence_group_nr in range(len(js["sentence_groups"])):
                if len(js["sentence_groups"][sentence_group_nr]) > 300:
                    js["sentence_groups"][sentence_group_nr] = js["sentences"][sentence_group_nr]
                    js["sentence_start_index"][sentence_group_nr] = 0
                    js["sentence_end_index"][sentence_group_nr] = len(js["sentences"][sentence_group_nr])
                    if len(js["sentence_groups"][sentence_group_nr])>300:
                        warnings.warn("Sentence with > 300 words; BERT may truncate.")
            if "clusters" not in js:
                js["clusters"] = []
            for field in ["ner", "relations", "events"]:
                if field not in js:
                    js[field] = [[] for _ in range(n_sentences)]

            cluster_dict_doc = make_cluster_dict(js["clusters"])
            #zipped = zip(js["sentences"], js["ner"], js["relations"], js["events"])
            zipped = zip(js["sentences"], js["ner"], js["relations"], js["events"], js["sentence_groups"], js["sentence_start_index"], js["sentence_end_index"])

            # Loop over the sentences.
            if self._predict_hack:
                instances = []

            for sentence_num, (sentence, ner, relations, events, groups, start_ix, end_ix) in enumerate(zipped):

                sentence_end = sentence_start + len(sentence) - 1
                cluster_tmp, cluster_dict_doc = cluster_dict_sentence(
                    cluster_dict_doc, sentence_start, sentence_end)

                # TODO(dwadden) too many outputs. Re-write as a dictionary.
                # Make span indices relative to sentence instead of document.
                ner_dict, relation_dict, cluster_dict, trigger_dict, argument_dict = \
                    format_label_fields(ner, relations, cluster_tmp, events, sentence_start)
                sentence_start += len(sentence)
                instance = self.text_to_instance(
                    sentence, ner_dict, relation_dict, cluster_dict, trigger_dict, argument_dict,
                    doc_key, dataset, sentence_num, groups, start_ix, end_ix, ner, relations, events)

                if self._predict_hack:
                    instances.append(instance)
                else:
                    yield instance

            if self._predict_hack:
                yield(instances)


    @overrides
    def text_to_instance(self,
                         sentence: List[str],
                         ner_dict: Dict[Tuple[int, int], str],
                         relation_dict,
                         cluster_dict,
                         trigger_dict,
                         argument_dict,
                         doc_key: str,
                         dataset: str,
                         sentence_num: int,
                         groups: List[str],
                         start_ix: int,
                         end_ix: int,
                         ner: List,
                         relations: List,
                         events: List):
        """
        TODO(dwadden) document me.
        """

        sentence = [self._normalize_word(word) for word in sentence]

        text_field = TextField([Token(word) for word in sentence], self._token_indexers)
        text_field_with_context = TextField([Token(word) for word in groups], self._token_indexers)

        # Put together the metadata.
        metadata = dict(sentence=sentence,
                        ner_dict=ner_dict,
                        relation_dict=relation_dict,
                        cluster_dict=cluster_dict,
                        trigger_dict=trigger_dict,
                        argument_dict=argument_dict,
                        doc_key=doc_key,
                        dataset=dataset,
                        groups=groups,
                        start_ix=start_ix,
                        end_ix=end_ix,
                        sentence_num=sentence_num,
                        ner=ner,
                        relations=relations,
                        events=events)

        metadata_field = MetadataField(metadata)

        # Trigger labels. One label per token in the input.
        token_trigger_labels = []
        for i in range(len(text_field)):
            token_trigger_labels.append(trigger_dict[i])

        trigger_label_field = SequenceLabelField(token_trigger_labels, text_field,
                                                 label_namespace="trigger_labels")

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
        ner_label_field = SequenceLabelField(span_ner_labels, span_field,
                                             label_namespace="ner_labels")
        coref_label_field = SequenceLabelField(span_coref_labels, span_field,
                                               label_namespace="coref_labels")

        # Generate labels for relations and arguments. Only store non-null values.
        # For the arguments, by convention the first span specifies the trigger, and the second
        # specifies the argument. Ideally we'd have an adjacency field between (token, span) pairs
        # for the event arguments field, but AllenNLP doesn't make it possible to express
        # adjacencies between two different sequences.
        n_spans = len(spans)
        span_tuples = [(span.span_start, span.span_end) for span in spans]
        candidate_indices = [(i, j) for i in range(n_spans) for j in range(n_spans)]

        relations = []
        relation_indices = []
        for i, j in candidate_indices:
            span_pair = (span_tuples[i], span_tuples[j])
            relation_label = relation_dict[span_pair]
            if relation_label:
                relation_indices.append((i, j))
                relations.append(relation_label)

        relation_label_field = AdjacencyField(
            indices=relation_indices, sequence_field=span_field, labels=relations,
            label_namespace="relation_labels")

        arguments = []
        argument_indices = []
        n_tokens = len(sentence)
        candidate_indices = [(i, j) for i in range(n_tokens) for j in range(n_spans)]
        for i, j in candidate_indices:
            token_span_pair = (i, span_tuples[j])
            argument_label = argument_dict[token_span_pair]
            if argument_label:
                argument_indices.append((i, j))
                arguments.append(argument_label)

        argument_label_field = AdjacencyFieldAssym(
            indices=argument_indices, row_field=text_field, col_field=span_field, labels=arguments,
            label_namespace="argument_labels")

        # Pull it  all together.
        fields = dict(text=text_field_with_context,
                      spans=span_field,
                      ner_labels=ner_label_field,
                      coref_labels=coref_label_field,
                      trigger_labels=trigger_label_field,
                      argument_labels=argument_label_field,
                      relation_labels=relation_label_field,
                      metadata=metadata_field)

        return Instance(fields)

    @overrides
    def _instances_from_cache_file(self, cache_filename):
        with open(cache_filename, "rb") as f:
            for entry in pkl.load(f):
                yield entry

    @overrides
    def _instances_to_cache_file(self, cache_filename, instances):
        with open(cache_filename, "wb") as f:
            pkl.dump(instances, f)


    @staticmethod
    def _normalize_word(word):
        if word == "/." or word == "/?":
            return word[1:]
        else:
            return word
