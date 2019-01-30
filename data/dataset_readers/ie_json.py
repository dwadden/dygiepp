import logging
import collections
from typing import Any, Dict, List, Optional, Tuple, DefaultDict, Set
import json

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, ListField, TextField, SpanField, MetadataField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.dataset_readers.dataset_utils import Ontonotes, enumerate_spans

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def make_cluster_dict(clusters):
    """
    Returns a dict whose keys are spans, and values are the ID of the cluster of which the span is a
    member.
    """
    res = {}
    for i, spans in enumerate(clusters):
        for span in spans:
            res[tuple(span)] = i
    return res


def cluster_dict_sentence(cluster_dict, sentence_start, sentence_end):
    """
    Split cluster dict into clusters in current sentence, and clusters that come later.
    """
    def within_sentence(span):
        return span[0] >= sentence_start and span[1] <= sentence_end

    # Get the within-sentence spans.
    cluster_sent = {}
    for span, cluster in cluster_dict.items():
        if within_sentence(span):
            cluster_sent[span] = cluster
    # Create new cluster dict with the within-sentence clusters removed.
    new_cluster_dict = {span: cluster for span, cluster in cluster_dict.items()
                        if span not in cluster_sent}
    return cluster_sent, new_cluster_dict


def adjust_offsets(ner, relations, cluster_sent, sentence_start):
    new_ner = [[entry[0] + sentence_start, entry[1] + sentence_start, entry[2]] for entry in ner]
    import ipdb; ipdb.set_trace()


@DatasetReader.register("ie_json")
class IEJsonReader(DatasetReader):
    """
    Reads a single CoNLL-formatted file. This is the same file format as used in the
    :class:`~allennlp.data.dataset_readers.semantic_role_labelling.SrlReader`, but is preprocessed
    to dump all documents into a single file per train, dev and test split. See
    scripts/compile_coref_data.sh for more details of how to pre-process the Ontonotes 5.0 data
    into the correct format.

    Returns a ``Dataset`` where the ``Instances`` have four fields: ``text``, a ``TextField``
    containing the full document text, ``spans``, a ``ListField[SpanField]`` of inclusive start and
    end indices for span candidates, and ``metadata``, a ``MetadataField`` that stores the instance's
    original text. For data with gold cluster labels, we also include the original ``clusters``
    (a list of list of index pairs) and a ``SequenceLabelField`` of cluster ids for every span
    candidate.

    Parameters
    ----------
    max_span_width: ``int``, required.
        The maximum width of candidate spans to consider.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        This is used to index the words in the document.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    """
    def __init__(self,
                 max_span_width: int,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._max_span_width = max_span_width
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as f:
            for line in f:
                sentence_start = 0
                js = json.loads(line)
                doc_key = js["doc_key"]
                cluster_dict = make_cluster_dict(js["clusters"])
                zipped = zip(js["sentences"], js["ner"], js["relations"])
                # Loop over the sentences and convert them to instances.
                for sentence_num, (sentence, ner, relations) in enumerate(zipped):
                    sentence_end = sentence_start + len(sentence) - 1
                    cluster_sent, cluster_dict = cluster_dict_sentence(
                        cluster_dict, sentence_start, sentence_end)
                    yield self.text_to_instance(
                        sentence, ner, relations, cluster_sent, doc_key, sentence_start, sentence_num)

    @overrides
    def text_to_instance(self, sentence, ner, relations, cluster_sent, doc_key, sentence_start,
                         sentence_num):
        """
        TODO(dwadden) document me.
        """

        # Switch from indices in document to indices in sentence.
        ner, relations, cluster_sent = adjust_offsets(
            ner, relations, cluster_sent, sentence_start)

        # flattened_sentences = [self._normalize_word(word)
        #                        for sentence in sentences
        #                        for word in sentence]

        # metadata: Dict[str, Any] = {"original_text": flattened_sentences}
        # if gold_clusters is not None:
        #     metadata["clusters"] = gold_clusters

        # text_field = TextField([Token(word) for word in flattened_sentences], self._token_indexers)

        # cluster_dict = {}
        # if gold_clusters is not None:
        #     for cluster_id, cluster in enumerate(gold_clusters):
        #         for mention in cluster:
        #             cluster_dict[tuple(mention)] = cluster_id

        # spans: List[Field] = []
        # span_labels: Optional[List[int]] = [] if gold_clusters is not None else None

        # sentence_offset = 0
        # for sentence in sentences:
        #     for start, end in enumerate_spans(sentence,
        #                                       offset=sentence_offset,
        #                                       max_span_width=self._max_span_width):
        #         if span_labels is not None:
        #             if (start, end) in cluster_dict:
        #                 span_labels.append(cluster_dict[(start, end)])
        #             else:
        #                 span_labels.append(-1)

        #         spans.append(SpanField(start, end, text_field))
        #     sentence_offset += len(sentence)

        # span_field = ListField(spans)
        # metadata_field = MetadataField(metadata)

        # fields: Dict[str, Field] = {"text": text_field,
        #                             "spans": span_field,
        #                             "metadata": metadata_field}
        # if span_labels is not None:
        #     fields["span_labels"] = SequenceLabelField(span_labels, span_field)

        # return Instance(fields)

    @staticmethod
    def _normalize_word(word):
        if word == "/." or word == "/?":
            return word[1:]
        else:
            return word
