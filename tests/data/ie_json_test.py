"""
Short unit tests to make sure our dataset readers are behaving correctly.
"""

import itertools

from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list
from allennlp.data.vocabulary import Vocabulary

from dygie.data import IEJsonReader


class TestIEJsonReader(AllenNlpTestCase):
    """
    Read in data and do some spot-checks.
    """
    def setUp(self):
        # Sentence lengths: [20, 23, 36, 14, 14, 30, 31, 15].
        # Cumulative sentence lengths: [20, 43, 79, 93, 107, 137, 168, 183].
        self.reader = IEJsonReader(max_span_width=5)
        self.instances = self.reader.read("tests/fixtures/scierc_article.json")

    def tearDown(self):
        pass

    def test_tokens_correct(self):
        instance = self.instances[3]
        tokens = instance["text"].tokens
        assert len(tokens) == 14
        text = [token.text for token in tokens]
        assert text[:4] == ["Secondly", "the", "dynamical", "model"]

    def test_ner_correct(self):
        instance = self.instances[4]
        ner_field = instance["ner_labels"]
        for label, span in zip(ner_field.labels, ner_field.sequence_field.field_list):
            start, end = span.span_start, span.span_end
            if start == 3 and end == 5:
                assert label == "OtherScientificTerm"
            elif start == 10 and end == 12:
                assert label == "Method"
            else:
                assert label == ""

    def test_coref_correct(self):
        # A list, one entry per sentence. For each sentence, a dict mapping spans to cluster id's.
        cluster_mappings = [{(6, 6): 1},
                            {},
                            {(19, 21): 0},
                            {(11, 12): 0, (2, 3): 2},
                            {(3, 5): 0},
                            {(5, 7): 0, (19, 20): 2, (22, 24): 3},
                            {(5, 5): 3},
                            {(2, 2): 1}]
        for instance, cluster_mapping in zip(self.instances, cluster_mappings):
            coref_field = instance["coref_labels"]
            for label, span in zip(coref_field.labels, coref_field.sequence_field.field_list):
                start, end = span.span_start, span.span_end
                if (start, end) in cluster_mapping:
                    assert cluster_mapping[(start, end)] == label
                else:
                    assert label == -1

    def test_relation_correct(self):
        instance = self.instances[5]
        relation_field = instance["relation_labels"]
        span_list = relation_field.sequence_field.field_list
        for label, (span1, span2) in zip(relation_field.labels,
                                         itertools.product(span_list, span_list)):
            if (span1.span_start == 19 and span1.span_end == 20 and
                span2.span_start == 22 and span2.span_end == 24):
                assert label == "USED-FOR"
            else:
                assert label == ""

    def test_vocab_size_correct(self):
        vocab = Vocabulary.from_instances(self.instances)
        # There are 4 unique NER labels and 6 relation labels in the text fixture doc. Need to add 1
        # for the null label.
        assert vocab.get_vocab_size("ner_labels") == 5
        assert vocab.get_vocab_size("relation_labels") == 7
        # For numeric labels, vocab size is 0.
        assert vocab.get_vocab_size("coref_labels") == 0
