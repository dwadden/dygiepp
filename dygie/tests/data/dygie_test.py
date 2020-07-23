"""
Short unit tests to make sure our dataset readers are behaving correctly.
Checks a sample from the scierc data
"""

import unittest
from allennlp.data.vocabulary import Vocabulary

from dygie.data import DyGIEReader


class TestDygieReader(unittest.TestCase):

    def setUp(self):
        # scierc
        # Sentence lengths: [20, 23, 36, 14, 14, 30, 31, 15].
        # Cumulative sentence lengths: [20, 43, 79, 93, 107, 137, 168, 183].
        self.reader = DyGIEReader(max_span_width=5)
        self.dataset = self.reader.read("dygie/tests/fixtures/scierc_article.json")

    def tearDown(self):
        pass

    def test_tokens_correct_scierc(self):
        # instances are now entire documents instead of sentences
        instance = self.dataset.instances[0]
        tokens = instance["text"][4][0:]
        assert len(tokens) == 14
        text = [token.text for token in tokens]
        assert text[:6] == ["Thirdly", "the", "learned", "intrinsic", "object", "structure"]

    def test_ner_correct_scierc(self):
        instance = self.dataset.instances[0]
        ner_field = instance["ner_labels"][3]
        spans = instance["spans"][3]

        for label, span in zip(ner_field, spans):
            start, end = span.span_start, span.span_end
            if start == 2 and end == 3:
                assert label.label == "Method"
            elif start == 11 and end == 12:
                assert label.label == "Method"
            else:
                assert label.label == ""

    def test_relation_correct_scierc(self):
        instance = self.dataset.instances[0]
        relation_field = instance["relation_labels"][5]
        span_list = relation_field.sequence_field
        # There should be one relation in this sentence,
        indices = relation_field.indices
        labels = relation_field.labels
        assert len(indices) == len(labels) == 1
        ix = indices[0]
        label = labels[0]
        # Check that the relation refers to the correct spans
        span1 = span_list[ix[0]]
        span2 = span_list[ix[1]]
        assert ((span1.span_start == 19 and span1.span_end == 20 and
                 span2.span_start == 22 and span2.span_end == 24))
        # Check that the label's correct.
        assert label == "USED-FOR"

    def test_coref_correct_scierc(self):
        instance = self.dataset.instances[0]
        coref_field = instance["coref_labels"]
        spans = instance["spans"]
        # A list, one entry per sentence. For each sentence, a dict mapping spans to cluster id's.
        cluster_mappings = [{(6, 6): 1},
                            {},
                            {(19, 21): 0},
                            {(11, 12): 0, (2, 3): 2},
                            {(3, 5): 0},
                            {(5, 7): 0, (19, 20): 2, (22, 24): 3},
                            {(5, 5): 3},
                            {(2, 2): 1}]
        for instance, cluster_mapping, span in zip(coref_field, cluster_mappings, spans):
            curr_coref_field = instance
            curr_span = span
            for label, span in zip(curr_coref_field, curr_span):
                start, end = span.span_start, span.span_end
                if (start, end) in cluster_mapping:
                    # print(start, end)
                    # print(label.label)
                    assert cluster_mapping[(start, end)] == label.label
                else:
                    assert label.label == -1

    def test_vocab_size_correct_scierc(self):
        vocab = Vocabulary.from_instances(self.dataset.instances)
        # There are 4 unique NER labels and 6 relation labels in the text fixture doc. For the ner
        # labels, there is an extra category for the null label. For the relation labels, there
        # isn't. This is due to the way their respective `Field`s represent labels.
        assert vocab.get_vocab_size("None__ner_labels") == 5
        assert vocab.get_vocab_size("None__relation_labels") == 6
        # For numeric labels, vocab size is 0.
        assert vocab.get_vocab_size("coref_labels") == 0


if __name__ == "__main__":
    unittest.main()
