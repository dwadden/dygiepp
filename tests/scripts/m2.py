"""
Unit tests for the coref module.
"""

import json

from allennlp.common.testing import ModelTestCase
from allennlp.nn import util

from dygie.models import DyGIE
from dygie.data import IEJsonReader


class TestDyGIE(ModelTestCase):
    def setUp(self):
        # TODO(dwadden) create smaller model for testing.
        super(TestDyGIE, self).setUp()
        self.config_file = "../fixtures/dygie_test.jsonnet"
        self.data_file = "../../../../scierc_coref_multitask_bb/data/processed_data/json/train.json"
        self.set_up_model(self.config_file, self.data_file)
        self.test_dataset()

    def get_raw_data(self):
        lines = []
        with open(self.data_file, "r") as f:
            for line in f:
                lines.append(json.loads(line))
        return lines

    def test_dataset(self):
        """
        To compute coreference evaluation metrics, the evaluator needs access to the list of
        coreference clusters, given in the same form as the original input. I check to make sure
        that the clusters I pass in are indeed equivalent to the original input.
        """
        # Pull together the relevant training data.
        data = self.dataset.as_tensor_dict()
        import ipdb; ipdb.set_trace()
        metadata = data["metadata"]
        text_mask = util.get_text_field_mask(data["text"]).float()
        sentence_lengths = text_mask.sum(dim=1).long()
        # Make sure the sentence lengths from the text mask are the same as the number of tokens.
        assert sentence_lengths.tolist() == [len(entry["sentence"]) for entry in metadata]

        # Convert metadata back to form used for coref evaluation
        evaluation_metadata = self.model._coref._make_evaluation_metadata(metadata, sentence_lengths)
        clusters_metadata = evaluation_metadata[0]["clusters"]
        # Convert from tuples to list to facilitate comparison.
        clusters_metadata = [[list(span) for span in cluster] for cluster in clusters_metadata]

        # Get the raw data, and sort to match the metadata.
        clusters_raw = self.get_raw_data()[0]["clusters"]
        clusters_raw = sorted(clusters_raw, key=lambda entry: entry[0][0])

        # Compare the raw data to the converted metadata I have.
        assert clusters_metadata == clusters_raw

t = TestDyGIE()
t.setUp()
