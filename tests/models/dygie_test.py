"""
Unit tests for the dygie.
"""

from allennlp.common.testing import ModelTestCase

# TODO(dwadden) Figure out why tests break on CUDA.


class TestDyGIE(ModelTestCase):
    def setUp(self):
        # TODO(dwadden) create smaller model for testing.
        super(TestDyGIE, self).setUp()
        self.config_file = "tests/fixtures/dygie_test_full.jsonnet"
        self.data_file = "tests/fixtures/scierc_article.json"
        self.set_up_model(self.config_file, self.data_file)

    def test_dygie_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
