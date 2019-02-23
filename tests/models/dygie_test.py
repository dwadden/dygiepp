"""
Unit tests for the dygie.
"""

from allennlp.common.testing import ModelTestCase


class TestDyGIE(ModelTestCase):
    def setUp(self):
        # TODO(dwadden) create smaller model for testing.
        super(TestDyGIE, self).setUp()
        self.config_file = "tests/fixtures/dygie_test.jsonnet"
        self.data_file = "tests/fixtures/scierc_article.json"
        self.set_up_model(self.config_file, self.data_file)

    def test_dygie_model_can_train_svae_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
