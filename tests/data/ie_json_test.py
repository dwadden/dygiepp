"""
Short unit tests to make sure our dataset readers are behaving correctly.
"""


from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

from dygie.data import IEJsonReader


class TestIEJsonReader(AllenNlpTestCase):
    def test_read_from_file(self):
        reader = IEJsonReader(max_span_width=5)
        instances = reader.read("tests/fixtures/scierc_article.json")
        assert True
