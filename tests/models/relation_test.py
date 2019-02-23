"""
Unit tests for the relation module.
"""

import json

import torch

from allennlp.common.testing import ModelTestCase
from allennlp.nn import util
from allennlp.data.dataset import Batch

from dygie.models import DyGIE
from dygie.data import IEJsonReader
from dygie.data import DocumentIterator


class TestRelation(ModelTestCase):
    def setUp(self):
        # TODO(dwadden) create smaller model for testing.
        super(TestRelation, self).setUp()
        self.config_file = "tests/fixtures/dygie_test.jsonnet"
        self.data_file = "tests/fixtures/scierc_article.json"
        self.set_up_model(self.config_file, self.data_file)
        self.reader = IEJsonReader(max_span_width=5)
        self.instances = self.reader.read("tests/fixtures/scierc_article.json")

    def get_raw_data(self):
        lines = []
        with open(self.data_file, "r") as f:
            for line in f:
                lines.append(json.loads(line))
        return lines

    def test_compute_span_pair_embeddings(self):
        top_span_embeddings = torch.randn([3, 51, 1220])  # Make up random embeddings.

        embeddings = self.model._relation._compute_span_pair_embeddings(top_span_embeddings)

        batch_ix = 1
        ix1 = 22
        ix2 = 43
        emb1 = top_span_embeddings[batch_ix, ix1]
        emb2 = top_span_embeddings[batch_ix, ix2]
        emb_prod = emb1 * emb2
        emb = torch.cat([emb1, emb2, emb_prod])

        assert torch.allclose(emb, embeddings[batch_ix, ix1, ix2])

    def test_compute_relation_scores(self):
        self.model.eval()       # Need eval on in order to reproduce.
        relation = self.model._relation
        pairwise_embeddings = torch.randn(3, 46, 46, 3660, requires_grad=True)
        top_span_mention_scores = torch.randn(3, 46, 1, requires_grad=True)

        scores = relation._compute_relation_scores(pairwise_embeddings, top_span_mention_scores)

        batch_ix = 0
        ix1 = 31
        ix2 = 4

        score = relation._relation_scorer(
            relation._relation_feedforward(pairwise_embeddings[batch_ix, ix1, ix2].unsqueeze(0)))
        score += top_span_mention_scores[batch_ix, ix1] + top_span_mention_scores[batch_ix, ix2]
        score = torch.cat([torch.tensor([0.0]), score.squeeze()])

        assert torch.allclose(scores[batch_ix, ix1, ix2], score)

    def test_get_pruned_gold_relations(self):
        pass
        # import ipdb; ipdb.set_trace()
        # foo = Batch(self.instances)
        # for batch in it(self.instances):
        #     import ipdb; ipdb.set_trace()

        # instance = self.instances[6]
