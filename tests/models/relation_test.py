"""
Unit tests for the relation module.

This module wasn't matching TensorFlow performance so it's tested pretty heavily.
"""

import torch

from allennlp.common.testing import ModelTestCase

# Needed to get the test framework to see the dataset readers and models.
from dygie import models
from dygie import data


class TestRelation(ModelTestCase):
    def setUp(self):
        super(TestRelation, self).setUp()
        self.config_file = "tests/fixtures/dygie_test.jsonnet"
        self.data_file = "tests/fixtures/scierc_article.json"
        self.set_up_model(self.config_file, self.data_file)

    def test_decode(self):
        def convert(x):
            return self.model.vocab.get_token_from_index(x, namespace="relation_labels")

        top_spans = torch.tensor([[[0, 2], [1, 3], [1, 3]],
                                  [[1, 6], [2, 4], [3, 8]],
                                  [[0, 1], [0, 1], [0, 1]]])
        predicted_relations = torch.tensor([[[-1, -1, 1],
                                             [1, -1, -1],
                                             [-1, 0, -1]],
                                            [[-1, -1, -1],
                                             [1, -1, 2],
                                             [-1, -1, 4]],
                                            [[1, 1, 2],
                                             [1, 3, 2],
                                             [-1, 2, 1]]])
        num_spans_to_keep = torch.tensor([2, 3, 0])
        predict_dict = {"top_spans": top_spans,
                        "predicted_relations": predicted_relations,
                        "num_spans_to_keep": num_spans_to_keep}
        decoded = self.model._relation.decode(predict_dict)
        expected = [{((1, 3), (0, 2)): convert(1)},
                    {((2, 4), (1, 6)): convert(1),
                     ((2, 4), (3, 8)): convert(2),
                     ((3, 8), (3, 8)): convert(4)},
                    {}]
        assert expected == decoded["decoded_relations_dict"]

    def test_compute_span_pair_embeddings(self):
        top_span_embeddings = torch.randn([3, 51, 1160])  # Make up random embeddings.

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
        pairwise_embeddings = torch.randn(3, 46, 46, 3480, requires_grad=True)
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
        # Getting the pruned gold labels should add one to the input relation labels, then set all
        # the masked entries to -1.
        relation_labels = torch.tensor([[[-1, -1, 2, 3],
                                         [1, -1, -1, 0],
                                         [-1, 3, -1, 1],
                                         [0, -1, -1, -1]],
                                        [[0, 2, 1, 2],
                                         [-1, -1, -1, -1],
                                         [3, 0, -1, -1],
                                         [-1, 0, 1, -1]]])
        top_span_indices = torch.tensor([[0, 1, 3],
                                         [0, 2, 2]])
        top_span_masks = torch.tensor([[1, 1, 1],
                                       [1, 1, 0]]).unsqueeze(-1)

        labels = self.model._relation._get_pruned_gold_relations(
            relation_labels, top_span_indices, top_span_masks)

        expected_labels = torch.tensor([[[0, 0, 4],
                                         [2, 0, 1],
                                         [1, 0, 0]],
                                        [[1, 2, -1],
                                         [4, 0, -1],
                                         [-1, -1, -1]]])

        assert torch.equal(labels, expected_labels)

    def test_cross_entropy_ignore_index(self):
        # Make sure that the cross entropy loss is ignoring entries whose gold label is -1, which
        # corresponds, to masked-out entries.
        relation_scores = torch.randn(2, 3, 3, self.model._relation._n_labels + 1)
        gold_relations = torch.tensor([[[0, 0, 4],
                                        [2, 0, 1],
                                        [1, 0, 0]],
                                       [[1, 2, -1],
                                        [4, 0, -1],
                                        [-1, -1, -1]]])

        # Calculate the loss with a loop over entries.
        total_loss = torch.tensor([0.0])
        for fold in [0, 1]:
            for i in range(3):
                for j in range(3):
                    scores_entry = relation_scores[fold, i, j].unsqueeze(0)
                    gold_entry = gold_relations[fold, i, j].unsqueeze(0)
                    if gold_entry >= 0:
                        loss_entry = self.model._relation._loss(scores_entry, gold_entry)
                        total_loss += loss_entry

        model_loss = self.model._relation._get_cross_entropy_loss(relation_scores, gold_relations)
        assert torch.allclose(total_loss, model_loss)
