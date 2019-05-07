"""
Class to handle span propagations.
"""

from typing import Optional

from torch.nn import functional as F
import torch

from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.data import Vocabulary


class SpanProp(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 n_span_prop: int,
                 emb_dim: int,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None):
        # There's a hack. We also need the trig arg embeddings creator and the argument scorer. I
        # just set those explicitly in the init of dygie.
        super(SpanProp, self).__init__(vocab, regularizer)

        self._n_argument_labels = vocab.get_vocab_size("argument_labels")
        self._n_span_prop = n_span_prop

        # These are the A matrices in my writeup.
        self._transition_trigger = torch.nn.Linear(
            in_features=self._n_argument_labels, out_features=emb_dim)
        self._transition_argument = torch.nn.Linear(
            in_features=self._n_argument_labels, out_features=emb_dim)

        # These are the gates.
        self._gate_trigger = torch.nn.Linear(in_features=2 * emb_dim, out_features=emb_dim)
        self._gate_argument = torch.nn.Linear(in_features=2 * emb_dim, out_features=emb_dim)

        initializer(self)

    def forward(self, trig_embeddings, arg_embeddings, trig_mask, arg_mask, trig_scores, arg_scores):
        n_trig = trig_embeddings.size(1)
        n_arg = arg_embeddings.size(1)

        arg_mask_unsqueezed = arg_mask.unsqueeze(1).float()
        trig_mask_unsqueezed = trig_mask.unsqueeze(2).float()

        # TODO(dwadden write a correctness spot-check.)
        for _ in range(self._n_span_prop):
            pairwise_embeddings = self._trig_arg_embedder(
                trig_embeddings, arg_embeddings)
            argument_scores = self._argument_scorer(
                pairwise_embeddings, trig_scores, arg_scores, prepend_zeros=False)
            truncated_scores = F.relu(argument_scores)

            # The trigger update.
            # TODO(dwadden) time permitting do this with matrix mults instead.
            # TODO(dwadden) Add normalization like Ulme?
            transition_trig = self._transition_trigger(truncated_scores)
            # Mask appropriately so that masked args won't contribute to update.
            arg_embeddings_tiled = (arg_embeddings.unsqueeze(1).repeat(1, n_trig, 1, 1) *
                                    arg_mask_unsqueezed)
            updates_trig = torch.sum(transition_trig * arg_embeddings_tiled, dim=2)
            gate_trig = torch.sigmoid(
                self._gate_trigger(torch.cat([trig_embeddings, updates_trig], dim=-1)))
            trig_embeddings = gate_trig * trig_embeddings + (1 - gate_trig) * updates_trig

            # Now the arguments
            transition_arg = self._transition_argument(truncated_scores)
            # As above, masked triggers shouldn't be part of the update.
            trig_embeddings_tiled = (trig_embeddings.unsqueeze(2).repeat(1, 1, n_arg, 1) *
                                     trig_mask_unsqueezed)
            updates_arg = torch.sum(transition_arg * trig_embeddings_tiled, dim=1)
            gate_arg = torch.sigmoid(
                self._gate_argument(torch.cat([arg_embeddings, updates_arg], dim=-1)))
            arg_embeddings = gate_arg * arg_embeddings + (1 - gate_arg) * updates_arg

        return trig_embeddings, arg_embeddings
