"""
Class to handle span propagations.
"""

from typing import Optional

from torch.nn import functional as F
import torch

from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.data import Vocabulary


# TODO(dwadden) We don't re-score the trigger and argument scores during this process. That's
# what Yi does. But maybe we should do this...


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
        # First, the code to check it works.
        def check_trigger():
            # Spot-check a single entry of the trigger update matrix.
            batch_size = trig_embeddings.size(0)
            batch_ix = max(batch_size - 3, 0)
            these_truncated_scores = truncated_scores[batch_ix]
            this_trig_mask = trig_mask[batch_ix]
            num_trigs = this_trig_mask.sum()
            this_arg_mask = arg_mask[batch_ix]
            num_args = this_arg_mask.sum()
            batch_trig_emb = trig_embeddings[batch_ix]
            batch_arg_emb = arg_embeddings[batch_ix]
            trig_ix = max(num_trigs - 1, 0)
            this_trig_emb = batch_trig_emb[trig_ix]

            # Compute expected trigger update for a single entry and make sure it matches.
            trig_update = torch.zeros_like(this_trig_emb)
            for j in range(num_args):
                transitioned = self._transition_trigger(these_truncated_scores[trig_ix, j])
                update_contribution = transitioned * batch_arg_emb[j]
                trig_update += update_contribution

            expected_trig_update = updates_trig[batch_ix, trig_ix]
            assert torch.allclose(trig_update, expected_trig_update)

        def check_arg():
            # Spot-check a single entry of the argument update.
            batch_size = trig_embeddings.size(0)
            batch_ix = max(batch_size - 3, 0)
            these_truncated_scores = truncated_scores[batch_ix]
            this_trig_mask = trig_mask[batch_ix]
            num_trigs = this_trig_mask.sum()
            this_arg_mask = arg_mask[batch_ix]
            num_args = this_arg_mask.sum()
            batch_trig_emb = trig_embeddings[batch_ix]
            batch_arg_emb = arg_embeddings[batch_ix]
            arg_ix = max(num_args - 4, 0)
            this_arg_emb = batch_arg_emb[arg_ix]

            # Compute expected trigger update for a single entry and make sure it matches.
            arg_update = torch.zeros_like(this_arg_emb)
            for i in range(num_trigs):
                transitioned = self._transition_argument(these_truncated_scores[i, arg_ix])
                update_contribution = transitioned * batch_trig_emb[i]
                arg_update += update_contribution

            expected_arg_update = updates_arg[batch_ix, arg_ix]
            assert torch.allclose(arg_update, expected_arg_update)

        ####################

        # The actual function body.

        for _ in range(self._n_span_prop):
            pairwise_embeddings = self._trig_arg_embedder(
                trig_embeddings, arg_embeddings)
            argument_scores = self._argument_scorer(
                pairwise_embeddings, trig_scores, arg_scores, prepend_zeros=False)
            truncated_scores = F.relu(argument_scores)

            # The trigger update.
            # TODO(dwadden) Add normalization like Ulme?
            transition_trig = self._transition_trigger(truncated_scores)
            # Mask appropriately so that masked args won't contribute to update.
            updates_trig = self._compute_updates(
                transition_trig, arg_embeddings, arg_mask, [0, 3, 1, 2])

            if self._check:
                # Do a correctness spot-check of the updates by checking a single element.
                check_trigger()

            gate_trig = torch.sigmoid(
                self._gate_trigger(torch.cat([trig_embeddings, updates_trig], dim=-1)))
            trig_embeddings = gate_trig * trig_embeddings + (1 - gate_trig) * updates_trig

            # Now the arguments
            transition_arg = self._transition_argument(truncated_scores)
            updates_arg = self._compute_updates(
                transition_arg, trig_embeddings, trig_mask, [0, 3, 2, 1])

            if self._check:
                check_arg()

            gate_arg = torch.sigmoid(
                self._gate_argument(torch.cat([arg_embeddings, updates_arg], dim=-1)))
            arg_embeddings = gate_arg * arg_embeddings + (1 - gate_arg) * updates_arg

        return trig_embeddings, arg_embeddings

    @staticmethod
    def _compute_updates(transition, embeddings, mask, permute_order):
        # Use matrix mults to avoid repeats of matrices.
        permuted_transition = transition.permute(permute_order)
        permuted_embeddings = (embeddings * mask.float()).permute([0, 2, 1]).unsqueeze(-1)
        prod = torch.matmul(permuted_transition, permuted_embeddings).squeeze(-1)
        res = prod.permute([0, 2, 1])
        return res
