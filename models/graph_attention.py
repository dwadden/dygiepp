from typing import Optional

from torch.nn import functional as F
import torch

from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.data import Vocabulary

from torch_geometric.nn import GATConv


# In the Bengio paper, for the PPI data set (which is big like ours) they use 3 layers, 4 heads of
# 256 dims each, no dropout or regularization.


class GraphAttention(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 input_dim, output_dim, n_heads, n_layers, dropout,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None):
        # The dimension of each head is the output dimension divided by the number of heads.
        # Not gonna bother with all the AllenNLP over head at this point, not worth the trouble.
        super(GraphAttention, self).__init__(vocab, regularizer)

        assert output_dim % n_heads == 0
        head_dim = int(output_dim / n_heads)
        layers = []
        input_layer = GATConv(in_channels=input_dim, out_channels=head_dim,
                              heads=n_heads, concat=True, dropout=dropout)
        layers.append(input_layer)
        for _ in range(1, n_layers):
            this_layer = GATConv(in_channels=head_dim * n_heads, out_channels=head_dim,
                                 heads=n_heads, concat=True, dropout=dropout)
            layers.append(this_layer)

        self._layers = torch.nn.ModuleList(layers)

        initializer(self)

    def forward(self, pair_embeddings, num_trigs_kept, num_arg_spans_kept):
        # Get the data ready
        batch_size = pair_embeddings.size(0)
        res = []

        # Do the graph attentions one batch at a time. When finished, concatenate back together.
        for i in range(batch_size):
            n_trigs = num_trigs_kept[i].item()
            n_args = num_arg_spans_kept[i].item()
            emb_slice = pair_embeddings[i, :n_trigs, :n_args]
            n_features = emb_slice.size(-1)

            row_ix, _ = torch.sort(torch.arange(n_trigs).repeat(n_args))
            col_ix = torch.arange(n_args).repeat(n_trigs)
            adj = (row_ix == row_ix.unsqueeze(1)) | (col_ix == col_ix.unsqueeze(1))
            emb_flat = emb_slice.contiguous().view(n_trigs * n_args, n_features)
            edge_index = adj.nonzero().t().cuda(emb_flat.device)  # This is how the model needs it.

            x = emb_flat
            for layer in self._layers:
                x = layer(x, edge_index)
                x = F.elu(x)

            out_features = x.size(-1)

            # Fill in the relevant entries of the batched score matrix.
            fill = x.view(n_trigs, n_args, out_features)
            scores = torch.zeros([num_trigs_kept.max(), num_arg_spans_kept.max(), x.size(-1)],
                                 device=pair_embeddings.device)
            scores[:n_trigs, :n_args] = fill
            res.append(scores.unsqueeze(0))

        return torch.cat(res, 0)
