from typing import Any, Dict, List, Optional

import torch
from torch.nn import functional as F

from torch_geometric.nn import GATConv
from torch_geometric.data import Data

# In the Bengio paper, for the PPI data set (which is big like ours) they use 3 layers, 4 heads of
# 256 dims each, no dropout or regularization.

class GraphAttention(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=4, dropout=0.2):
        # The dimension of each head is the output dimension divided by the number of heads.
        # Not gonna bother with all the AllenNLP over head at this point, not worth the trouble.
        super(GraphAttention, self).__init__()

        assert output_dim % num_heads == 0
        head_dim = int(output_dim / num_heads)
        self._input_layer = GATConv(in_channels=input_dim, out_channels=head_dim,
                                    heads=num_heads, concat=True, dropout=dropout)
        self._output_layer = GATConv(in_channels=head_dim, out_channels=head_dim,
                                     heads=num_heads, concat=True, dropout=dropout)
        self._init_params()

    def _init_params(self):
        for layer in [self._input_layer, self._output_layer]:
            for param in layer.parameters():
                if param.ndimension() > 1:
                    torch.nn.init.xavier_normal_(param)

    def forward(self, pair_embeddings, num_trigs_kept, num_arg_spans_kept):
        # Get the data ready
        batch_size = pair_embeddings.size(0)
        emb_slice = pair_embeddings[0, :num_trigs_kept[0], :num_arg_spans_kept[0]]

        row_ix, _ = torch.sort(torch.arange(num_trigs_kept[0]).repeat(num_arg_spans_kept[0]))
        col_ix = torch.arange(num_arg_spans_kept[0]).repeat(num_trigs_kept[0])
        adj = (row_ix == row_ix.unsqueeze(1)) | (col_ix == col_ix.unsqueeze(1))
        emb_flat = emb_slice.contiguous().view(-1, emb_slice.size(-1))
        edge_index = adj.nonzero().t().cuda(emb_flat.device)  # This is how the model needs it.

        import ipdb; ipdb.set_trace()

        x = self._input_layer(emb_flat, edge_index)
        x = F.elu(x)
        x = self._output_layer(x, edge_index)
        return F.elu(x)
