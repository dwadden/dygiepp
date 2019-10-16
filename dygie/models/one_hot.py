# Got this one from https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/33, under
# posting "A batch version that works".

import torch


def make_embedder(kind, num_embeddings, embedding_dim):
    assert kind in ["one_hot", "learned"]
    if kind == "one_hot":
        return OneHotEncoder(num_embeddings)
    else:
        return torch.nn.Embedding(num_embeddings=num_embeddings,
                                  embedding_dim=embedding_dim)


class OneHotEncoder(torch.nn.Module):
    """
    A one-hot encoder class. Only a module in the trivial sense.
    """
    def __init__(self, num_classes):
        super().__init__()
        self._num_classes = num_classes

    def forward(self, x):
        assert x.min().item() >= 0
        batch_size, seq_length = x.size()
        res = torch.zeros(batch_size, seq_length, self._num_classes, device=x.device)
        res.scatter_(2, x.unsqueeze(2), 1)
        return res
