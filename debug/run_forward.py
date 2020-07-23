import json
from _jsonnet import evaluate_file

from allennlp.data import token_indexers, vocabulary
from allennlp.modules import token_embedders, text_field_embedders

from dygie.data.dataset_readers.dygie import DyGIEReader
from dygie.data.iterators import batch_iterator
from dygie.models import dygie



token_indexers = {"tokens": token_indexers.SingleIdTokenIndexer()}
reader = DyGIEReader(max_span_width=8, token_indexers=token_indexers)
data = reader.read("data/scierc/processed_data/json/train-head.json")
vocab = vocabulary.Vocabulary.from_instances(data)
data.index_with(vocab)

token_embedder = token_embedders.Embedding(num_embeddings=vocab.get_vocab_size("tokens"), embedding_dim=100)
embedder = text_field_embedders.BasicTextFieldEmbedder({"tokens": token_embedder})

file_dict = json.loads(evaluate_file("training_config/debug.jsonnet"))
model_dict = file_dict["model"]
for name in ["type", "embedder", "initializer", "module_initializer"]:
    del model_dict[name]

iterator = batch_iterator.BatchIterator(batch_size=1, dataset=data)
for batch in iterator:
    break

model = dygie.DyGIE(vocab=vocab,
                    embedder=embedder,
                    **model_dict)


inst = data[0]
res = model(**batch)
