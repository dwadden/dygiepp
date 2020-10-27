'''
Script for debugging the model without calling allennlp train/predict and thus avoiding costly init.
This script will run a forward pass without all the initialization logic and without loading pretrained embeddings.

Usage:
With `dygiepp` modeling env. active, run: `python debug_forward_pass.py <training_conf_filepath>`
See `python debug_forward_pass.py --help` for optional args.
'''

import argparse
import json
from _jsonnet import evaluate_file

from allennlp.data import token_indexers, vocabulary
from allennlp.modules import token_embedders, text_field_embedders
from allennlp.data.dataloader import PyTorchDataLoader

from dygie.data.dataset_readers.dygie import DyGIEReader
from dygie.models import dygie


def get_args():
    parser = argparse.ArgumentParser(description="Debug forward pass of model.")
    parser.add_argument("training_config", type=str, help="Path to the config file.")
    parser.add_argument("--use_bert", action="store_true",
                        help="If given, use the BERT model in the config. Else, use random embeddings.")
    parser.add_argument("--model_archive", type=str, default=None,
                        help="If given, load an archived model instaed of initializing from scratch.")
    parser.add_argument("--max_instances", type=int, default=5,
                        help="Maximum number of instances to load.")

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    # Read config.
    conf_dict = json.loads(evaluate_file(args.training_config))
    model_dict = conf_dict["model"]

    if args.use_bert:
        bert_name = model_dict["embedder"]["token_embedders"]["bert"]["model_name"]
        bert_max_length = model_dict["embedder"]["token_embedders"]["bert"]["max_length"]
    else:
        bert_name = None

    # Hack to replace components that we're setting in the script.
    for name in ["type", "embedder", "initializer", "module_initializer"]:
        del model_dict[name]

    # Create indexer.
    if args.use_bert:
        tok_indexers = {"bert": token_indexers.PretrainedTransformerMismatchedIndexer(
            bert_name, max_length=bert_max_length)}
    else:
        tok_indexers = {"tokens": token_indexers.SingleIdTokenIndexer()}

    # Read input data.
    reader_dict = conf_dict["dataset_reader"]
    reader = DyGIEReader(reader_dict["max_span_width"], max_trigger_span_width=reader_dict["max_trigger_span_width"], token_indexers=tok_indexers, max_instances=args.max_instances)
    data = reader.read(conf_dict["train_data_path"])
    vocab = vocabulary.Vocabulary.from_instances(data)
    data.index_with(vocab)

    # Create embedder.
    if args.use_bert:
        token_embedder = token_embedders.PretrainedTransformerMismatchedEmbedder(
            bert_name, max_length=bert_max_length)
        embedder = text_field_embedders.BasicTextFieldEmbedder({"bert": token_embedder})
    else:
        token_embedder = token_embedders.Embedding(
            num_embeddings=vocab.get_vocab_size("tokens"), embedding_dim=100)
        embedder = text_field_embedders.BasicTextFieldEmbedder({"tokens": token_embedder})

    # Create iterator and model.
    iterator = PyTorchDataLoader(batch_size=1, dataset=data)
    if args.model_archive is None:
        model = dygie.DyGIE(vocab=vocab,
                            embedder=embedder,
                            **model_dict)
    else:
        model = dygie.DyGIE.from_archive(args.model_archive)

    # Run forward pass over a single entry.
    for batch in iterator:
        output_dict = model(**batch)
        print(output_dict)


if __name__ == "__main__":
    main()
