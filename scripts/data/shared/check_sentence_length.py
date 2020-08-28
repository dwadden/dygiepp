import argparse

from transformers import AutoConfig
from allennlp.data import vocabulary, token_indexers

from dygie.data.dataset_readers.dygie import DyGIEReader


def get_args():
    parser = argparse.ArgumentParser(
        description="Check for sentences that are longer than the length limit of the encoder.")
    parser.add_argument("input_file", type=str,
                        help="The dataset to check.")
    parser.add_argument("--model_name", type=str, default="bert-base-cased",
                        help="The BERT model to be used.")
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    indexers = {"bert": token_indexers.PretrainedTransformerMismatchedIndexer(args.model_name)}
    config = AutoConfig.from_pretrained(args.model_name)
    max_length = config.max_position_embeddings
    reader = DyGIEReader(max_span_width=8, token_indexers=indexers)
    data = reader.read(args.input_file)
    vocab = vocabulary.Vocabulary.from_instances(data)
    print(f"The following documents have sentences over {max_length} tokens:")
    for instance in data:
        instance.index_fields(vocab)
        td = instance.as_tensor_dict()
        n_wordpieces = td["text"]["bert"]["wordpiece_mask"].sum(dim=1)
        too_long = (n_wordpieces > max_length).nonzero(as_tuple=False).squeeze(-1).tolist()
        lengths = n_wordpieces[too_long]

        if too_long:
            msg_start = f"Document {td['metadata'].doc_key}: "
            msg_body = [f"sentence {sentence_ix} ({sentence_length} tokens)"
                        for sentence_ix, sentence_length in zip(too_long, lengths)]
            msg_body = ", ".join(msg_body)
            msg = msg_start + msg_body + "."
            print(msg)


if __name__ == "__main__":
    main()
