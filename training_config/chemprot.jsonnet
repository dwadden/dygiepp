local template = import "template.libsonnet";

template.DyGIE {
  bert_model: "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
  cuda_device: 2,
  data_paths: {
    train: "data/chemprot/collated_data/training.jsonl",
    validation: "data/chemprot/collated_data/development.jsonl",
    test: "data/chemprot/collated_data/test.jsonl",
  },
  loss_weights: {
    ner: 0.2,
    relation: 1.0,
    coref: 0.0,
    events: 0.0
  },
  target_task: "relation",
  trainer +: {
    num_epochs: 25
  },
}
