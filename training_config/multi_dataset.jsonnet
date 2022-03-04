local template = import "template.libsonnet";

template.DyGIE {
  bert_model: "roberta-base",
  cuda_device: 0,
  data_paths: {
    train: "dygie/tests/fixtures/multi_dataset/train.jsonl",
    validation: "dygie/tests/fixtures/multi_dataset/dev.jsonl",
    test: "dygie/tests/fixtures/multi_dataset/test.jsonl",
  },
  loss_weights: {
    ner: 0.2,
    relation: 1.0,
    coref: 1.0,
    events: 1.0
  },
  target_task: "relation",
  trainer +: {
    num_epochs: 1
  },
}
