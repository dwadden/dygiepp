local template = import "template.libsonnet";

template.DyGIE {
  bert_model: "roberta-base",
  cuda_device: 1,
  data_paths: {
    train: "data/ace05/collated-data/json/train.json",
    validation: "data/ace05/collated-data/json/dev.json",
    test: "data/ace05/collated-data/json/test.json",
  },
  loss_weights: {
    ner: 0.2,
    relation: 1.0,
    coref: 0.0,
    events: 0.0
  },
  target_task: "relation"
}
