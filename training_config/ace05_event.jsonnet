local template = import "template.libsonnet";

template.DyGIE {
  bert_model: "bert-base-cased",
  cuda_device: 2,
  data_paths: {
    train: "data/ace-event/collated-data/default-settings/json/train.json",
    validation: "data/ace-event/collated-data/default-settings/json/dev.json",
    test: "data/ace-event/collated-data/default-settings/json/test.json",
  },
  loss_weights: {
    ner: 0.5,
    relation: 1.0,
    coref: 0.0,
    events: 1.0
  },
  target_task: "events",
}
