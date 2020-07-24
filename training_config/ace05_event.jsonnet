local template = import "template.libsonnet";

template.DyGIE {
  bert_model: "bert-base-cased",
  cuda_device: 1,
  data_paths: {
    train: "data/ace-event/normalized-data/default-settings/json/train.json",
    validation: "data/ace-event/normalized-data/default-settings/json/dev.json",
    test: "data/ace-event/normalized-data/default-settings/json/test.json",
  },
  loss_weights: {
    ner: 0.5,
    relation: 0.0,
    coref: 0.0,
    events: 1.0
  },
  target_task: "events",
}
