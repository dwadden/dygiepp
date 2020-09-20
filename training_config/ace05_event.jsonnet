local template = import "template.libsonnet";

template.DyGIE {
  bert_model: "roberta-base",
  cuda_device: 3,
  data_paths: {
    train: "data/ace-event/collated-data/default-settings/json/train.json",
    validation: "data/ace-event/collated-data/default-settings/json/dev.json",
    test: "data/ace-event/collated-data/default-settings/json/test.json",
  },
  loss_weights: {
    ner: 0.5,
    relation: 0.5,
    coref: 0.0,
    events: 1.0
  },
  target_task: "events"
}
