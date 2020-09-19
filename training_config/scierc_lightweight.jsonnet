local template = import "template.libsonnet";

template.DyGIE {
  bert_model: "allenai/scibert_scivocab_cased",
  cuda_device: 1,
  data_paths: {
    train: "data/scierc/collated_data/json/train.json",
    validation: "data/scierc/collated_data/json/dev.json",
    test: "data/scierc/collated_data/json/test.json",
  },
  loss_weights: {
    ner: 0.2,
    relation: 1.0,
    coref: 0.0,
    events: 0.0
  },
  target_task: "relation",
}
