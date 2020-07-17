local template = import "template.libsonnet";

template.DyGIE {
  bert_model: "allenai/scibert_scivocab_cased",
  cuda_device: 0,
  data_paths: {
    train: "data/scierc/processed_data/json/train.json",
    validation: "data/scierc/processed_data/json/dev.json",
    test: "data/scierc/processed_data/json/test.json",
  },
  loss_weights: {
    ner: 1.0,
    relation: 1.0,
    coref: 0.0,
    events: 0.0
  },
}
