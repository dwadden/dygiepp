local template = import "template.libsonnet";

template.DyGIE {
  bert_model: "allenai/scibert_scivocab_cased",
  cuda_device: 0,
  data_paths: {
    train: "data/genia/normalized-data/json-ner/train.json",
    validation: "data/genia/normalized-data/json-ner/dev.json",
    test: "data/genia/normalized-data/json-ner/test.json",
  },
  loss_weights: {
    ner: 1.0,
    relation: 0.0,
    coref: 0.0,
    events: 0.0
  },
  target_task: "ner"
}
