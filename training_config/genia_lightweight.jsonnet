local template = import "template.libsonnet";

template.DyGIE {
  bert_model: "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
  cuda_device: 1,
  data_paths: {
    train: "data/genia/collated-data/json-ner/train.json",
    validation: "data/genia/collated-data/json-ner/dev.json",
    test: "data/genia/collated-data/json-ner/test.json",
  },
  loss_weights: {
    ner: 1.0,
    relation: 0.0,
    coref: 0.0,
    events: 0.0
  },
  target_task: "ner",
  trainer +: {
    num_epochs: 15   # It's a fairly big dataset; we don't need 50 epochs.
  },
}
