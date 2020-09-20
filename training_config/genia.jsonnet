local template = import "template.libsonnet";

template.DyGIE {
  bert_model: "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
  cuda_device: 0,
  data_paths: {
    train: "data/genia/normalized-data/json-coref-ident-only/train.json",
    validation: "data/genia/normalized-data/json-coref-ident-only/dev.json",
    test: "data/genia/normalized-data/json-coref-ident-only/test.json",
  },
  loss_weights: {
    ner: 1.0,
    relation: 0.0,
    coref: 1.0,
    events: 0.0
  },
  model +: {
    modules +: {
      coref +: {
        coref_prop: 2
      }
    }
  },
  target_task: "ner",
  trainer +: {
    num_epochs: 15   # It's a fairly big dataset; we don't need 50 epochs.
  },
}
