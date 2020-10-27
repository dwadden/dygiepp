local template = import "template.libsonnet";

template.DyGIE {
  bert_model: "roberta-base",
  cuda_device: 2,
  data_paths: {
    train: "data/sentivent/preproc_multi/train.jsonl",
    validation: "data/sentivent/preproc_multi/dev.jsonl",
    test: "data/sentivent/preproc_multi/test.jsonl",
  },
  loss_weights: {
    ner: 0.5,
    relation: 1.0,
    coref: 0.0,
    events: 1.0
  },
  target_task: "events",
  max_span_width: 8,
  max_trigger_span_width: 4,
}