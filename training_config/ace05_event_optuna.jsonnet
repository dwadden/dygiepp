local template = import "template.libsonnet";

template.DyGIE {
  bert_model: "roberta-base",
  cuda_device: 3,
  max_span_width: std.parseInt(std.extVar('max_span_width')),
  data_paths: {
    train: "data/ace-event/collated-data/default-settings/json/train.json",
    validation: "data/ace-event/collated-data/default-settings/json/dev.json",
    test: "data/ace-event/collated-data/default-settings/json/test.json",
  },
  loss_weights: {
    ner: std.parseJson(std.extVar('lossw_ner')),
    relation: std.parseJson(std.extVar('lossw_relation')),
    events: std.parseJson(std.extVar('lossw_events')),
    coref: 0.0,
  },
  target_task: "events",
  model +: {
    feature_size:  std.parseInt(std.extVar("feature_size")),
    feedforward_params +: {
      num_layers: std.parseInt(std.extVar("ffwd_num_layers")),
      hidden_dims: std.parseInt(std.extVar("ffwd_hidden_dims")),
      dropout: std.parseJson(std.extVar("ffwd_dropout")),
    },
    modules +: {
      relation +: {
        spans_per_word: std.parseJson(std.extVar("relation_spans_per_word")),
      },
      events +: {
        trigger_spans_per_word: std.parseJson(std.extVar("events_trigger_spans_per_word")),
        argument_spans_per_word: std.parseJson(std.extVar("events_argument_spans_per_word")),
        loss_weights +: {
          trigger: std.parseJson(std.extVar("events_lossw_trigger")),
          arguments: std.parseJson(std.extVar("events_lossw_arguments")),
        },
      },
    },
  },
}