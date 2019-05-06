// Import template file.

local template = import "template.libsonnet";

////////////////////

// Set options.

local params = {
  // Primary prediction target. Watch metrics associated with this target.
  target: "rel",

  // If debugging, don't load expensive embedding files.
  debug: false,

  // Specifies the token-level features that will be created.
  use_glove: false,
  use_char: false,
  use_elmo: false,
  use_attentive_span_extractor: false,
  use_bert_base: true,
  use_bert_large: false,
  finetune_bert: true,
  rel_prop: 0,
  context_width: 3,
  rel_prop_dropout_A: 0.0,
  rel_prop_dropout_f: 0.0,

  // Specifies the model parameters.
  lstm_hidden_size: 200,
  lstm_n_layers: 1,
  feature_size: 20,
  feedforward_layers: 2,
  char_n_filters: 50,
  feedforward_dim: 150,
  max_span_width: 8,
  feedforward_dropout: 0.4,
  lexical_dropout: 0.5,
  lstm_dropout: 0.0,
  loss_weights: {          // Loss weights for the modules.
    ner: 1.0,
    relation: 1.0,
    coref: 0.0,
    events: 0.0
  },
  loss_weights_events: {   // Loss weights for trigger and argument ID in events.
    trigger: 1.0,
    arguments: 1.0,
  },

  // Coref settings.
  coref_spans_per_word: 0.4,
  coref_max_antecedents: 100,

  // Relation settings.
  relation_spans_per_word: 0.5,
  relation_positive_label_weight: 1.0,

  // Event settings.
  trigger_spans_per_word: 0.4,
  argument_spans_per_word: 0.8,
  events_positive_label_weight: 1.0,

  // Model training
  batch_size: 8,
  num_epochs: 250,
  patience: 15,
  optimizer: {
    type: "bert_adam",
    lr: 1e-3,
    warmup: 0.1,
    schedule: "warmup_linear",
    t_total: 200000,
    weight_decay: 0.0,
    parameter_groups: [
      [["_text_field_embedder"], {"lr": 1e-4, "warmup": 0.4, "schedule": "warmup_linear", "t_total": 200000, "weight_decay": 0.1}],
    ],
  },
  learning_rate_scheduler:  {
    type: "reduce_on_plateau",
    factor: 0.5,
    mode: "max",
    patience: 4
  }
};

////////////////////

// Feed options into template.

template(params)
