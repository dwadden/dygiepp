// Import template file.

local template = import "data/dwadden/proj/dygie/dygie/training_config/template_dw.libsonnet";

////////////////////

// Set options.

local params = {
  // Primary prediction target. Watch metrics associated with this target.
  target: "events",

  // If debugging, don't load expensive embedding files.
  debug: false,

  // Specifies the token-level features that will be created.
  use_glove: true,
  use_char: true,
  use_elmo: true,
  use_attentive_span_extractor: false,

  // Specifies the model parameters.
  lstm_hidden_size: 300,
  lstm_n_layers: 1,
  feature_size: 20,
  feedforward_layers: 2,
  char_n_filters: 50,
  feedforward_dim: 400,
  max_span_width: 10,
  feedforward_dropout: 0.4,
  lexical_dropout: 0.5,
  lstm_dropout: 0.4,
  loss_weights: {          // Loss weights for the modules.
    ner: 1.0,
    relation: 0.0,
    coref: 0.0,
    events: 1.0
  },
  loss_weights_events: {   // Loss weights for trigger and argument ID in events.
    trigger: 1.0,
    arguments: 1.0,
  },

  // Coref settings.
  coref_spans_per_word: 0.4,
  coref_max_antecedents: 100,

  // Relation settings.
  relation_spans_per_word: 0.4,
  relation_positive_label_weight: 1.0,

  // Event settings.
  trigger_spans_per_word: 0.4,
  argument_spans_per_word: 0.8,
  events_positive_label_weight: 1.0,

  // Model training
  batch_size: 10,
  num_epochs: 250,
  patience: 15,
  // optimizer: {
  //   type: "adam",
  //   lr: 0.001
  // },
  optimizer: {
    type: "sgd",
    lr: 0.01,
    momentum: 0.9,
    nesterov: true
  },
  // learning_rate_scheduler: {
  //   type: "exponential",
  //   gamma: 0.997
  // }
  learning_rate_scheduler:  {
    type: "reduce_on_plateau",
    factor: 0.5,
    mode: "max",
    patience: 3
  }
};

////////////////////

// Feed options into template.

template(params)
