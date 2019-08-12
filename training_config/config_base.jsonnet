// Import template file.

local template = import "/data/dwadden/proj/dygie/dygie/training_config/template_dw.libsonnet";

////////////////////

// Set options.

local params = {
  // Primary prediction target. Watch metrics associated with this target.
  target: "events",

  random_seed: 76,
  numpy_seed: 1776,
  pytorch_seed: 76766,

  // If debugging, don't load expensive embedding files.
  debug: false,

  // Specifies the token-level features that will be created.
  use_glove: false,
  use_char: false,
  use_elmo: false,
  use_attentive_span_extractor: false,
  use_bert_base: false,
  use_bert_large: true,
  finetune_bert: false,
  context_width: 3,

  // Specifies the model parameters.
  lstm_hidden_size: 400,
  lstm_n_layers: 1,
  feature_size: 20,
  feedforward_layers: 2,
  char_n_filters: 50,
  feedforward_dim: 600,
  max_span_width: 12,
  feedforward_dropout: 0.4,
  lexical_dropout: 0.5,
  lstm_dropout: 0.4,
  loss_weights: {          // Loss weights for the modules.
    ner: 0.5,
    relation: 0.0,
    coref: 0.0,
    events: 1.0
  },
  loss_weights_events: {   // Loss weights for trigger and argument ID in events.
    trigger: 0.2,
    arguments: 1.0,
  },

  // Coref settings.
  coref_spans_per_word: 0.4,
  coref_max_antecedents: 100,
  coref_prop: 0,
  co_train: false,

  // Relation settings.
  relation_spans_per_word: 0.4,
  relation_positive_label_weight: 1.0,
  rel_prop: 0,
  rel_prop_dropout_A: 0.0,
  rel_prop_dropout_f: 0.0,

  // Event settings.
  trigger_spans_per_word: 0.4,
  argument_spans_per_word: 0.8,
  events_positive_label_weight: 1.0,
  events_entity_beam: false,             // Use entity beam.
  event_args_use_ner_labels: true,      // Use ner labels when predicting roles.
  event_args_use_trigger_labels: false,  // Use trigger labels when predicting roles.
  label_embedding_method: "one_hot",     // Label embedding method.
  event_args_label_predictor: "softmax", // Method for predicting labels at test time.
  event_args_label_emb: 10,          // Label embedding dimension.
  event_args_gold_candidates: false, // If true, use gold candidate spans.
  n_trigger_labels: 34,        // Need # of trigger and ner labels in order to add as features.
  n_ner_labels: 8,
  events_context_window: 0,
  shared_attention_context: false,
  trigger_attention_context: false,
  event_n_span_prop: 0,

  // Model training
  batch_size: 15,
  num_epochs: 250,
  patience: 15,
  optimizer: {
    type: "sgd",
    lr: 0.02,
    momentum: 0.9,
    nesterov: true,
    weight_decay: 1e-6,
  },
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
