// Library that accepts a parameter dict and returns a full config.

function(p) {
  // Location of ACE valid event configs
  local valid_events_dir = "/homes/gws/dwadden/proj/dygie/dygie-experiments/datasets/ace-event/valid-configurations",

  // Storing constants.

  local validation_metrics = {
    "ner": "+ner_f1",
    "rel": "+rel_f1",
    "coref": "+coref_f1",
    "events": "+arg_class_f1"
  },

  local display_metrics = {
    "ner": ["ner_precision", "ner_recall", "ner_f1"],
    "rel": ["rel_precision", "rel_recall", "rel_f1", "rel_span_recall"],
    "coref": ["coref_precision", "coref_recall", "coref_f1", "coref_mention_recall"],
    "events": ["trig_class_f1", "arg_class_f1", "frac_trig_arg", "frac_ner_arg"]
  },

  local glove_dim = 300,
  local elmo_dim = 1024,
  local bert_dim = 768,

  local module_initializer = [
    [".*linear_layers.*weight", {"type": "xavier_normal"}],
    [".*scorer._module.weight", {"type": "xavier_normal"}],
    ["_distance_embedding.weight", {"type": "xavier_normal"}]],

  local dygie_initializer = [
    ["_span_width_embedding.weight", {"type": "xavier_normal"}],
    ["_context_layer._module.weight_ih.*", {"type": "xavier_normal"}],
    ["_context_layer._module.weight_hh.*", {"type": "orthogonal"}]
  ],


  ////////////////////////////////////////////////////////////////////////////////

  // Helper function.
  // Get the attribute from the object. If the object doesn't have that attribute, return default.
  local getattr(obj, attrname, default) = if attrname in obj then p[attrname] else default,

  // Calculating dimensions.

  // If true, use ner and trigger labels as features to predict event arguments.
  // TODO(dwadden) At some point I should make arguments like this mandatory.
  local event_args_use_ner_labels = getattr(p, "event_args_use_ner_labels", false),
  local event_args_use_trigger_labels = getattr(p, "event_args_use_trigger_labels", false),
  // Embedding dim for labels used as features for argument prediction.
  local event_args_label_emb = getattr(p, "event_args_label_emb", 10),
  // If predicting labels, can either do "hard" prediction or "softmax". Default is hard.
  local event_args_label_predictor = getattr(p, "event_args_label_predictor", "hard"),
  local events_context_window = getattr(p, "events_context_window", 0),

  local token_embedding_dim = ((if p.use_glove then glove_dim else 0) +
    (if p.use_char then p.char_n_filters else 0) +
    (if p.use_elmo then elmo_dim else 0) +
    (if p.use_bert then bert_dim else 0)),
  local endpoint_span_emb_dim = 4 * p.lstm_hidden_size + p.feature_size,
  local attended_span_emb_dim = if p.use_attentive_span_extractor then token_embedding_dim else 0,
  local span_emb_dim = endpoint_span_emb_dim + attended_span_emb_dim,
  local pair_emb_dim = 3 * span_emb_dim,
  local relation_scorer_dim = pair_emb_dim,
  local coref_scorer_dim = pair_emb_dim + p.feature_size,
  local trigger_scorer_dim = 2 * p.lstm_hidden_size,  // Triggers are single contextualized tokens.

  // Calculation of argument scorer dim is a bit tricky. First, there's the triggers and the span
  // embeddings. Then, if we're using labels, include those. Then, if we're using a context window,
  // there are 2 x context_window extra entries for both arg and trigger, which makes 4 x total. The
  // dimension of each entry is twice the lstm hidden size.
  // For the labels, we look at ner and trigger separately, and we can either do one-hot encodings
  // or learned embeddings.
  // Allowed values are `one_hot` and `learned`.
  local label_embedding_method = getattr(p, "label_embedding_method", "one_hot"),
  local trigger_label_dim = (if event_args_use_trigger_labels
    then (if label_embedding_method == "one_hot" then p.n_trigger_labels else event_args_label_emb)
    else 0),
  local ner_label_dim = (if event_args_use_ner_labels
    then (if label_embedding_method == "one_hot" then p.n_ner_labels else event_args_label_emb)
    else 0),
  local argument_scorer_dim = (trigger_scorer_dim + span_emb_dim +
    (if event_args_use_ner_labels then ner_label_dim else 0) +
    (if event_args_use_trigger_labels then trigger_label_dim else 0) +
    (if events_context_window > 0 then 8 * events_context_window * p.lstm_hidden_size else 0)),

  ////////////////////////////////////////////////////////////////////////////////

  // Function definitions

  local make_feedforward(input_dim) = {
    input_dim: input_dim,
    num_layers: p.feedforward_layers,
    hidden_dims: p.feedforward_dim,
    activations: "relu",
    dropout: p.feedforward_dropout
  },

  // Model components

  local token_indexers = {
    [if p.use_glove then "tokens"]: {
      type: "single_id",
      lowercase_tokens: false
    },
    [if p.use_char then "token_characters"]: {
      type: "characters",
      min_padding_length: 5
    },
    [if p.use_elmo then "elmo"]: {
      type: "elmo_characters"
    },
    [if p.use_bert then "bert"]: {
      type: "bert-pretrained",
      pretrained_model: "bert-base-cased",
      do_lowercase: false,
      use_starting_offsets: true
    }
  },

  local text_field_embedder = {
    [if p.use_bert then "allow_unmatched_keys"]: true,
    [if p.use_bert then "embedder_to_indexer_map"]: {
      bert: ["bert", "bert-offsets"],
      token_characters: ["token_characters"],
    },
    token_embedders: {
      [if p.use_glove then "tokens"]: {
        type: "embedding",
        pretrained_file: if p.debug then null else "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.840B.300d.txt.gz",
        embedding_dim: 300,
        trainable: false
      },
      [if p.use_char then "token_characters"]: {
        type: "character_encoding",
        embedding: {
          num_embeddings: 262,
          embedding_dim: 16
        },
        encoder: {
          type: "cnn",
          embedding_dim: 16,
          num_filters: p.char_n_filters,
          ngram_filter_sizes: [5]
        }
      },
      [if p.use_elmo then "elmo"]: {
        type: "elmo_token_embedder",
        options_file: "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
        weight_file: "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
        do_layer_norm: false,
        dropout: 0.5
      },
      [if p.use_bert then "bert"]: {
        type: "bert-pretrained",
        pretrained_model: "bert-base-cased"
      }
    }
  },

  ////////////////////////////////////////////////////////////////////////////////

  // The model

  dataset_reader: {
    type: "ie_json",
    token_indexers: token_indexers,
    max_span_width: p.max_span_width
  },
  train_data_path: std.extVar("ie_train_data_path"),
  validation_data_path: std.extVar("ie_dev_data_path"),
  test_data_path: std.extVar("ie_test_data_path"),
  // If provided, use pre-defined vocabulary. Else compute on the fly.
  [if "vocab_path" in p then "vocabulary"]: {
    directory_path: p.vocab_path
  },
  model: {
    type: "dygie",
    text_field_embedder: text_field_embedder,
    initializer: dygie_initializer,
    loss_weights: p.loss_weights,
    lexical_dropout: p.lexical_dropout,
    lstm_dropout: p.lstm_dropout,
    feature_size: p.feature_size,
    use_attentive_span_extractor: p.use_attentive_span_extractor,
    max_span_width: p.max_span_width,
    display_metrics: display_metrics[p.target],
    valid_events_dir: valid_events_dir,
    check: getattr(p, "check", false), // If true, run a bunch of correctness assertions in the code.
    context_layer: {
      type: "lstm",
      bidirectional: true,
      input_size: token_embedding_dim,
      hidden_size: p.lstm_hidden_size,
      num_layers: p.lstm_n_layers,
      dropout: p.lstm_dropout
    },
    modules: {
      coref: {
        spans_per_word: p.coref_spans_per_word,
        max_antecedents: p.coref_max_antecedents,
        mention_feedforward: make_feedforward(span_emb_dim),
        antecedent_feedforward: make_feedforward(coref_scorer_dim),
        initializer: module_initializer
      },
      ner: {
        mention_feedforward: make_feedforward(span_emb_dim),
        initializer: module_initializer
      },
      relation: {
        spans_per_word: p.relation_spans_per_word,
        positive_label_weight: p.relation_positive_label_weight,
        mention_feedforward: make_feedforward(span_emb_dim),
        relation_feedforward: make_feedforward(relation_scorer_dim),
        initializer: module_initializer
      },
      events: {
        trigger_spans_per_word: p.trigger_spans_per_word,
        argument_spans_per_word: p.argument_spans_per_word,
        positive_label_weight: p.events_positive_label_weight,
        trigger_feedforward: make_feedforward(trigger_scorer_dim),
        trigger_candidate_feedforward: make_feedforward(trigger_scorer_dim),
        mention_feedforward: make_feedforward(span_emb_dim),
        argument_feedforward: make_feedforward(argument_scorer_dim),
        event_args_use_trigger_labels: event_args_use_trigger_labels,
        event_args_use_ner_labels: event_args_use_ner_labels,
        event_args_label_predictor: event_args_label_predictor,
        event_args_label_emb: event_args_label_emb,
        label_embedding_method: label_embedding_method,
        event_args_gold: getattr(p, "event_args_gold", false),
        initializer: module_initializer,
        loss_weights: p.loss_weights_events,
        entity_beam: getattr(p, "events_entity_beam", false),
        context_window: events_context_window
      }
    }
  },
  iterator: {
    // type: "ie_batch",
    // batch_size: p.batch_size
    "type": "bucket",
    "sorting_keys": [["text", "num_tokens"]],
    "batch_size" : p.batch_size
  },
  // validation_iterator: {
  //   type: "ie_document",
  // },
  trainer: {
    num_epochs: p.num_epochs,
    grad_norm: 5.0,
    patience : p.patience,
    cuda_device : std.parseInt(std.extVar("cuda_device")),
    validation_metric: validation_metrics[p.target],
    learning_rate_scheduler: p.learning_rate_scheduler,
    optimizer: p.optimizer
  }
}
