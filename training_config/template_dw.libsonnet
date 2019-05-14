// Library that accepts a parameter dict and returns a full config.

function(p) {
  // Location of ACE valid event configs
  local valid_events_dir = std.extVar("valid_events_dir"),

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
    "events": ["trig_class_f1", "arg_class_f1"]
  },

  local glove_dim = 300,
  local elmo_dim = 1024,
  local bert_base_dim = 768,
  local bert_large_dim = 1024,

  local module_initializer = [
    [".*weight", {"type": "xavier_normal"}],
    [".*weight_matrix", {"type": "xavier_normal"}]],

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
  local use_bert = (if p.use_bert_base then true else if p.use_bert_large then true else false),

  // If true, use ner and trigger labels as features to predict event arguments.
  // TODO(dwadden) At some point I should make arguments like this mandatory.
  local event_args_use_ner_labels = getattr(p, "event_args_use_ner_labels", false),
  local event_args_use_trigger_labels = getattr(p, "event_args_use_trigger_labels", false),
  // Embedding dim for labels used as features for argument prediction.
  local event_args_label_emb = getattr(p, "event_args_label_emb", 10),
  // If predicting labels, can either do "hard" prediction or "softmax". Default is hard.
  local event_args_label_predictor = getattr(p, "event_args_label_predictor", "hard"),
  local events_context_window = getattr(p, "events_context_window", 0),
  local shared_attention_context = getattr(p, "shared_attention_context", false),
  local trigger_attention_context = getattr(p, "trigger_attention_context", false),

  local token_embedding_dim = ((if p.use_glove then glove_dim else 0) +
    (if p.use_char then p.char_n_filters else 0) +
    (if p.use_elmo then elmo_dim else 0) +
    (if p.use_bert_base then bert_base_dim else 0) +
    (if p.use_bert_large then bert_large_dim else 0)),
  // If we're using Bert, no LSTM. We just pass the token embeddings right through.
  local context_layer_output_size = (if p.finetune_bert
    then token_embedding_dim
    else 2 * p.lstm_hidden_size),
  local endpoint_span_emb_dim = 2 * context_layer_output_size + p.feature_size,
  local attended_span_emb_dim = if p.use_attentive_span_extractor then token_embedding_dim else 0,
  local span_emb_dim = endpoint_span_emb_dim + attended_span_emb_dim,
  local pair_emb_dim = 3 * span_emb_dim,
  local relation_scorer_dim = pair_emb_dim,
  local coref_scorer_dim = pair_emb_dim + p.feature_size,
  local trigger_emb_dim = context_layer_output_size,  // Triggers are single contextualized tokens.
  // Add token embedding dim because we're including the cls token.
  local class_projection_dim = 200,
  local trigger_scorer_dim = ((if trigger_attention_context then 2 * trigger_emb_dim else trigger_emb_dim) +
    class_projection_dim),

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
  // The extra 1 is for the bit indicating whether the trigger is before or inside the argument.
  local argument_pair_dim = (trigger_emb_dim + span_emb_dim + p.feature_size + 2 +
    (if event_args_use_ner_labels then ner_label_dim else 0) +
    (if event_args_use_trigger_labels then trigger_label_dim else 0) +
    (if events_context_window > 0 then 4 * events_context_window * context_layer_output_size else 0)),
  // Add token embedding dim because of the cls token.
  local argument_scorer_dim = (argument_pair_dim +
    (if shared_attention_context then trigger_emb_dim else 0) +
    class_projection_dim),

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
    [if use_bert then "bert"]: {
      type: "bert-pretrained",
      pretrained_model: (if p.use_bert_base then "bert-base-cased" else "bert-large-cased"),
      do_lowercase: false,
      use_starting_offsets: true
    }
  },

  local text_field_embedder = {
    [if use_bert then "allow_unmatched_keys"]: true,
    [if use_bert then "embedder_to_indexer_map"]: {
      bert: ["bert", "bert-offsets"],
      token_characters: ["token_characters"]
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
      [if use_bert then "bert"]: {
        type: "bert-pretrained",
        pretrained_model: (if p.use_bert_base then "bert-base-cased" else "bert-large-cased"),
        requires_grad: p.finetune_bert
      }
    }
  },

  ////////////////////////////////////////////////////////////////////////////////

  // Modules

  // If finetuning Bert, no LSTM. Just pass through.
  local context_layer = (if p.finetune_bert
    then {
      type: "pass_through",
      input_dim: token_embedding_dim
    }
    else {
      type: "stacked_bidirectional_lstm",
      input_size: token_embedding_dim,
      hidden_size: p.lstm_hidden_size,
      num_layers: p.lstm_n_layers,
      recurrent_dropout_probability: p.lstm_dropout,
      layer_dropout_probability: p.lstm_dropout
    }
  ),


  ////////////////////////////////////////////////////////////////////////////////


  // The model

  dataset_reader: {
    type: "ie_json",
    token_indexers: token_indexers,
    max_span_width: p.max_span_width,
    context_width: p.context_width
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
    lstm_dropout: (if p.finetune_bert then 0 else p.lstm_dropout),
    rel_prop: p.rel_prop,
    feature_size: p.feature_size,
    use_attentive_span_extractor: p.use_attentive_span_extractor,
    max_span_width: p.max_span_width,
    display_metrics: display_metrics[p.target],
    valid_events_dir: valid_events_dir,
    check: getattr(p, "check", false), // If true, run a bunch of correctness assertions in the code.
    context_layer: context_layer,
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
        rel_prop_dropout_A: p.rel_prop_dropout_A,
        rel_prop_dropout_f: p.rel_prop_dropout_f,
        rel_prop: p.rel_prop,
        span_emb_dim: span_emb_dim,
        initializer: module_initializer
      },
      events: {
        trigger_spans_per_word: p.trigger_spans_per_word,
        argument_spans_per_word: p.argument_spans_per_word,
        positive_label_weight: p.events_positive_label_weight,
        trigger_feedforward: make_feedforward(trigger_scorer_dim), // Factor of 2 because of self attention.
        trigger_candidate_feedforward: make_feedforward(trigger_emb_dim),
        mention_feedforward: make_feedforward(span_emb_dim),
        argument_feedforward: make_feedforward(argument_scorer_dim),
        event_args_use_trigger_labels: event_args_use_trigger_labels,
        event_args_use_ner_labels: event_args_use_ner_labels,
        event_args_label_predictor: event_args_label_predictor,
        event_args_label_emb: event_args_label_emb,
        label_embedding_method: label_embedding_method,
        event_args_gold_candidates: getattr(p, "event_args_gold_candidates", false),
        initializer: module_initializer,
        loss_weights: p.loss_weights_events,
        entity_beam: getattr(p, "events_entity_beam", false),
        context_window: events_context_window,
        shared_attention_context: shared_attention_context,
        cls_projection: {
          input_dim: token_embedding_dim,
          num_layers: 1,
          hidden_dims: class_projection_dim,
          activations: "relu",
          dropout: p.feedforward_dropout
        },
        context_attention: {
          matrix_1_dim: argument_pair_dim,
          matrix_2_dim: trigger_emb_dim,
        },
        trigger_attention_context: trigger_attention_context,
        trigger_attention: {
          type: "multi_head_self_attention",
          num_heads: 8,
          input_dim: trigger_emb_dim,
          attention_dim: trigger_emb_dim,
          values_dim: trigger_emb_dim,
          attention_dropout_prob: 0.4
      },
      }
    }
  },
  iterator: {
    // type: "ie_batch",
    // batch_size: p.batch_size
    type: "bucket",
    sorting_keys: [["text", "num_tokens"]],
    batch_size : p.batch_size,
    [if "instances_per_epoch" in p then "instances_per_epoch"]: p.instances_per_epoch
  },
  // validation_iterator: {
  //   type: "ie_document",
  // },
  trainer: {
    num_epochs: p.num_epochs,
    grad_norm: 5.0,
    patience : p.patience,
    cuda_device : [std.parseInt(x) for x in std.split(std.extVar("cuda_device"), ",")],
    validation_metric: validation_metrics[p.target],
    learning_rate_scheduler: p.learning_rate_scheduler,
    optimizer: p.optimizer
  }
}
