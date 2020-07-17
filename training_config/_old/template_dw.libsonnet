// Library that accepts a parameter dict and returns a full config.

function(p) {
  local getattr(obj, attrname, default) = if attrname in obj then p[attrname] else default,

  // Storing constants.

  local event_validation_metric = (if "event_validation_metric" in p
    then p.event_validation_metric
    else "+arg_class_f1"),

  local validation_metrics = {
    "ner": "+ner_f1",
    "rel": "+rel_f1",
    "coref": "+coref_f1",
    "events": event_validation_metric
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
  local scibert_dim = 768,

   local module_initializer = {"regexes":
    [[".*weight", {"type": "xavier_normal"}],
    [".*weight_matrix", {"type": "xavier_normal"}]]
  },

  local dygie_initializer = {"regexes":
    [["_span_width_embedding.weight", {"type": "xavier_normal"}],
    ["_context_layer._module.weight_ih.*", {"type": "xavier_normal"}],
    ["_context_layer._module.weight_hh.*", {"type": "orthogonal"}]]
  },



  ////////////////////////////////////////////////////////////////////////////////

  // Helper function.
  // Calculating dimensions.
  local use_bert = (if p.use_bert_base then true
                    else if p.use_bert_large then true
                    else if p.use_scibert then true
                    else false),

  local event_n_span_prop = getattr(p, "event_n_span_prop", 0),

  // If true, use ner and trigger labels as features to predict event arguments.
  // TODO(dwadden) At some point I should make arguments like this mandatory.
  local event_args_use_ner_labels = getattr(p, "event_args_use_ner_labels", false),
  local event_args_use_trigger_labels = getattr(p, "event_args_use_trigger_labels", false),
  // Embedding dim for labels used as features for argument prediction.
  local event_args_label_emb = getattr(p, "event_args_label_emb", 10),
  // If predicting labels, can either do "hard" prediction or "softmax". Default is hard.
  local event_args_label_predictor = getattr(p, "event_args_label_predictor", "hard"),
  local events_context_window = getattr(p, "events_context_window", 0),
  local trigger_attention_context = getattr(p, "trigger_attention_context", false),

  local token_embedding_dim = ((if p.use_glove then glove_dim else 0) +
    (if p.use_char then p.char_n_filters else 0) +
    (if p.use_elmo then elmo_dim else 0) +
    (if p.use_bert_base then bert_base_dim else 0) +
    (if p.use_bert_large then bert_large_dim else 0) +
    (if p.use_scibert then scibert_dim else 0)),
  // If we're using Bert, no LSTM. We just pass the token embeddings right through.
  local context_layer_output_size = (if p.finetune_bert
    then token_embedding_dim
    else 2 * p.lstm_hidden_size),
  local endpoint_span_emb_dim = 2 * context_layer_output_size + p.feature_size,
  local span_emb_dim = endpoint_span_emb_dim,
  local pair_emb_dim = 3 * span_emb_dim,
  local relation_scorer_dim = pair_emb_dim,
  local coref_scorer_dim = pair_emb_dim + p.feature_size,
  local trigger_emb_dim = if event_n_span_prop > 0 then span_emb_dim else context_layer_output_size,
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
  local argument_pair_dim = (span_emb_dim + trigger_emb_dim + p.feature_size + 2 +
    (if event_args_use_ner_labels then ner_label_dim else 0) +
    (if event_args_use_trigger_labels then trigger_label_dim else 0) +
    (if events_context_window > 0 then 4 * events_context_window * context_layer_output_size else 0)),
  // Add token embedding dim because of the cls token.
  local argument_scorer_dim = (argument_pair_dim + class_projection_dim),

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
      type: "pretrained_transformer_mismatched",
      model_name: (if p.use_bert_base then "bert-base-cased"
                         else if p.use_bert_large then "bert-large-cased"
                         else "allenai/scibert_scivocab_cased")
    }
  },


 local text_field_embedder = {
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
        type: "pretrained_transformer_mismatched",
        model_name: (if p.use_bert_base then "bert-base-cased"
                           else if p.use_bert_large then "bert-large-cased"
                           else "allenai/scibert_scivocab_cased")

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

  random_seed: getattr(p, "random_seed", 13370),
  numpy_seed: getattr(p, "numpy_seed", 1337),
  pytorch_seed: getattr(p, "pytorch_seed", 133),
  dataset_reader: {
    type: "dygie",
    token_indexers: token_indexers,
    max_span_width: p.max_span_width,
    cache_directory: "cache"
  },
  train_data_path: std.extVar("ie_train_data_path"),
  validation_data_path: std.extVar("ie_dev_data_path"),
  test_data_path: std.extVar("ie_test_data_path"),
  // If provided, use pre-defined vocabulary. Else compute on the fly.
  [if "vocab_path" in p then "vocabulary"]: {
    type: "from_files",
    directory: p.vocab_path // config changed TODO
  },
  model: {
    type: "dygie",
    text_field_embedder: text_field_embedder,
    initializer: dygie_initializer,
    loss_weights: p.loss_weights,
    feature_size: p.feature_size,
    max_span_width: p.max_span_width,
    display_metrics: display_metrics[p.target],
    feedforward_params: {
      num_layers: p.feedforward_layers,
      hidden_dims: p.feedforward_dim,
      dropout: p.feedforward_dropout
    },
    modules: {
      coref: {
        spans_per_word: p.coref_spans_per_word,
        max_antecedents: p.coref_max_antecedents,
        coref_prop: p.coref_prop,
        initializer: module_initializer
      },
      ner: {
        initializer: module_initializer
      },
      relation: {
        spans_per_word: p.relation_spans_per_word,
        initializer: module_initializer
      },
      events: {
        trigger_spans_per_word: p.trigger_spans_per_word,
        argument_spans_per_word: p.argument_spans_per_word,
        initializer: module_initializer,
        loss_weights: p.loss_weights_events,
      }
    }
  },
  data_loader: {
    type: "ie_batch",
    batch_size: p.batch_size,
    [if "instances_per_epoch" in p then "instances_per_epoch"]: p.instances_per_epoch
  },
  validation_data_loader: {
   type: "ie_batch",
   batch_size: p.batch_size
  },

  trainer: {
    checkpointer : {
        num_serialized_models_to_keep: 3
    },
    num_epochs: p.num_epochs,
    grad_norm: 5.0,
    patience : p.patience,
    cuda_device : std.parseInt(std.extVar("cuda_device")),
    validation_metric: validation_metrics[p.target],
    optimizer: p.optimizer,
    [if "moving_average_decay" in p then "moving_average"]: {
      type: "exponential",
      decay: p.moving_average_decay
    }
  }
}
