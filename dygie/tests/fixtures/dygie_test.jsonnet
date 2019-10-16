// Quick test that doesn't load in any data.

// Primary prediction target. Watch metrics associated with this target.
local target = "rel";

// Specifies the token-level features that will be created.
local use_glove = true;
local use_char = true;
local use_elmo = false;
local use_attentive_span_extractor = true;

// Specifies the model parameters.
local lstm_hidden_size = 200;
local lstm_n_layers = 1;
local feature_size = 10;
local feedforward_layers = 2;
local char_n_filters = 50;
local feedforward_dim = 150;
local max_span_width = 8;
local feedforward_dropout = 0.2;
local lexical_dropout = 0.5;
local lstm_dropout = 0.4;
local loss_weights = {
  "ner": 1.0,
  "relation": 1.0,
  "coref": 1.0
};

// Coref settings.
local coref_spans_per_word = 0.4;
local coref_max_antecedents = 100;

// Relation settings.
local relation_spans_per_word = 0.4;
local relation_positive_label_weight = 1.0;

// Model training
local num_epochs = 250;
local patience = 25;
local learning_rate_scheduler = {
  "type": "reduce_on_plateau",
  "factor": 0.5,
  "mode": "max",
  "patience": 5
};
local learning_rate = 0.001;


////////////////////////////////////////////////////////////////////////////////

// Nothing below this line needs to change.


// Storing constants.

local validation_metrics = {
  "ner": "+ner_f1",
  "rel": "+rel_f1",
  "coref": "+coref_f1"
};

local display_metrics = {
  "ner": ["ner_precision", "ner_recall", "ner_f1"],
  "rel": ["rel_precision", "rel_recall", "rel_f1", "rel_span_recall"],
  "coref": ["coref_precision", "coref_recall", "coref_f1", "coref_mention_recall"]
};

local glove_dim = 300;
local elmo_dim = 1024;

local module_initializer = [
  [".*linear_layers.*weight", {"type": "xavier_normal"}],
  [".*scorer._module.weight", {"type": "xavier_normal"}],
  ["_distance_embedding.weight", {"type": "xavier_normal"}]];

local dygie_initializer = [
  ["_span_width_embedding.weight", {"type": "xavier_normal"}],
  ["_context_layer._module.weight_ih.*", {"type": "xavier_normal"}],
  ["_context_layer._module.weight_hh.*", {"type": "orthogonal"}]
];


////////////////////////////////////////////////////////////////////////////////

// Calculating dimensions.

local token_embedding_dim = ((if use_glove then glove_dim else 0) +
  (if use_char then char_n_filters else 0) +
  (if use_elmo then elmo_dim else 0));
local endpoint_span_emb_dim = 4 * lstm_hidden_size + feature_size;
local attended_span_emb_dim = if use_attentive_span_extractor then token_embedding_dim else 0;
local span_emb_dim = endpoint_span_emb_dim + attended_span_emb_dim;
local pair_emb_dim = 3 * span_emb_dim;
local relation_scorer_dim = pair_emb_dim;
local coref_scorer_dim = pair_emb_dim + feature_size;

////////////////////////////////////////////////////////////////////////////////

// Function definitions

local make_feedforward(input_dim) = {
  "input_dim": input_dim,
  "num_layers": feedforward_layers,
  "hidden_dims": feedforward_dim,
  "activations": "relu",
  "dropout": feedforward_dropout
};

// Model components

local token_indexers = {
  [if use_glove then "tokens"]: {
    "type": "single_id",
    "lowercase_tokens": false
  },
  [if use_char then "token_characters"]: {
    "type": "characters",
    "min_padding_length": 5
  },
  [if use_elmo then "elmo"]: {
    "type": "elmo_characters"
  }
};

local text_field_embedder = {
  "token_embedders": {
    [if use_glove then "tokens"]: {
      "type": "embedding",
      // "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.300d.txt.gz",
      "embedding_dim": 300,
      "trainable": false
    },
    [if use_char then "token_characters"]: {
      "type": "character_encoding",
      "embedding": {
        "num_embeddings": 262,
        "embedding_dim": 16
      },
      "encoder": {
        "type": "cnn",
        "embedding_dim": 16,
        "num_filters": char_n_filters,
        "ngram_filter_sizes": [5]
      }
    },
    [if use_elmo then "elmo"]: {
      "type": "elmo_token_embedder",
      "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
      "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
      "do_layer_norm": false,
      "dropout": 0.5
    }
  }
};


////////////////////////////////////////////////////////////////////////////////

// The model

{
  "dataset_reader": {
    "type": "ie_json",
    "token_indexers": token_indexers,
    "max_span_width": max_span_width
  },
  "train_data_path": "tests/fixtures/scierc_article.json",
  "validation_data_path": "tests/fixtures/scierc_article.json",
  "model": {
    "type": "dygie",
    "text_field_embedder": text_field_embedder,
    "initializer": dygie_initializer,
    "loss_weights": loss_weights,
    "lexical_dropout": lexical_dropout,
    "feature_size": feature_size,
    "use_attentive_span_extractor": use_attentive_span_extractor,
    "max_span_width": max_span_width,
    "display_metrics": display_metrics[target],
    "context_layer": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": token_embedding_dim,
      "hidden_size": lstm_hidden_size,
      "num_layers": lstm_n_layers,
      "dropout": lstm_dropout
    },
    "modules": {
      "coref": {
        "spans_per_word": coref_spans_per_word,
        "max_antecedents": coref_max_antecedents,
        "mention_feedforward": make_feedforward(span_emb_dim),
        "antecedent_feedforward": make_feedforward(coref_scorer_dim),
        "initializer": module_initializer
      },
      "ner": {
        "mention_feedforward": make_feedforward(span_emb_dim),
        "initializer": module_initializer
      },
      "relation": {
        "spans_per_word": relation_spans_per_word,
        "positive_label_weight": relation_positive_label_weight,
        "mention_feedforward": make_feedforward(span_emb_dim),
        "relation_feedforward": make_feedforward(relation_scorer_dim),
        "initializer": module_initializer,
      },
    },
  },
  "iterator": {
    "type": "ie_batch",
    "batch_size": 10
  },
  "validation_iterator": {
    "type": "ie_document",
  },
  "trainer": {
    "num_epochs": num_epochs,
    "grad_norm": 5.0,
    "patience" : patience,
    "cuda_device" : -1,
    "validation_metric": validation_metrics[target],
    "learning_rate_scheduler": learning_rate_scheduler,
    "optimizer": {
      "type": "adam",
      "lr": learning_rate,
    },
  }
}
