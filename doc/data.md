# Data

We provide details on the data preprocessing for each of the datasets available here.

## Table of contents

- [Data format](#data-format)
- [Format for predictions](#format-for-predictions)
- [Code for data manipulation](#code-for-data-manipulation)
- [Formatting a new dataset](#formatting-a-new-dataset)
- [Preprocessing details for existing datasets](#preprocesing-details-for-existing-datasets)

## Data format

After preprocessing, all the datasets will be formatted like the [SciERC dataset](http://nlp.cs.washington.edu/sciIE/). After downloading the data, you can look at `data/scierc/normalized_data/json/train.json` as an example. Each line in the dataset is a JSON representation of a document (technically, the files should be given the `.jsonl` extension since each line is a JSON object, sorry for the confusion).

### Mandatory fields

- `doc_key`: A unique string identifier for the document.
- `dataset`: A string identifier for the dataset this document comes from. For more on this field, see the document on [multi-datset training](model.md)
- `sentences`: The senteces in the document, written as a nested list of tokens. For instance,
  ```json
  [
    ["Seattle", "is", "a", "rainy", "city", "."],
    ["Jenny", "Durkan", "is", "the", "city's", "mayor", "."],
    ["She", "was", "elected", "in", "2017", "."]
  ]
  ```
  Empty strings (`""`) are not allowed as entries in sentences; the reader will raise an error if it encounters these.

### Optional annotation fields

- `weight`: When training, multiply the loss for the document by this weight (a float). This is useful when combining datasets of different sizes, or when combining weakly-labeled data with gold annotations.

- `ner`: The named entities in the document, written as a nested list - one sublist per sentence. Each list entry is of the form `[start_tok, end_tok, label]`. The `start_tok` and `end_tok` token indices are with respect to the _document_, not the sentence. For instance the entities in the sentence above might be:
  ```json
  [
    [[0, 0, "City"]],
    [[6, 7, "Person"], [9, 10, "City"]],
    [[13, 13, "Person"], [17, 17, "Year"]]
  ]
  ```
  These entity types are just an example; they don't reflect an entity schema for an actual dataset.
- `relations`: The relations in the document, also one sublist per sentence. Each list entry is of the form `[start_tok_1, end_tok_1, start_tok_2, end_tok_2, label]`.
   ```json
   [
     [],
     [[6, 7, 9, 10, "Mayor-Of"]],
     [[13, 13, 17, 17, "Elected-In"]]
   ]
   ```
- `clusters`: The coreference clusters. This is a nested list, but here each sublist gives the spans of each mention in the coreference cluster. Clusters can cross sentence boundaries. For instance, the first cluster in this example is for Seattle and the second is for the mayor.
  ```json
  [
    [
      [0, 0], [9, 10]
    ],
    [
      [6, 7], [13, 13]
    ]
  ]
  ```

The SciERC dataset does not have any event data. To see an example of event data, run the ACE event preprocessing steps described in the [README](README.md) and look at one of the files in `data/ace-event/processed-data`. You will see the following additional field:
- `events`: The events in the document, with one sublist per sentence. An event with `N` arguments will be written as a list of the form `[[trigger_tok, event_type], [start_tok_arg1, end_tok_arg1, arg1_type], [start_tok_arg2, end_tok_arg2, arg2_type], ..., [start_tok_argN, end_tok_argN, argN_type]]`. Note that in ACE, event triggers can only be a single token. For instance,
  ```json
  [
    [],
    [],
    [
      [
        [15, "Peronnel.Election"],
        [13, 13, "Person"],
        [17, 17, "Date"]
      ]
    ]
  ]
  ```

- `event_clusters`: The event coreference clusters. The structure is the same as `clusters`, but each cluster corresponds to an event, rather than an entity. Each span corresponds to the span of the trigger. While event triggers can only be a single token in ACE, we keep the ending token for consistency with `clusters`. NOTE: Event clusters were added by a contributor and are not "officially supported".
  ```json
  [
    [
      [517, 517], [711, 711], [723, 723]
    ],
    [
      [603, 603], [741, 741]
    ]
  ]
  ```
There may also be a `sentence_start` field indicating the token index of the start of each sentence with respect to the document. This can be ignored.


### User-defined sentence metadata

You can define additional metadata associated with each sentence that will be ignored by the model; these metadata fields should be prefixed with `_`. For instance, if you wanted to explicitly keep track of the index of each sentence in a document, you could add a field to your input document

```python
{
  "doc_key": "some_document",
  "dataset": "some_dataset",
  "weight": 0.5,
  "sentences": [["One", "sentence"], ["Another", "sentence"]],
  "_sentence_index": [0, 1]   # User-added metadata field.
}
```

## Format for predictions

When model predictions are saved to file, they are formatted as described above, but with the following changes:

- The field names have the word `predicted` prepended. For instance, `predicted_ner`, `predicted_relations`, etc.
- Each prediction has two additional entries appended, specifying the logit score and softmax probability for the predicted label. For instance:
  - A single predicted relation prediction has the form `[start_tok_1, end_tok_1, start_tok_2, end_tok_2, predicted_label, label_logit, label_softmax]`.
  - A single predicted event has the form `[[trigger_tok, predited_event_type, event_type_logit, event_type_softmax], [start_tok_arg1, end_tok_arg1, predicted_arg1_type, arg1_type_logit, arg1_type_softmax], ...]`.
  - TODO: This hasn't been implemented yet for coreference.


## Code for data manipulation

The module [document.py](../dygie/data/dataset_readers/document.py) contains classes and methods to load, save, manipulate, and visualize DyGIE-formatted data. See [document.ipynb](../notebooks/document.ipynb) for usage examples.


## Formatting a new dataset

If you'd like to use a pretrained DyGIE++ model to make predictions on a new dataset, the `dataset` field in your new dataset must match the `dataset` that the original model was trained on; this indicates to the model which label namespace it should use for predictions. See the section on [available pretrained models](../README.md#pretrained-models) for the dataset names that go with each model. For more on label namespaces, see the section on [multi-dataset training](model.md/#multi-dataset-training).

### Unlabled data

In the case where your unlabeled data are stored as a directory of `.txt` files (one file per document), you can run `python scripts/data/new-dataset/format_new_dataset.py [input-directory] [output-file]` to format the documents into a `jsonl` file, with one line per document. If your dataset is scientific text, add the `--use-scispacy` flag to have [SciSpacy](https://allenai.github.io/scispacy/) do the tokenization.

If your data do not come in this form, you can follow this basic recipe as in the script:

-  Use [Spacy](https://spacy.io) (or [SciSpacy](https://allenai.github.io/scispacy/) for scientific text) to split each document into sentences and then tokens.
- Collect all the documents into a single `jsonl` file, one line per document, using some appropriate scheme for the `doc_key`'s of each document.

### Labeled data

Many labeled datasets have two files per document:
1. A file with the document text (usually a `.txt` file).
2. A file with the character indices of each entity mention in the source document, as well as annotations indicating the entity mentions that take part in relations and events (often an `.ann` file).

Getting a dataset like this into the DyGIE format requires tokenizing the text, aligning the character-level named entity annotations to the tokenized text, and mapping relation and event mentions to tokens. This can be noisy, because the token boundaries generated by the tokenizer may not always align with the character indices of the named entities. The simplest way to handle this is to just throw out unmatched entities. The general process is implemented in `scripts/data/chemprot/02_chemprot_to_input.py`.

If you're stuck on preprocessing a dataset, post an issue. Or, if you come up with a nice, general preprocessing script for labeled data, submit a PR!

#### Converting data labeled with brat 
The script [brat_to_input.py](https://github.com/dwadden/dygiepp/tree/master/scripts/new-dataset/brat_to_input.py) is a general preprocessing script for data that was annotated with the [brat rapid annotation tool](https://brat.nlplab.org/). This script performs the tokenization and alignment of the  text in which character indices are converted to document-level token indices, and relations and events are mapped to these tokens. The output of this script is a file containing one json-formatted dict per document, with the required fields for input to DyGIE. Entities that can't be aligned using the spacy tokenization of the text are thrown out with a warning, and `coref` and `event` fields will only be included if those data types are present. You can use the script like this:
```
python brat_to_input.py \
  path/to/my/data/ \
  output_file.jsonl \
  scierc \
  --use-scispacy \
  --coref \
```  
The `--use-scispacy` flag indicates that scispacy will be used as the tokenizer. The --coref flag indicates whether or not to treat brat's equivalence relation type (with the type `*` in the `.ann` file) as coreference clusters. This is currently the only way to include coreferences. NOTE: This code was added by a contributor, and is not "officially supported". If you run into problems, create an issue and tag @serenalotreck. 

### Dealing with long sentences

Most transformer-based encoders have a 512-token limit. Sentences longer than this will cause an error. Unfortunately, you can't just check that each of your `sentences` fields is at most 512 tokens. These tokens are converted to BERT's byte pair encoding, and a single "word token" may be split into multiple "BERT tokens". We have provided a script `scripts/data/shared/check_sentence_length.py`, which you can run on an input file. It will identify sentences whose byte pair encodings exceed the limit of the encoder you're using.

If you have sentences that are too long, you have two options:

1. Split the long sentences, and re-run `check_sentence_length.py` to make check that they're short enough.

2. Modify the config so that the encoder splits long sentences into bite-sized pieces, encodes them separately, and merges them back together. I haven't tried this (PR's getting this to work would be welcome). To train a model this way, it should work to modify the training config like so:

```jsonnet
dataset_reader +: {
  token_indexers +: {
    bert +: {
      max_length: 512
    }
  }
},
model +: {
  embedder +: {
    token_embedders +: {
      bert +: {
        max_length: 512
      }
    }
  }
}
```

I'm not sure this will work with `allennlp predict`; these options might have to be set with the `--overrides` flag to the `predict` command.

For more information, check the `max_length` parameter on AllenNLP's [PretrainedTransformerMismatchedIndexer](https://docs.allennlp.org/master/api/data/token_indexers/pretrained_transformer_mismatched_indexer/). Note that the token indexer and token embedder must be given the same `max_length`.

If you're using a model other than BERT and you don't know the correct `max_length` to set, you can get it like this:

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained([model_name])
print(config.max_position_embeddings)
```


## Preprocesing details for existing datasets

Many information extraction datasets have two files for each document, of roughly the following form:
1. The text of the document (often a `.txt` file).
2. A file with mappings from character indices to NER mentions, and from pairs / tuples of NER mentions to relation mentions / event mentions (often an `.ann` file, but sometimes `.xml` or something else).

To preprocess this kind of data, you generally need to:
- Sentencize and tokenize the document text using [Spacy](https://spacy.io) or similar.
- Align the character-level NER mentions with the tokenized document. Often, there are a few cases where token boundaries to not line up with the character indices of the annotation entity mentions.
- Map the relation / event mentions to the token-aligned NER mentions.

This process is generally messy. As an example, see `./scripts/data/chemprot/02_chemprot_to_input.py`. Feel free to create an issue if there's a specific case that warrants attention.

### SciERC

The [SciERC dataset](http://nlp.cs.washington.edu/sciIE/) contains entity, relation, and coreference annotations for abstracts from computer science research papers.

No preprocessing is required for this data. Run the script `./scripts/data/get_scierc.sh`, and the data will be downloaded and placed in `./data/scierc/processed_data`


### GENIA

The [GENIA dataset](https://orbit.nlm.nih.gov/browse-repository/dataset/human-annotated/83-genia-corpus/visit) contains entity, relation, and event annotations for abstracts from biomedical research papers. Entities can be nested over overlapping. Our preprocessing code converts entity and coreference links to our JSON format. Event shave a complex hierarchical structure, and are left to future work.

To download the GENIA data and preprocess it into the form used in our paper, run the script `./scripts/data/get_genia.sh`. The final `json` versions of the data will be placed in `./data/genia/processed-data`. We use the `json-coref-ident-only` version. The script will take roughly 10 minutes to run.

In GENIA, coreference annotations are labeled one of `IDENT, NONE, RELAT, PRON, APPOS, OTHER, PART-WHOLE, WHOLE-PART`. In the processed data folder, `json-coref-all` has all coreference annotations. `json-coref-ident-only` uses only `IDENT` coreferences. We use the `ident-only` version in our experiments. `json-ner` has only the named entity annotations.

We followed the preprocessing and train / dev / test split from the [SUTD NLP group's](https://gitlab.com/sutd_nlp/overlapping_mentions/tree/master/data/GENIA) work on overlapping entity mention detection for GENIA. We added some additional scripts to convert their named entity data to our JSON format, and to merge in the GENIA coreference data. Some documents were named slightly differently in the entity and coreference data, and we did our best to stitch the annotations back together.

We encountered off-by-one errors stitching together the coref and ner annotations for 10 training documents, and excluded these. They are listed in `./scripts/data/genia/exclude.txt`. If for some reason you want to include these documents anyhow, pass the `--keep-excluded` flag as detailed in a comment at the end of  `./scripts/data/get_genia.sh`.


### ACE Relation

The [ACE 2005](https://catalog.ldc.upenn.edu/LDC2006T06) dataset contains entity, relation, and event annotations for an assortment of newswire and online text. Our preprocessing code is based on the code from the [LSTM-ER repo](https://github.com/tticoin/LSTM-ER), and uses the train / dev / test split described in [Miwa and Bansal (2016)](https://www.semanticscholar.org/paper/End-to-End-Relation-Extraction-using-LSTMs-on-and-Miwa-Bansal/3899f87a2031f3434f89beb68c11a1ca6428328a).


### ACE Event

We start off with the same data as for ACE Relation, but use different splits and preprocessing. For ACE Event, we use the standard split for event extraction used in [Yang and Mitchell (2016)](https://www.semanticscholar.org/paper/Joint-Extraction-of-Events-and-Entities-within-a-Yang-Mitchell/c558e2b5dcab8d89f957f3045a9bbd43fd6a28ed). Unfortunately, there are a number of different ways that the ACE data can be preprocessed. We follow the conventions of [Zhang et al. (2019)](https://www.semanticscholar.org/paper/Joint-Entity-and-Event-Extraction-with-Generative-Zhang-Ji/ea00a63c2acd145839eb6f6bbc01a5cfb4930d43), which claimed SOTA at the time our paper was submitted.

Unfortunately, different papers have used different conventions and therefore our results may not be directly comparable. However, we have included flags in the script `./scripts/data/ace-event/parse_ace_event.py` to allow researchers to make different preprocessing choices. The available flags are:

- **use_span_extent**: By default, when defining entity mentions, we use the `head` of the mention, rather than its `extent`, as in this example:
  ```xml
  <entity_mention ID="AFP_ENG_20030330.0211-E3-1" TYPE="NOM" LDCTYPE="NOM" LDCATR="FALSE">
    <extent>
      <charseq START="134" END="170">Some 2,500 mainly university students</charseq>
    </extent>
    <head>
      <charseq START="163" END="170">students</charseq>
    </head>
  </entity_mention>
  ```
  Running `parse_ace_event.py` with the flag `--use_span_extent` will use `extent`s rather than `head`s.

- **include_times_and_values**: By default, `timex2` and `value` mentions are *not* treated as entity mentions, and are ignored. For instance, this annotation would be ignored:
  ```xml
  <timex2 ID="AFP_ENG_20030327.0022-T1" VAL="2003-03-27">
    <timex2_mention ID="AFP_ENG_20030327.0022-T1-1">
      ...
    </timex2_mention>
  </timex2>
  ```
  So would this one:
  ```xml
  <value ID="AFP_ENG_20030330.0211-V1" TYPE="Numeric" SUBTYPE="Percent">
    <value_mention ID="AFP_ENG_20030330.0211-V1-1">
      ...
    </value_mention>
  </value>
  ```
  To include these mentions as entity mentions, use the flag `--include_times_and_values`. Note that all values are given entity type `VALUE`. Some work has assigned entity types using the `TYPE` of the value - for instance `"Numeric"` in the example above. We welcome a pull request to add this feature.

- **include_pronouns**: By default, pronouns (entities with `TYPE="PRO"`) are also *ignored*. For instance, this annotation  would be ignored:
  ```xml
  <entity_mention ID="AFP_ENG_20030330.0211-E3-2" TYPE="PRO" LDCTYPE="WHQ" LDCATR="FALSE">
    ...
  </entity_mention>
  ```
  To include pronouns as entity mentions, use the flag `--include_pronouns`.

- **include_entity_coreference**: Added by PR, not "officially supported". Include entity coreference clusters.

- **include_event_coreference**: Added by PR, not "officially supported". Include event coreference clusters.


### WLPC

TODO
