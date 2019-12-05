We provide details on the data preprocessing for each of the datasets available here.

## Data format

TODO

## SciERC

The [SciERC dataset](http://nlp.cs.washington.edu/sciIE/) contains entity, relation, and coreference annotations for abstracts from computer science research papers.

No preprocessing is required for this data. Run the script `./scripts/data/get_scierc.sh`, and the data will be downloaded and placed in `./data/scierc/processed_data`


## GENIA

The [GENIA dataset](https://orbit.nlm.nih.gov/browse-repository/dataset/human-annotated/83-genia-corpus/visit) contains entity, relation, and event annotations for abstracts from biomedical research papers. Entities can be nested over overlapping. Our preprocessing code converts entity and coreference links to our JSON format. Event shave a complex hierarchical structure, and are left to future work.

To download the GENIA data and preprocess it into the form used in our paper, run the script `./scripts/data/get_genia.sh`. The final `json` versions of the data will be placed in `./data/genia/processed-data`. We use the `json-coref-ident-only` version. The script will take roughly 10 minutes to run.

In GENIA, coreference annotations are labeled one of `IDENT, NONE, RELAT, PRON, APPOS, OTHER, PART-WHOLE, WHOLE-PART`. In the processed data folder, `json-coref-all` has all coreference annotations. `json-coref-ident-only` uses only `IDENT` coreferences. We use the `ident-only` version in our experiments. `json-ner` has only the named entity annotations.

We followed the preprocessing and train / dev / test split from the [SUTD NLP group's](https://gitlab.com/sutd_nlp/overlapping_mentions/tree/master/data/GENIA) work on overlapping entity mention detection for GENIA. We added some additional scripts to convert their named entity data to our JSON format, and to merge in the GENIA coreference data. Some documents were named slightly differently in the entity and coreference data, and we did our best to stitch the annotations back together.

We encountered off-by-one errors stitching together the coref and ner annotations for 10 training documents, and excluded these. They are listed in `./scripts/data/genia/exclude.txt`. If for some reason you want to include these documents anyhow, pass the `--keep-excluded` flag as detailed in a comment at the end of  `./scripts/data/get_genia.sh`.


## ACE Relation

The [ACE 2005](https://catalog.ldc.upenn.edu/LDC2006T06) dataset contains entity, relation, and event annotations for an assortment of newswire and online text. Our preprocessing code is based on the code from the [LSTM-ER repo](https://github.com/tticoin/LSTM-ER), and uses the train / dev / test split described in [Miwa and Bansal (2016)](https://www.semanticscholar.org/paper/End-to-End-Relation-Extraction-using-LSTMs-on-and-Miwa-Bansal/3899f87a2031f3434f89beb68c11a1ca6428328a).


## ACE Event

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


## WLPC

TODO
