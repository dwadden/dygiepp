We provide details on the data preprocessing for each of the datasets available here.

## Data format

TODO

## SciERC

The [SciERC dataset](http://nlp.cs.washington.edu/sciIE/) contains entity, relation, and coreference annotations for abstracts from computer science research papers.

No preprocessing is required for this data. Run the script `./scripts/data/get_scierc.sh`, and the data will be downloaded and placed in `./data/scierc/processed_data`


## GENIA

The [GENIA dataset](https://orbit.nlm.nih.gov/browse-repository/dataset/human-annotated/83-genia-corpus/visit) contains entity, relation, and event annotations for abstracts from biomedical research papers. Entities can be nested over overlapping. Our preprocessing code converts entity and coreference links to our JSON format. Event shave a complex hierarchical structure, and are left to future work.

To download the GENIA data and preprocess it into the form used in our paper, run the script `./scripts/data/get_genia.sh`. The final `json` versions of the data will be placed in `./data/genia/processed-data`. We use the `json-coref-ident-only` version. The script will take roughly 10 minutes to run.

### Details

In GENIA, coreference annotations are labeled one of `IDENT, NONE, RELAT, PRON, APPOS, OTHER, PART-WHOLE, WHOLE-PART`. In the processed data folder, `json-coref-all` has all coreference annotations. `json-coref-ident-only` uses only `IDENT` coreferences. We use the `ident-only` version in our experiments. `json-ner` has only the named entity annotations.

We followed the preprocessing and train / dev / test split from the [SUTD NLP group's](https://gitlab.com/sutd_nlp/overlapping_mentions/tree/master/data/GENIA) work on overlapping entity mention detection for GENIA. We added some additional scripts to convert their named entity data to our JSON format, and to merge in the GENIA coreference data. Some documents were named slightly differently in the entity and coreference data, and we did our best to stitch the annotations back together.

We encountered off-by-one errors stitching together the coref and ner annotations for 10 training documents, and excluded these. They are listed in `./scripts/data/genia/exclude.txt`. If for some reason you want to include these documents anyhow, pass the `--keep-excluded` flag as detailed in a comment at the end of  `./scripts/data/get_genia.sh`.


## ACE

TODO


## WLPC

TODO
