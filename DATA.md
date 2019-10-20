We provide details on the data preprocessing for each of the datasets available here.

## Data format

TODO

## SciERC

The [SciERC dataset](http://nlp.cs.washington.edu/sciIE/) contains entity, relation, and coreference annotations for abstracts from computer science research papers.

No preprocessing is required for this data. Run the script `./scripts/data/get_scierc.sh`, and the data will be downloaded and placed in a folder.


## GENIA

The [GENIA dataset](https://orbit.nlm.nih.gov/browse-repository/dataset/human-annotated/83-genia-corpus/visit) contains entity, relation, and event annotations for abstracts from biomedical research papers. Entities can be nested over overlapping. Our preprocessing code converts entity and coreference links to our JSON format. Event shave a complex hierarchical structure, and are left to future work.

To download the GENIA data and preprocess it into the form used in our paper, run the script `./scripts/data/get_genia.sh`.

We followed the preprocessing and train / dev / test split from the [https://gitlab.com/sutd_nlp/overlapping_mentions/tree/master/data/GENIA](SUTD NLP group's) work on overlapping entity mention detection for GENIA. We added some additional scripts to convert their named entity data to our JSON format, and to merge in the GENIA coreference data. Some documents were named slightly differently in the entity and coreference data, and we did our best to stitch the annotations back together.

In GENIA, coreference annotations are labeled one of `IDENT, NONE, RELAT, PRON, APPOS, OTHER, PART-WHOLE, WHOLE-PART`. We preprocess two versions of the data; `json-coref-all` has all coreference annotations, while `json-coref-ident-only` uses only `IDENT` coreferences. We use the `ident-only` version in our experiments.


## ACE

TODO


## WLPC

TODO
