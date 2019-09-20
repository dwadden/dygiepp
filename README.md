# DyGIE++

This repository contains an implementation of the DyGIE++ information extraction model presented in TODO(Dave and Ulme's paper). DyGIE++ achieves state-of the art or competitive performance on three sentence-level IE tasks:

- Named entity recognition, entities with nested or overlapping text spans.
- Relation extraction.
- Event extraction, which involves extracting event triggers together with their arguments.

We have trained and evaluated DyGIE++ on the following datasets:

- `ACE05`: Entity and relation extraction on the [Automatic Content Extraction](https://www.ldc.upenn.edu/collaborations/past-projects/ace) (ACE) corpus of newswire and internet text.
- `ACE05-Event`: Entity, relation, and event extraction on the ACE corpus.
- `SciERC`: Entiy and relation extraction on the [SciERC](http://nlp.cs.washington.edu/sciIE/) computer science abstract corpus.
- `GENIA`: Entity extraction on the [GENIA](http://www.geniaproject.org/) corpus of biomedical abstracts.
- `Wet Lab Protocol Corpus (WLPC)`: Entity and relation extraction for [Wet Lab Protocol Corpus](http://bionlp.osu.edu:5000/protocols) (WLPC).

<!-- See TODO(cite the paper) for more details on the data. -->

This repository is a work in progress. We will shortly be providing scripts to format the datasets and train models described in the paper. The `master` branch is intended for release, but isn't working yet. The `develop` branch works, but it's messy. The `master` branch should be functional by the end of October, together with data. If you need to run experiments with this code before then, create an issue and I'll help.

<!-- This repository provides scripts to obtain the data sets, run the primary experiments described in TODO(Dave and Ulme's paper), and make predictions using a pre-trained DyGIE++ model on new data sets. -->

<!-- ## Installation

DyGIE++ is implemented using the AllenNLP framework. TODO(add a requirements.txt file) of things we need to install.

## Obtaining the data sets.

We provide scripts to obtain and preprocess the data sets used to evaluate DyGIE++, located in the `scripts/data` directory.

- `ACE05`: The ACE corpus requires a license and cannot be made available for download. We provide a script `get_ace05.sh` which accepts a path to a download of the ACE data as input, splits it into train, dev, and test as described in TODO(cite the paper), and places preprocessed data at TODO(where does it go)?
- `ACE05-Event`: We provide a script `get_ace05_event.sh` which accepts a path to the ACE data, splits it as in TODO(the paper), and places the prcoessed data at TODO(where)?
- `SciERC`: The SciERC corpus is freely available online. The `get_scierc.sh` script will download the corpus and place it in `data/scierc`.
- `GENIA`: TODO The GENIA corpus can be downloaded from TODO.
- `Wet Lab Protocol Corpus (WLPC)`: The WLPC can be downloaded with permission for the paper's authors. Once downloaded, run `get_wlpc.sh` on the downloaded data folder.


## Making predictions on a new data set

We have pre-trained DyGIE++ models on each of the five datasets described above. The models are available for download at TODO(add a URL). To make predictions on a new dataset, the data must be converted to DyGIE++ readable `.json` files. -->
