# DyGIE++

Implements the model described in the paper [Entity, Relation, and Event Extraction with Contextualized Span Representations](https://www.semanticscholar.org/paper/Entity%2C-Relation%2C-and-Event-Extraction-with-Span-Wadden-Wennberg/fac2368c2ec81ef82fd168d49a0def2f8d1ec7d8).

This repository is under construction. To train a joint IE model on the `scierc` dataset, see the [model training instructions](#training-a-model). Support for more datasets will be added.

## Depenencies

The only dependencies for the modeling code are [AllenNLP](https://allennlp.org/) 1.2.0 and [PyTorch](https://pytorch.org/)0.9.0. It may run with newer versions, but this is not guarenteed. The necessary dependencies can be installed with `pip install -r requirements.txt`. For PyTorch GPU support, follow the instructions on the [PyTorch](https://pytorch.org/).

Pandas is also required for data preprocessing.


## Training a model

To train a model for named entity recognition, relation extraction, and coreference resolution on the SciERC dataset:

- **Download the data**. From the top-level folder for this repo, enter `bash ./scripts/data/get_scierc.sh`. This will download the scierc dataset into a folder `./data/scierc`
- **Train the model**. Enter `bash ./scripts/train/train_scierc.sh`. This will kick off training, and save a model at `./models/scierc`.

The model uses BERT and coreference propagation to create globally-contextualized embeddings. During training, you may get warnings `WARNING - root - NaN or Inf found in input tensor.`. This may be due to an  [AllenNLP issue](https://github.com/allenai/allennlp/issues/3116). We're looking into it.


<!-- ## To organize.

This repository contains an implementation of the DyGIE++ information extraction model presented in TODO(Dave and Ulme's paper). DyGIE++ achieves state-of the art or competitive performance on three sentence-level IE tasks:

- Named entity recognition, entities with nested or overlapping text spans.
- Relation extraction.
- Event extraction, which involves extracting event triggers together with their arguments.

We have trained and evaluated DyGIE++ on the following datasets:

- `ACE05`: Entity and relation extraction on the [Automatic Content Extraction](https://www.ldc.upenn.edu/collaborations/past-projects/ace) (ACE) corpus of newswire and internet text.
- `ACE05-Event`: Entity, relation, and event extraction on the ACE corpus.
- `SciERC`: Entiy and relation extraction on the [SciERC](http://nlp.cs.washington.edu/sciIE/) computer science abstract corpus.
- `GENIA`: Entity extraction on the [GENIA](http://www.geniaproject.org/) corpus of biomedical abstracts.
- `Wet Lab Protocol Corpus (WLPC)`: Entity and relation extraction for [Wet Lab Protocol Corpus](http://bionlp.osu.edu:5000/protocols) (WLPC). -->

<!-- See TODO(cite the paper) for more details on the data. -->

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
