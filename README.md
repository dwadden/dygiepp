# DyGIE++

Implements the model described in the paper [Entity, Relation, and Event Extraction with Contextualized Span Representations](https://www.semanticscholar.org/paper/Entity%2C-Relation%2C-and-Event-Extraction-with-Span-Wadden-Wennberg/fac2368c2ec81ef82fd168d49a0def2f8d1ec7d8).

This repository is under construction. To train a joint IE model on the `SciERC` or `GENIA` dataset, see the [model training instructions](#training-a-model). To download a pretrained model a evaluate it, see the section on [pretrained models](#pretrained-models). Support for more datasets will be added.

## Depenencies

This code was developed using Python 3.7. To create a new Conda environment using Python 3.7, do `conda create --name dygiepp python=3.7`.

The necessary dependencies can be installed with `pip install -r requirements.txt`.

The only dependencies for the modeling code are [AllenNLP](https://allennlp.org/) 0.9.0 and [PyTorch](https://pytorch.org/) 1.2.0. It may run with newer versions, but this is not guarenteed. For PyTorch GPU support, follow the instructions on the [PyTorch](https://pytorch.org/).

For data preprocessing a few additional data and string processing libraries are required including, [Pandas](https://pandas.pydata.org) and [Beautiful Soup 4](https://www.crummy.com/software/BeautifulSoup/bs4/doc/).

Finally, you'll need SciBERT for the scientific datasets. Run `python scripts/pretrained/get_scibert.py` to download and extract the SciBERT model to `./pretrained`.

## Training a model

### SciERC

**Status**: Should give test set F1 results close to those in Table 1 of the paper.

To train a model for named entity recognition, relation extraction, and coreference resolution on the SciERC dataset:

- **Download the data**. From the top-level folder for this repo, enter `bash ./scripts/data/get_scierc.sh`. This will download the scierc dataset into a folder `./data/scierc`
- **Train the model**. Enter `bash ./scripts/train/train_scierc.sh [gpu-id]`. The `gpu-id` should be an integer like `1`, or `-1` to train on CPU. The program will train a model and save a model at `./models/scierc`.

The model uses BERT and coreference propagation to create globally-contextualized embeddings. During training, you may get warnings `WARNING - root - NaN or Inf found in input tensor`. This is due to a [tensorboard logging issue](https://github.com/allenai/allennlp/issues/3116) and doesn't mean your model is diverging.


### GENIA

**Status**: In progress. May not reproduce paper yet.

The steps are similar to SciERC.

- **Download the data**. From the top-level folder for this repo, enter `bash ./scripts/data/get_genia.sh`.
- **Train the model**. Enter `bash ./scripts/train/train_genia.sh [gpu-id]`. The program will train a model and save a model at `./models/genia`.


## Evaluating a model

To check the performance of one of your models or a pretrained model,, you can run the script `bash scripts/evaluate/evaluate_model.sh`. See the script for usage instructions. For example, to evaluate the [pretrained SciERC model](#pretrained-models), you could do

```shell
bash scripts/evaluate/evaluate_model.sh \
  pretrained/scierc.tar.gz \  # Path to model.
  data/scierc/processed_data/json/test.json \  # Path to test data.
  2  # CUDA device
```

## Pretrained models

We versions of DyGIE++ trained on SciERC and GENIA available. More coming soon.

### Downloads

Run `./scripts/pretrained/get_dygiepp_pretrained.sh` to download all the available pretrained models to the `pretrained` directory. If you only want one model, here are the download links:

- [SciERC](https://s3-us-west-2.amazonaws.com/ai2-s2-research/dygiepp/scierc.tar.gz)
- [GENIA](https://s3-us-west-2.amazonaws.com/ai2-s2-research/dygiepp/scierc.tar.gz)

#### Performance of downloaded models

The SciERC model gives slightly better test set performance than reported in the paper:

```
2019-11-20 16:03:12,692 - INFO - allennlp.commands.evaluate - Finished evaluating.
...
2019-11-20 16:03:12,693 - INFO - allennlp.commands.evaluate - _ner_f1: 0.6855290303565666
...
2019-11-20 16:03:12,693 - INFO - allennlp.commands.evaluate - rel_f1: 0.4867781975175391
```

Similarly for GENIA:
```
2019-11-21 14:45:44,505 - INFO - allennlp.commands.evaluate - ner_f1: 0.7818707451272466
```

### Predicting

To make a prediction on the SciERC test set with the pretrained SciERC model, run the script `bash ./scripts/predict/predict_scierc_pretrained.sh`. The predictions will be output to `predictions/scierc_test.json`. The gold labels are also included for easy comparison. All predicted fields start with the prefix `predicted_`, for instance `predicted_ner`.

The prediction code should work but is not cleaned up yet, so please file an issue if you run into problems!


<!-- TODO: multi-GPU training. -->

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
