# DyGIE++

Implements the model described in the paper [Entity, Relation, and Event Extraction with Contextualized Span Representations](https://www.semanticscholar.org/paper/Entity%2C-Relation%2C-and-Event-Extraction-with-Span-Wadden-Wennberg/fac2368c2ec81ef82fd168d49a0def2f8d1ec7d8).

This repository is under construction and we're in the process of adding support for more datasets.

## Table of Contents
- [Dependencies](#dependencies)
- [Model training](#training-a-model)
- [Model evaluation](#evaluating-a-model)
- [Pretrained models](#pretrained-models)
- [Making predictions on existing datasets](#making-predictions-on-existing-datasets)
- [Working with new datasets](#working-with-new-datasets)
- [Contact](#contact)


## Dependencies

This code was developed using Python 3.7. To create a new Conda environment using Python 3.7, do `conda create --name dygiepp python=3.7`.

The necessary dependencies can be installed with `pip install -r requirements.txt`.

The only dependencies for the modeling code are [AllenNLP](https://allennlp.org/) 0.9.0 and [PyTorch](https://pytorch.org/) 1.2.0. It may run with newer versions, but this is not guarenteed. For PyTorch GPU support, follow the instructions on the [PyTorch](https://pytorch.org/).

For data preprocessing a few additional data and string processing libraries are required including, [Pandas](https://pandas.pydata.org) [Beautiful Soup 4](https://www.crummy.com/software/BeautifulSoup/bs4/doc/), and [scispacy](scispacy).

Finally, you'll need SciBERT for the scientific datasets. Run `python scripts/pretrained/get_scibert.py` to download and extract the SciBERT model to `./pretrained`.

### Docker build
A `Dockerfile` is provided with the Pytorch + CUDA + CUDNN base image for a full-stack install.
It will create conda environments `dygiepp` for modeling & `ace-event-preprocess` for ACE05-Event preprocessing.

By default the build downloads datasets and dependencies for all tasks. 
This takes a long time, so you will want to comment out unneeded tasks in the Dockerfile.

- Comment out unneeded task sections in Dockerfile.
- Build container: `docker build --tag dygiepp:dev .`
- Run the container interactively: `docker run --gpus all -it --rm --ipc=host --name dygiepp dygiep:dev`

## Training a model

Warning about coreference resolution: The coreference code will break on sentences with only a single token. If you have these in your dataset, either get rid of them or deactivate the coreference resolution part of the model.

### SciERC

To train a model for named entity recognition, relation extraction, and coreference resolution on the SciERC dataset:

- **Download the data**. From the top-level folder for this repo, enter `bash ./scripts/data/get_scierc.sh`. This will download the scierc dataset into a folder `./data/scierc`
- **Train the model**. Enter `bash ./scripts/train/train_scierc.sh [gpu-id]`. The `gpu-id` should be an integer like `1`, or `-1` to train on CPU. The program will train a model and save a model at `./models/scierc`.
- To train a "lightweight" version of the model that doesn't do coreference propagation and uses a context width of 1, do `bash ./scripts/train/train_scierc_lightweight.sh [gpu-id]` instead. The result will go in `./models/scierc-lightweight`. More info on why you'd want to do this in the section on [making predictions](#making-predictions).


### GENIA

The steps are similar to SciERC.

- **Download the data**. From the top-level folder for this repo, enter `bash ./scripts/data/get_genia.sh`.
- **Train the model**. Enter `bash ./scripts/train/train_genia.sh [gpu-id]`. The program will train a model and save a model at `./models/genia`.
- As with SciERC, we also offer a "lightweight" version with a context width of 1 and no coreference propagation.


### ChemProt

The [ChemProt](https://biocreative.bioinformatics.udel.edu/news/corpora/chemprot-corpus-biocreative-vi/) corpus contains entity and relation annotations for drug / protein interaction. Follow these steps:

- **Get the data**. Run `bash ./scripts/data/get_chemprot.sh`. This will download the data and process it into the DyGIE input format.
  - NOTE: This is a quick-and-dirty script that skips entities whose character offsets don't align exactly with the tokenization produced by SciSpacy. We lose about 10% of the named entities and 20% of the relations in the dataset as a result.
- **Train the model**. Enter `bash ./scripts/train/train_chemprot.sh [gpu-id]`. The model will be saved in `./models/chemprot`.


### ACE05 (ACE for entities and relations)

#### Creating the dataset

For more information on ACE relation and event preprocessing, see [DATA.md](DATA.md) and [this issue](https://github.com/dwadden/dygiepp/issues/11).

We use preprocessing code adapted from the [DyGIE repo](https://github.com/luanyi/DyGIE), which is in turn adapted from the [LSTM-ER repo](https://github.com/tticoin/LSTM-ER). The following software is required:
- Java, to run CoreNLP.
- Perl.
- zsh. If this isn't available on your system, you can create a conda environment and install [zsh](https://anaconda.org/conda-forge/zsh).

First, we need to download Stanford CoreNLP:
```
bash scripts/data/ace05/get_corenlp.sh
```
Then, run the driver script to preprocess the data:
```
bash scripts/data/get_ace05.sh [path-to-ACE-data]
```

The results will go in `./data/ace05/processed-data`. The intermediate files will go in `./data/ace05/raw-data`.

#### Training a model

Enter `bash ./scripts/train/train_ace05_relation.sh [gpu-id]`. A model trained this way will not reproduce the numbers in the paper. We're in the process of debugging and will update.

### ACE05 Event

#### Creating the dataset

The preprocessing code I wrote breaks with the newest version of Spacy. So unfortunately, we need to create a separate virtualenv that uses an old version of Spacy and use that for preprocessing.
```shell
conda deactivate
conda create --name ace-event-preprocess python=3.7
conda activate ace-event-preprocess
pip install -r scripts/data/ace-event/requirements.txt
python -m spacy download en
```
Then, collect the relevant files from the ACE data distribution with
```
bash ./scripts/data/ace-event/collect_ace_event.sh [path-to-ACE-data].
```
The results will go in `./data/ace-event/raw-data`.

Now, run the script
```
python ./scripts/data/ace-event/parse_ace_event.py [output-name] [optional-flags]
```
You can see the available flags by calling `parse_ace_event.py -h`. For detailed descriptions, see [DATA.md](DATA.md). The results will go in `./data/ace-event/processed-data/[output-name]`. We require an output name because you may want to preprocess the ACE data multiple times using different flags. For default preprocessing settings, you could do:
```
python ./scripts/data/ace-event/parse_ace_event.py default-settings
```
When finished, you should `conda deactivate` the `ace-event-preprocess` environment and re-activate your modeling environment.

#### Training the model

Enter `bash ./scripts/train/train_ace05_event.sh [gpu-id]`. The result will go in `models/ace05-event`.  A model trained in this fashion will reproduce (within 0.1 F1 or so) the results in Table 4 of the paper. To reproduce the results in Table 1 requires training an ensemble model of 4 trigger detectors. The basic process is as follows:

- Merge the ACE event train + dev data, then create 4 new train / dev splits.
- Train a separate trigger detection model on each split. To do this, modify `training-config/ace05_event.jsonnet` by setting
  ```jsonnet
  loss_weights_events: {   // Loss weights for trigger and argument ID in events.
    trigger: 1.0,
    arguments: 0.5
  },
  ```
- Make trigger predictions using a majority vote of the 4 ensemble models.
- Use these predicted triggers when making event argument predictions based on the event argument scores output by the model saved at `models/ace05-event`.

If you need more details, email me.

## Evaluating a model

To check the performance of one of your models or a pretrained model, you can use the `allennlp evaluate` command.

Note that `allennlp` commands will only be able to discover the code in this package if:
- You run the commands from the root folder of this project, `dygiepp`, or:
- You add the code to your Python path by running `conda develop .` from the root folder of this project.

Otherwise, you will get an error `ModuleNotFoundError: No module named 'dygie'`.

In general, you can make evaluate a model like this:
```shell
allennlp evaluate \
  [model-file] \
  [data-path] \
  --cuda-device [cuda-device] \
  --include-package dygie \
  --output-file [output-file] # Optional; if not given, prints metrics to console.
```
For example, to evaluate the [pretrained SciERC model](#pretrained-models), you could do
```shell
allennlp evaluate \
  pretrained/scierc.tar.gz \
  data/scierc/processed_data/json/test.json \
  --cuda-device 2 \
  --include-package dygie
```
To evaluate a model you trained on the SciERC data, you could do
```shell
allennlp evaluate \
  models/scierc/model.tar.gz \
  data/scierc/processed_data/json/test.json \
  --cuda-device 2  \
  --include-package dygie \
  --output-file models/scierc/metrics_test.json
```

## Pretrained models

We have versions of DyGIE++ trained on SciERC and GENIA available. There are two versions:
- The "lightweight" versions don't use coreference propagation, and use a context window of 1. If you've got a new dataset and you just want to get some reasonable predictions, use these.
- The "full" versions use coreference propagatation and a context window of 3. Use these if you need to squeeze out another F1 point or two. These models take longer to run, and they may break if they're given inputs that are too long.

### Downloads

Run `./scripts/pretrained/get_dygiepp_pretrained.sh` to download all the available pretrained models to the `pretrained` directory. If you only want one model, here are the download links.

- [SciERC](https://s3-us-west-2.amazonaws.com/ai2-s2-research/dygiepp/scierc.tar.gz)
- [SciERC lightweight](https://s3-us-west-2.amazonaws.com/ai2-s2-research/dygiepp/scierc-lightweight.tar.gz)
- [GENIA](https://s3-us-west-2.amazonaws.com/ai2-s2-research/dygiepp/genia.tar.gz)
- [GENIA lightweight](https://ai2-s2-research.s3-us-west-2.amazonaws.com/dygiepp/genia-lightweight.tar.gz)
- [ChemProt (lightweight only)](https://ai2-s2-research.s3-us-west-2.amazonaws.com/dygiepp/chemprot.tar.gz)
- [ACE05 event (uses BERT large)](https://ai2-s2-research.s3-us-west-2.amazonaws.com/dygiepp/ace05-event.tar.gz)

#### Performance of downloaded models

- SciERC
  ```
  2019-11-20 16:03:12,692 - INFO - allennlp.commands.evaluate - Finished evaluating.
  2019-11-20 16:03:12,693 - INFO - allennlp.commands.evaluate - _ner_f1: 0.6855290303565666
  2019-11-20 16:03:12,693 - INFO - allennlp.commands.evaluate - rel_f1: 0.4867781975175391
  ```

- SciERC lightweight
  ```
  2020-03-31 21:23:34,708 - INFO - allennlp.commands.evaluate - Finished evaluating.
  2020-03-31 21:23:34,709 - INFO - allennlp.commands.evaluate - _ner_f1: 0.6778959810874204
  2020-03-31 21:23:34,709 - INFO - allennlp.commands.evaluate - rel_f1: 0.4638157894736842
  ```

- GENIA
  ```
  2019-11-21 14:45:44,505 - INFO - allennlp.commands.evaluate - ner_f1: 0.7818707451272466
  ```

- GENIA lightweight
  And the lightweight version:
  ```
  2020-05-08 11:18:59,761 - INFO - allennlp.commands.evaluate - ner_f1: 0.7671077504725398
  ```

- ChemProt
  ```
  2020-05-08 23:20:59,648 - INFO - allennlp.commands.evaluate - _ner_f1: 0.8850947021684925
  2020-05-08 23:20:59,648 - INFO - allennlp.commands.evaluate - rel_f1: 0.35027598896044154
  ```
  Note that we're doing span-level evaluation using predicted entities. We're also evaluating on all ChemProt relation classes, while the official task only evaluates on a subset (see [Liu et al.](https://www.semanticscholar.org/paper/Attention-based-Neural-Networks-for-Chemical-Liu-Shen/a6261b278d1c2155e8eab7ac12d924fc2207bd04) for details). Thus, our relation extraction performance is lower than, for instance, [Verga et al.](https://www.semanticscholar.org/paper/Simultaneously-Self-Attending-to-All-Mentions-for-Verga-Strubell/48f786f66eb846012ceee822598a335d0388f034), where they use gold entities as inputs for relation prediction.

- ACE05-Event
  ```
  2020-05-25 17:05:14,044 - INFO - allennlp.commands.evaluate - _ner_f1: 0.906369532679145
  2020-05-25 17:05:14,044 - INFO - allennlp.commands.evaluate - _trig_id_f1: 0.735042735042735
  2020-05-25 17:05:14,044 - INFO - allennlp.commands.evaluate - trig_class_f1: 0.7029914529914529
  2020-05-25 17:05:14,044 - INFO - allennlp.commands.evaluate - _arg_id_f1: 0.5414364640883979
  2020-05-25 17:05:14,044 - INFO - allennlp.commands.evaluate - arg_class_f1: 0.5130228887134964
  ```

## Making predictions on existing datasets

To make a prediction, you can use `allennlp predict`. For example, to make a prediction with the pretrained scierc model, you can do:

```
allennlp predict pretrained/scierc.tar.gz \
    data/scierc/processed_data/json/test.json \
    --predictor dygie \
    --include-package dygie \
    --use-dataset-reader \
    --output-file predictions/scierc-test.jsonl \
    --cuda-device 0
```

**Caveat**: Models trained to predict coreference clusters need to make predictions on a whole document at once. This can cause memory issues. To get around this there are two options:

- Make predictions using a model that doesn't do coreference propagation. These models predict a sentence at a time, and shouldn't run into memory issues. Use the "lightweight" models to avoid this. To train your own coref-free model, set [coref loss weight](https://github.com/dwadden/dygiepp/blob/master/training_config/scierc_working_example.jsonnet#L50) to 0 in the relevant training config.
- Split documents up into smaller chunks (5 sentences should be safe), make predictions using a model with coref prop, and stitch things back together.

See the [docs](https://allenai.github.io/allennlp-docs/api/commands/predict/) for more prediction options.

### Relation extraction evaluation metric

Following [Li and Ji (2014)](https://www.semanticscholar.org/paper/Incremental-Joint-Extraction-of-Entity-Mentions-and-Li-Ji/ab3f1a4480c1ef8409d1685889600f7efb76af24), we consider a predicted relation to be correct if "its relation type is
correct, and the head offsets of two entity mention arguments are both correct".

In particular, we do *not* require the types of the entity mention arguments to be correct, as is done in some work (e.g. [Zhang et al. (2017)](https://www.semanticscholar.org/paper/End-to-End-Neural-Relation-Extraction-with-Global-Zhang-Zhang/ee13e1a3c1d5f5f319b0bf62f04974165f7b0a37)). We welcome a pull request that implements this alternative evaluation metric. Please open an issue if you're interested in this.


## Working with new datasets

Follow the instructions as described in [Formatting a new dataset](DATA.md#formatting-a-new-dataset).

### Making predicitons on a new dataset

To make predictions on a new, unlabeled dataset:

1. Download the [pretrained model](#pretrained-models) that most closely matches your text domain.
2. Make predictions the same way as with the [existing datasets](#making-predictions-on-existing-datasets):
```
allennlp predict pretrained/[name-of-pretrained-model].tar.gz \
    [input-path] \
    --predictor dygie \
    --include-package dygie \
    --use-dataset-reader \
    --output-file [output-path] \
    --cuda-device [cuda-device]
```

### Training a model on a new (labeled) dataset

Follow the process described in [Training a model](#training-a-model), but adjusting the input and output file paths as appropriate.

# Contact

For questions or problems with the code, create a GitHub issue (preferred) or email `dwadden@cs.washington.edu`.
