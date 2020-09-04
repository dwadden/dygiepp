# DyGIE++

**USABILITY**: This branch is an ongoing update of the DyGIE++ code to play nicely with AllenNLP V1 (and by extension, Huggingface Transformers). The relation, named entity, and coreference modules should work, but event extraction will break. The performance may be a point or two lower on relation extraction due to tuning issues. I will update this status as changes are made.

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

This library relies on [AllenNLP](https://allennlp.org) and uses AllenNLP shell [commands](https://docs.allennlp.org/master/#package-overview) to kick off training, evaluation, and testing.


## Training a model

*Warning about coreference resolution*: The coreference code will break on sentences with only a single token. If you have these in your dataset, either get rid of them or deactivate the coreference resolution part of the model.

We rely on [Allennlp train](https://docs.allennlp.org/master/api/commands/train/) to handle model training. The `train` command takes a configuration file as an argument, and initializes a model based on the configuration, and serializes the traing model. More details on the configuration process for DyGIE can be found in [doc/config.md](doc/config.md).

To train a model, enter `bash scripts/train.sh [config_name]` at the command line, where the `config_name` is the name of a file in the `training_config` directory. For instance, to train a model using the `scierc.jsonnet` config, you'd enter

```bash
bash scripts/train.sh scierc
```

The resulting model will go in `models/scierc`. For more information on how to modify training configs (e.g. to change the GPU used for training), see [config.md](doc/config.md).

Information on preparing specific training datasets is below. For more information on how to create training batches that utilize GPU resources efficiently, see [model.md](doc/model.md)


### SciERC

To train a model for named entity recognition, relation extraction, and coreference resolution on the SciERC dataset:

- **Download the data**. From the top-level folder for this repo, enter `bash ./scripts/data/get_scierc.sh`. This will download the scierc dataset into a folder `./data/scierc`
- **Train the model**. Enter `bash scripts/train.sh scierc`.
- To train a "lightweight" version of the model that doesn't do coreference propagation and uses a context width of 1, do `bash scripts/train.sh scierc_lightweight` instead. More info on why you'd want to do this in the section on [making predictions](#making-predictions).


### GENIA

The steps are similar to SciERC.

- **Download the data**. From the top-level folder for this repo, enter `bash ./scripts/data/get_genia.sh`.
  - **NOTE**: A few of the documents get thrown out because they have sentences containing empty strings, which breaks things (see [data.md](doc/data.md)). I'd welcome a PR that fixes these!
- **Train the model**. Enter `bash scripts/train genia`.
- As with SciERC, we also offer a "lightweight" version with a context width of 1 and no coreference propagation.


### ChemProt

The [ChemProt](https://biocreative.bioinformatics.udel.edu/news/corpora/chemprot-corpus-biocreative-vi/) corpus contains entity and relation annotations for drug / protein interaction. The ChemProt preprocessing requires a separate environment:

```shell
conda deactivate
conda create --name chemprot-preprocess python=3.7
conda activate chemprot-preprocess
pip install -r scripts/data/chemprot/requirements.txt
```

Then, follow these steps:

- **Get the data**.
  - Run `bash ./scripts/data/get_chemprot.sh`. This will download the data and process it into the DyGIE input format.
    - NOTE: This is a quick-and-dirty script that skips entities whose character offsets don't align exactly with the tokenization produced by SciSpacy. We lose about 10% of the named entities and 20% of the relations in the dataset as a result.
  - Switch back to your DyGIE environment.
  - For a quick spot-check to see how much of the data was lost:
    ```
    python scripts/data/chemprot/03_spot_check.py
    ```
  - Collate the data:
    ```
    mkdir -p data/chemprot/collated_data

    python scripts/data/shared/collate.py \
      data/chemprot/processed_data \
      data/chemprot/collated_data \
      --train_name=training \
      --dev_name=development
    ```
- **Train the model**. TODO need to add this. Enter `bash scripts/train chemprot`.


### ACE05 (ACE for entities and relations)

#### Creating the dataset

For more information on ACE relation and event preprocessing, see [doc/data.md](doc/data.md) and [this issue](https://github.com/dwadden/dygiepp/issues/11).

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

Enter `bash scripts/train ace05_relation`. A model trained this way will not reproduce the numbers in the paper. We're in the process of debugging and will update.

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
You can see the available flags by calling `parse_ace_event.py -h`. For detailed descriptions, see [data.md](doc/data.md). The results will go in `./data/ace-event/processed-data/[output-name]`. We require an output name because you may want to preprocess the ACE data multiple times using different flags. For default preprocessing settings, you could do:
```
python ./scripts/data/ace-event/parse_ace_event.py default-settings
```
Now `conda deactivate` the `ace-event-preprocess` environment and re-activate your modeling environment.

Finally, collate the version of the dataset you just created. For instance, continuing the example above,
```
mkdir -p data/ace-event/collated-data/default-settings/json

python scripts/data/shared/collate.py \
  data/ace-event/processed-data/default-settings/json \
  data/ace-event/collated-data/default-settings/json \
  --file_extension json
```

#### Training the model

To train on the data preprocessed with default settings, enter `bash scripts/train.sh ace05_event`. A model trained in this fashion will reproduce (within 0.1 F1 or so) the results in Table 4 of the paper. To train on a different version, modify `training_config/ace05_event.jsonnet` to point to the appropriate files.

To reproduce the results in Table 1 requires training an ensemble model of 4 trigger detectors. The basic process is as follows:

- Merge the ACE event train + dev data, then create 4 new train / dev splits.
- Train a separate trigger detection model on each split. To do this, modify `training_config/ace05_event.jsonnet` by setting
  ```jsonnet
  model +: {
    modules +: {
      events +: {
        loss_weights: {
          trigger: 1.0,
          arguments: 0.5
        }
      }
    }
  }
  ```
- Make trigger predictions using a majority vote of the 4 ensemble models.
- Use these predicted triggers when making event argument predictions based on the event argument scores output by the model saved at `models/ace05_event`.

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
  data/scierc/normalized_data/json/test.json \
  --cuda-device 2 \
  --include-package dygie
```
To evaluate a model you trained on the SciERC data, you could do
```shell
allennlp evaluate \
  models/scierc/model.tar.gz \
  data/scierc/normalized_data/json/test.json \
  --cuda-device 2  \
  --include-package dygie \
  --output-file models/scierc/metrics_test.json
```

## Pretrained models

We have versions of DyGIE++ trained on SciERC and GENIA available. There are two versions:
- The "lightweight" versions don't use coreference propagation, and use a context window of 1. If you've got a new dataset and you just want to get some reasonable predictions, use these.
- The "full" versions use coreference propagatation and a context window of 3. Use these if you need to squeeze out another F1 point or two. These models take longer to run, and they may break if they're given inputs that are too long.

### Downloads

Run `scripts/pretrained/get_dygiepp_pretrained.sh` to download all the available pretrained models to the `pretrained` directory. If you only want one model, here are the download links.

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

```bash
allennlp predict pretrained/scierc.tar.gz \
    data/scierc/processed_data/json/test.json \
    --predictor dygie \
    --include-package dygie \
    --use-dataset-reader \
    --output-file predictions/scierc-test.jsonl \
    --cuda-device 0 \
    --silent
```

The predictions include the predict labels, as well as logits and softmax scores. For more information see, [docs/data.md](docs/data.md).

**Caveat**: Models trained to predict coreference clusters need to make predictions on a whole document at once. This can cause memory issues. To get around this there are two options:

- Make predictions using a model that doesn't do coreference propagation. These models predict a sentence at a time, and shouldn't run into memory issues. Use the "lightweight" models to avoid this. To train your own coref-free model, set [coref loss weight](https://github.com/dwadden/dygiepp/blob/master/training_config/scierc_working_example.jsonnet#L50) to 0 in the relevant training config.
- Split documents up into smaller chunks (5 sentences should be safe), make predictions using a model with coref prop, and stitch things back together.

See the [docs](https://allenai.github.io/allennlp-docs/api/commands/predict/) for more prediction options.

### Relation extraction evaluation metric

Following [Li and Ji (2014)](https://www.semanticscholar.org/paper/Incremental-Joint-Extraction-of-Entity-Mentions-and-Li-Ji/ab3f1a4480c1ef8409d1685889600f7efb76af24), we consider a predicted relation to be correct if "its relation type is
correct, and the head offsets of two entity mention arguments are both correct".

In particular, we do *not* require the types of the entity mention arguments to be correct, as is done in some work (e.g. [Zhang et al. (2017)](https://www.semanticscholar.org/paper/End-to-End-Neural-Relation-Extraction-with-Global-Zhang-Zhang/ee13e1a3c1d5f5f319b0bf62f04974165f7b0a37)). We welcome a pull request that implements this alternative evaluation metric. Please open an issue if you're interested in this.


## Working with new datasets

Follow the instructions as described in [Formatting a new dataset](doc/data.md#formatting-a-new-dataset).

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

A couple tricks to make things run smoothly:

1. If you're predicting on a big dataset, you probably want to load it lazily rather than loading the whole thing in before predicting. To accomplish this, add the following flag to the above command:
  ```
  --overrides "{'dataset_reader' +: {'lazy': true}}"
  ```
2. If the model runs out of GPU memory on a given prediction, it will warn you and continue with the next example rather than stopping entirely. This is less annoying than the alternative. Examples for which predictions failed will still be written to the specified `jsonl` output, but they will have an additional field `{"_FAILED_PREDICTION": true}` indicating that the model ran out of memory on this example.
3. The `dataset` field in the dataset to be predicted must match one of the `dataset`s on which the model was trained; otherwise, the model won't know which labels to apply to the predicted data. I'd welcome a PR to allow the user to ask for predictions for multiple different label namespaces.

### Training a model on a new (labeled) dataset

Follow the process described in [Training a model](#training-a-model), but adjusting the input and output file paths as appropriate.

# Contact

For questions or problems with the code, create a GitHub issue (preferred) or email `dwadden@cs.washington.edu`.
