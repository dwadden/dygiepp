# DyGIE++

Implements the model described in the paper [Entity, Relation, and Event Extraction with Contextualized Span Representations](https://www.semanticscholar.org/paper/Entity%2C-Relation%2C-and-Event-Extraction-with-Span-Wadden-Wennberg/fac2368c2ec81ef82fd168d49a0def2f8d1ec7d8).

## Table of Contents
- [Updates](#updates)
- [Project status](#project-status)
- [Issues](#issues)
- [Dependencies](#dependencies)
- [Model training](#training-a-model)
- [Model evaluation](#evaluating-a-model)
- [Pretrained models](#pretrained-models)
- [Making predictions on existing datasets](#making-predictions-on-existing-datasets)
- [Working with new datasets](#working-with-new-datasets)
- [Contact](#contact)

See the `doc` folder for documentation with more details on the [data](doc/data.md), [model implementation and debugging](doc/model.md), and [model configuration](doc/config.md).


## Updates

**December 2021**: A couple nice additions thanks to PR's from contributors:

- There is now a script to convert [BRAT-formatted](https://github.com/nlplab/brat) annotations to DyGIE. See [here](https://github.com/dwadden/dygiepp/blob/master/doc/data.md#converting-data-labeled-with-brat) for more details. Thanks to @serenalotreck for this feature.
- There are Spacy bindings for DyGIE entity and relation extraction; see the section on  [Spacy bindings](#spacy-bindings). Thanks to @e3oroush for this feature.

**April 2021**: We've added data and models for the MECHANIC dataset, presented in the NAACL 2021 paper [Extracting a Knowledge Base of Mechanisms from COVID-19 Papers](https://www.semanticscholar.org/paper/c4ce6aca9aed41d57d588674484932e0c2cd3547).

- [Download the dataset](https://ai2-s2-mechanic.s3-us-west-2.amazonaws.com/data/data.zip)
- [Download the "coarse" model](https://ai2-s2-mechanic.s3-us-west-2.amazonaws.com/models/mechanic-coarse.tar.gz)
- [Download the "granular" model](https://ai2-s2-mechanic.s3-us-west-2.amazonaws.com/models/mechanic-granular.tar.gz)

You can also get the data by running `bash scripts/data/get_mechanic.sh`, which will put the data in `data/mechanic`.

After moving the models to the `pretrained` folder, you can make predictions like this:

```bash
allennlp predict \
  pretrained/mechanic-coarse.tar.gz \
  data/mechanic/coarse/test.json \
  --predictor dygie \
  --include-package dygie \
  --use-dataset-reader \
  --output-file predictions/covid-coarse.jsonl \
  --cuda-device 0 \
  --silent
```


## Project status

This branch used to be named `allennlp-v1`, and it has been made the new `master`. It's compatible with new version of AllenNLP, and the model configuration process has been simplified. I'd recommend using this branch for all future work. If for some reason you need the older version of the code, it's on the branch [emnlp-2019](https://github.com/dwadden/dygiepp/tree/emnlp-2019).

Unfortunately, I don't have the bandwidth at this point to add additional features. But please create a new issue if you have problems with:
- Reproducing the results reported in the README.
- Making predictions on a new dataset using pre-trained models.
- Training your own model on a new dataset.

See [below](#issues) for guidelines on creating an issue.

There are a number of ways this code could be improved, and I'd definitely welcome pull requests. If you're interested, see [contributions.md](doc/contributions.md) for a list of ieas.

### Submit a model!

If you have a DyGIE model that you've trained on a new dataset, feel free to upload it [here](https://docs.google.com/forms/d/e/1FAIpQLSdwws7zVAqF15-kBqkKBupymWe0ASkXhODH8yomYkRDy5DvCw/viewform?usp=sf_link) and I'll add it to the collection of pre-trained models.

## Issues

If you're unable to run the code, feel free to create an issue. Please do the following:

- Confirm that you've set up a Conda environement exactly as in the [Dependencies](#dependencies) section below. I can only offer support if you're running code within this environment.
- Specify any commands you used to download pretrained models or to download / preprocess data. Please enclose the code in code blocks, for instance:
  ```bash
  # Download pretrained models.

  bash scripts/pretrained/get_dygiepp_pretrained.sh
  ```
- Share the command that you ran to cause the issue, for instance:
  ```
  allennlp evaluate \
  pretrained/scierc.tar.gz \
  data/scierc/normalized_data/json/test.json \
  --cuda-device 2 \
  --include-package dygie
  ```
- If you're using your own dataset, attach a minimal example of the data which, when given as input, causes the error you're seeing. This could be, for instance, a single line form a `.jsonl` file.
- Include the full error message that you're getting.


## Dependencies

Clone this repository and navigate the the root of the repo on your system. Then execute:

```
conda create --name dygiepp python=3.7
pip install -r requirements.txt
conda develop .   # Adds DyGIE to your PYTHONPATH
```

This library relies on [AllenNLP](https://allennlp.org) and uses AllenNLP shell [commands](https://docs.allennlp.org/master/#package-overview) to kick off training, evaluation, and testing.

If you run into an issue installing `jsonnet`, [this issue](https://github.com/allenai/allennlp/issues/2779) may prove helpful.

### Docker build
A `Dockerfile` is provided with the Pytorch + CUDA + CUDNN base image for a full-stack GPU install.
It will create conda environments `dygiepp` for modeling & `ace-event-preprocess` for ACE05-Event preprocessing.

By default the build downloads datasets and dependencies for all tasks.
This takes a long time and produces a large image, so you will want to comment out unneeded datasets/tasks in the Dockerfile.

- Comment out unneeded task sections in `Dockerfile`.
- Build container: `docker build --tag dygiepp:dev <dygiepp-repo-dirpath>`
- Run the container interactively, mount this project dir to /dygiepp/: `docker run --gpus all -it --ipc=host -v <dygiepp-repo-dirpath>:/dygiepp/ --name dygiepp dygiep:dev`

**NOTE**: This Dockerfile was added in a PR from a contributor. I haven't tested it, so it's not "officially supported". More PR's are welcome, though.

## Training a model

*Warning about coreference resolution*: The coreference code will break on sentences with only a single token. If you have these in your dataset, either get rid of them or deactivate the coreference resolution part of the model.

We rely on [Allennlp train](https://docs.allennlp.org/master/api/commands/train/) to handle model training. The `train` command takes a configuration file as an argument, and initializes a model based on the configuration, and serializes the traing model. More details on the configuration process for DyGIE can be found in [doc/config.md](doc/config.md).

To train a model, enter `bash scripts/train.sh [config_name]` at the command line, where the `config_name` is the name of a file in the `training_config` directory. For instance, to train a model using the `scierc.jsonnet` config, you'd enter

```bash
bash scripts/train.sh scierc
```

The resulting model will go in `models/scierc`. For more information on how to modify training configs (e.g. to change the GPU used for training), see [config.md](doc/config.md).

Information on preparing specific training datasets is below. For more information on how to create training batches that utilize GPU resources efficiently, see [model.md](doc/model.md).
Hyperparameter optimization search is implemented using [Optuna](https://optuna.readthedocs.io), see [model.md](doc/model.md).

### SciERC

To train a model for named entity recognition, relation extraction, and coreference resolution on the SciERC dataset:

- **Download the data**. From the top-level folder for this repo, enter `bash ./scripts/data/get_scierc.sh`. This will download the scierc dataset into a folder `./data/scierc`
- **Train the model**. Enter `bash scripts/train.sh scierc`.
- To train a "lightweight" version of the model that doesn't do coreference propagation and uses a context width of 1, do `bash scripts/train.sh scierc_lightweight` instead. More info on why you'd want to do this in the section on [making predictions](#making-predictions).


### GENIA

The steps are similar to SciERC.

- **Download the data**. From the top-level folder for this repo, enter `bash ./scripts/data/get_genia.sh`.
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
  - Collate the data:
    ```
    mkdir -p data/chemprot/collated_data

    python scripts/data/shared/collate.py \
      data/chemprot/processed_data \
      data/chemprot/collated_data \
      --train_name=training \
      --dev_name=development
   - For a quick spot-check to see how much of the data was lost, you can run:
    ```
    python scripts/data/chemprot/03_spot_check.py
    ```   ```
- **Train the model**. Enter `bash scripts/train chemprot`.


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

The results will go in `./data/ace05/collated-data`. The intermediate files will go in `./data/ace05/raw-data`.

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
python -m spacy download en_core_web_sm
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


### MECHANIC

You can get the dataset by running `bash scripts/data/get_mechanic.sh`. For detailed training instructions, see the [DyGIE-COFIE](https://github.com/AidaAmini/DyGIE-COFIE) repo.


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

A number of models are available for download. They are named for the dataset they are trained on. "Lightweight" models are models trained on datasets for which coreference resolution annotations were available, but we didn't use them. This is "lightweight" because coreference resolution is expensive, since it requires predicting cross-sentence relationships between spans.

If you want to use one of these pretrained models to make predictions on a new dataset, you need to set the `dataset` field for the instances in your new dataset to match the name of the `dataset` the model was trained on. For example, to make predictions using the pretrained SciERC model, set the `dataset` field in your new instances to `scierc`. For more information on the `dataset` field, see [data.md](doc/data.md).

To download all available models, run `scripts/pretrained/get_dygiepp_pretrained.sh`. Or, click on the links below to download only a single model.

### Available models

Below are links to the available models, followed by the name of the `dataset` the model was trained on.

- [SciERC](https://s3-us-west-2.amazonaws.com/ai2-s2-research/dygiepp/master/scierc.tar.gz): `scierc`
- [SciERC lightweight](https://s3-us-west-2.amazonaws.com/ai2-s2-research/dygiepp/master/scierc-lightweight.tar.gz): `scierc`
- [GENIA](https://s3-us-west-2.amazonaws.com/ai2-s2-research/dygiepp/master/genia.tar.gz): `genia`
- [GENIA lightweight](https://ai2-s2-research.s3-us-west-2.amazonaws.com/dygiepp/master/genia-lightweight.tar.gz): `genia`
- [ChemProt (lightweight only)](https://ai2-s2-research.s3-us-west-2.amazonaws.com/dygiepp/master/chemprot.tar.gz): `chemprot`
- [ACE05 relation](https://ai2-s2-research.s3-us-west-2.amazonaws.com/dygiepp/master/ace05-relation.tar.gz): `ace05`
- [ACE05 event](https://ai2-s2-research.s3-us-west-2.amazonaws.com/dygiepp/master/ace05-event.tar.gz): `ace-event`
- [MECHANIC "coarse"](https://ai2-s2-mechanic.s3-us-west-2.amazonaws.com/models/mechanic-coarse.tar.gz) `None`
- [MECHANIC "granular"](https://ai2-s2-mechanic.s3-us-west-2.amazonaws.com/models/mechanic-granular.tar.gz) `covid-event`


### Spacy bindings

DyGIE can now be called from Spacy! For example usage, see the [demo notebook](notebooks/spacy-interface-example.ipynb). This feature was added by a contributor; please tag @e3oroush on related issues.


### Performance of pretrained models

- SciERC
  ```
  "_scierc__ner_f1": 0.6846741045214326,
  "_scierc__relation_f1": 0.46236559139784944
  ```

- SciERC lightweight
  ```
  "_scierc__ner_f1": 0.6717245404143566,
  "_scierc__relation_f1": 0.4670588235294118
  ```

- GENIA
  ```
  "_genia__ner_f1": 0.7713070807912737
  ```

- GENIA lightweight
  And the lightweight version:
  ```
  "_genia__ner_f1": 0.7690401296349251
  ```

- ChemProt
  ```
  "_chemprot__ner_f1": 0.9059113300492612,
  "_chemprot__relation_f1": 0.5404867256637169
  ```
  Note that we're doing span-level evaluation using predicted entities. We're also evaluating on all ChemProt relation classes, while the official task only evaluates on a subset (see [Liu et al.](https://www.semanticscholar.org/paper/Attention-based-Neural-Networks-for-Chemical-Liu-Shen/a6261b278d1c2155e8eab7ac12d924fc2207bd04) for details). Thus, our relation extraction performance is lower than, for instance, [Verga et al.](https://www.semanticscholar.org/paper/Simultaneously-Self-Attending-to-All-Mentions-for-Verga-Strubell/48f786f66eb846012ceee822598a335d0388f034), where they use gold entities as inputs for relation prediction.

- ACE05-Relation
  ```
  "_ace05__ner_f1": 0.8634611855386309,
  "_ace05__relation_f1": 0.6484907497565725,
  ```

- ACE05-Event
  ```
  "_ace-event__ner_f1": 0.8927209418006965,
  "_ace-event_trig_class_f1": 0.6998813760379595,
  "_ace-event_arg_class_f1": 0.5,
  "_ace-event__relation_f1": 0.5514950166112956
  ```

## Making predictions on existing datasets

To make a prediction, you can use `allennlp predict`. For example, to make a prediction with the pretrained scierc model, you can do:

```bash
allennlp predict pretrained/scierc.tar.gz \
    data/scierc/normalized_data/json/test.json \
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

### Making predictions on a new dataset

To make predictions on a new, unlabeled dataset:

1. Download the [pretrained model](#pretrained-models) that most closely matches your text domain.
2. Make sure that the `dataset` field for your new dataset matches the label namespaces for the pretrained model. See [here](doc/model.md#multi-dataset-training) for more on label namespaces. To view the available label namespaces for a pretrained model, use [print_label_namespaces.py](scripts/debug/print_label_namespaces.py).
3. Make predictions the same way as with the [existing datasets](#making-predictions-on-existing-datasets):
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
3. The `dataset` field in the dataset to be predicted must match one of the `dataset`s on which the model was trained; otherwise, the model won't know which labels to apply to the predicted data.

### Training a model on a new (labeled) dataset

Follow the process described in [Training a model](#training-a-model), but adjusting the input and output file paths as appropriate.

# Contact

For questions or problems with the code, create a GitHub issue (preferred) or email `dwadden@cs.washington.edu`.
