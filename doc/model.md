# Model

We include some notes and common modeling issues here.

## Table of Contents
- [Debugging](#debugging)
- [Batching and batch size](#batching-and-batch-size)
- [Multi-dataset training](#multi-dataset-training)
- [Hyperparameter optimization](#hyperparameter-optimization)
- [Reproducibility and nondeterminism](#reproducibility-and-nondeterminism)

## Debugging

Debugging by running `allennlp train` or `allennlp predict` isn't optimal, because the model takes more than 10 seconds just to initialize. To speed up the debugging loop, there's a script [debug_forward_pass.py](../scripts/debug/debug_forward_pass.py) that will run a forward pass for you without doing all the initialization logic, and without loading in the BERT embeddings. See the script for usage information.

### Common problems encountered during debugging

- `nan`'s in gradients or parameters: This may be due to characters that the tokenizer doesn't recognize. See [this issue](https://github.com/allenai/allennlp/issues/4612) for more details. Before doing any more extensive debugging, check your input documents for weird unicode characters.


## Batching and batch size

AllenNLP has a data structure to represent an [Instance](https://guide.allennlp.org/reading-data#1), which it defines as "the atomic unit of prediction in machine learning". For example, in sentiment classification, an `Instance` would usually be a single sentence.

`Instance`s are slighly awkward for DyGIE++, because
three tasks (named entity tagging, relation extraction, event extraction) are *within-sentence*, making a sentence the natural unit for an `Instance`. However, coreference resolution is *cross-sentence*, making a *document* the natural unit for an `Instance`.

The choice we have made is to model an `Instance` as a *document*. By default, we use a batch size of 1, which means that each minibatch during training is a single *document*. We make this choice because it's conceptually the simplest, but it is not optimal in some circumstances. We describe these and offer some solutions. These solutions mostly involve doing data preprocessing *outside the modeling code*; this keeps the (already somewhat confusing) modeling code as simple as possible.

--------------------

- **Problem**: If you're not doing coreference resolution, then it's wasteful to have minibatches with sentences of widely varying lengths. Instead, you should create minibatches of similar-length sentences from different documents.
- **Solution**: Our solution is as follows:
  - "Collate" the dataset into "psuedo-documents" containing sentences of similar length. Keep track of the original document that each sentence came from. Users may write their own script, or use [collate.py](../scripts/data/shared/collate.py) to accomplish this.
  - Run training / prediction / whatever else.
  - For prediction, "un-collate" the predictions to recover the original documents using [uncollate.py](../scripts/data/shared/uncollate.py)
- **Details**: It's up to the user to collate the sentences in a way that makes good use of GPU memory. The `collate` script has two options to help control this.
  - `max_spans_per_doc`: In general, GPU usage for DyGIE++ scales with the number of spans in the document, which scales as the *square* of the sentence length. Thus, `collate.py` takes `max_spans_per_doc` as input. We calculate the number of spans per doc as `n_sentences * (longest_sentence_length ** 2)`. We've found that setting `max_spans_per_doc=50000` creates batches that utilize GPU effectively. However, we have not explored this exhaustively and we welcome feedback and PR's.
  - `max_sentences_per_doc`
    - **If If you're training a model**, you probably want to avoid the creation of pseudo-documents containg hundreds of short sentences - even if they'd fit in GPU. Having wildly varying numbers of sentences per batch seemed to do weird things during training, though this is anecdotal. To avoid this, set `max_sentences_per_doc` to some reasonable value. The default of 16 seems safe, though bigger might be OK too.
    - **If you're using an existing model to make predictions**: Just set this to a big number to make the best use of your GPU.

--------------------

- **Problem**: Your documents are too long to fit in memory. You're getting errors like `RuntimeError: CUDA out of memory. Tried to allocate 1.97 GiB (GPU 0; 10.92 GiB total capacity; 7.63 GiB already allocated; 1.46 GiB free; 1.12 GiB cached)`.
- **Solution (datasets *without* coreference annotations)**: If you don't have coreference annotations in your dataset, you can use [collate.py](../scripts/data/shared/collate.py) as described above. If for some reason you don't want to do this, you can use [normalize.py](../scripts/data/shared/normalize.py) to split long documents without shuffling sentences from different documents.
- **Solution (dataset *with* coreference annotations)**: If you have coreference annotations, you can't create minibatches composed of sentences from different documents. The [normalize.py](../scripts/data/shared/normalize.py) should be able to split long documents with coref annotations, but I haven't implemented this yet (I'd welcome a PR that accomplishes this). So, unfortunately, you'll have to write your own script to split long documents.

--------------------

- **Problem**: You're doing coreference resolution, but the documents in your dataset are short; using a batch size of 1 wastes GPU memory.
- **Solution**: We're working on writing a data loader that will handle this for you.


## Multi-dataset training

DyGIE is capable of performing multi-task learning for 4 tasks:
- Named entity recognition (NER)
- Relation extraction
- Event extraction
- Coreference resolution

There may be instances where it is desirable to train different tasks on multiple different datasets - for instance, a coreference resolution model trained on OntoNotes could be used to improve NER predictions on the ACE dataset. It may even be useful to train multiple named entity recognition models sharing the same underlying span representations.

Multi-dataset training with DyGIE is implemented as follows. Each line in the [data](data.md) input to the model must have a `dataset` field. We make the following assumptions:

- The NER, relation, and event label namespaces for different `dataset`s are _disjoint_. A separate model is trained for each dataset.
- The coreference labels for different `dataset`s are _shared_. A single coreference resolution model is trained on all datasets.

As a concrete example: suppose you've decided to train a scientific information extraction model on:

- SciERC (NER, relation, coref)
- GENIA (NER, coref)

The model will create the following label namespaces:

- `scierc__ner`
- `genia__ner`
- `scierc__relation`
- `coref`

Namespaces are named like `[dataset]__[task]`.

A separate module will be created to make predictions for each namespace. All modules will share the same span representations, but will have different task-specific and namespace-specific weights. The model will compute different performance metrics for each namespace, for instance

- `scierc__ner_precision`
- `scierc__ner_recall`
- `scierc__ner_f1`
- `genia__ner_precision`
- `genia__ner_recall`
- `genia__ner_f1`
- etc.

For each task, it will also compute average performance by averaging over the namespaces:

- `MEAN__ner_precision`
- `MEAN__ner_recall`
- `MEAN__ner_f1`

When making predictions, the `dataset` field of the inputs to be predicted must match the `dataset` field of one of the input datasets the model was trained on.

For an example training a model on 4 different datasets, see [multi_dataset_test.sh](../dygie/tests/models/multi_dataset_test.sh).


## Hyperparameter optimization

Tune hyperparameters with [Optuna](https://optuna.org), see example `./scripts/tuning/train_optuna.py`. This feature was contributed in a PR and is not "officially supported", but looks very useful.

Requirements:
- `sqlite3` for storage of trials.
If do not install `sqlite3`, you can use in-memory storage: change `storage` to `None` in `optuna.create_study` (not recommended).

Usage:
- Place Jsonnet configuration file in `./training_config/` dir, see example `./training_config/ace05_event_optuna.jsonnet`:
  - Mask values of hyperparameters with Jsonnet method calling `std.extVar('{param_name}')` with `std.parseInt` for integer or `std.parseJson` for floating-point and other types.
  - Override nested default template values with `+:`, see [config.md](`./doc/config.md`).
- Edit the `objective`-function in `./scripts/tuning/optuna_train.py`:
  - Add [trial suggestions with `suggest` functions](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html).
  - Change `metrics`-argument off `executor` to the relevant optimization goal.
- With the `dygiepp` modeling env activated, run: `python optuna_train.py <CONFIG_NAME>`
- The best config will be dumped at `./training_config/best_<CONFIG_NAME>.json`.

For more details see [Optuna blog](https://medium.com/optuna/hyperparameter-optimization-for-allennlp-using-optuna-54b4bfecd78b).


## Reproducibility and nondeterminism

Some users have observed that results across multiple DyGIE training runs are not reproducible, even though the relevant random seeds are set in the [config template](../training_config/template.libsonnet). This is an underlying issue with [PyTorch](https://pytorch.org/docs/stable/notes/randomness.html). In particular, the Torch `index_select()` function is nondeterministic. As a result so is AllenNLP's `batched_index_select()`, which DyGIE uses in a number of places. I'd welcome a PR to address this, though it's not obvious how this could be done. Thanks to [@khuangaf](https://github.com/khuangaf) for pointing this out.
