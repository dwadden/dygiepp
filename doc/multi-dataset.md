# Multi-dataset training

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

At prediction time, you may request that the model output predictions for all available label namespaces, or only a subset of them (TODO I still need to implement this).
