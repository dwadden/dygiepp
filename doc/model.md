# Model

We include some notes and common modeling issues here. This document will grow over time.

## Debugging

Debugging by running `allennlp train` or `allennlp predict` isn't optimal, because the model takes more than 10 seconds just to initialize. To speed up the debugging loop, there's a script [debug_forward_pass.py](../scripts/debug/debug_forward_pass.py) that will run a forward pass for you without doing all the initialization logic, and without loading in the BERT embeddings. See the script for usage information.


## Batching and batch size

AllenNLP has a data structure to represent an [Instance](https://guide.allennlp.org/reading-data#1), which it defines as "the atomic unit of prediction in machine learning". For example, in sentiment classification, an `Instance` would usually be a single sentence.

`Instance`s are slighly awkward for DyGIE++, because
three tasks (named entity tagging, relation extraction, event extraction) are *within-sentence*, making a sentence the natural unit for an `Instance`. However, coreference resolution is *cross-sentence*, making a *document* the natural unit for an `Instance`.

The choice we have made is to model an `Instance` as a *document*. By default, we use a batch size of 1, which means that each minibatch during training is a single *document*. We make this choice because it's conceptually the simplest, but it is not optimal in some circumstances. We describe these and offer some solutions. These solutions mostly involve doing data preprocessing *outside the modeling code*; this keeps the (already somewhat confusing) modeling code as simple as possible.

--------------------

- **Problem**: If you're not doing coreference resolution, then it's wasteful to have minibatches with sentences of widely varying lengths. Instead, you should create minibatches of similar-length sentences from different documents.
- **Solution**: Our solution is as follows:
  - "Collate" the dataset into "psuedo-documents" containing sentences of similar length. Keep track of the original document that each sentence came from. Users may write their own script, or use [collate.py](../scripts/data/collate.py) to accomplish this.
  - Run training / prediction / whatever else.
  - For predictions, "de-collate" the predictions to recover the original documents. TODO write a "de-collating script".
- **Details**: It's up to the user to collate the sentences in a way that makes good use of GPU memory. In general, GPU usage for DyGIE++ scales with the number of spans in the document, which scales as the *square* of the sentence length. Thus, `collate.py` takes `max_spans_per_doc` as input. We calculate the number of spans per doc as `n_sentences * (longest_sentence_length ** 2)`. We've found that setting `max_spans_per_doc=50000` creates batches that utilize GPU effectively. However, we have not explored this exhaustively and we welcome feedback and PR's.

--------------------

- **Problem**: You're doing coreference resolution, but the documents in your dataset are short; using a batch size of 1 wastes GPU memory.
- **Solution**: We're working on writing a data loader that will handle this for you.

--------------------

- **Problem**: You're doing coreference resolution, and your documents are too long to fit in memory.
- **Solution**: Split the documents as a preprocessing step, run the model, and merge in post-processing. We don't have a script to do this and would welcome a PR.
