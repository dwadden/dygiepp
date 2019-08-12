Things needed for code release

## README instructions

- [ ] Instructions on how to use a pre-trained model to make predictions on a new dataset.
- [ ] Instructions on how to train a new model.

## Data

- [ ] The scierc dataset. This can come as part of the package.
- [ ] Script to download and preprocess the WLPC dataset.
- [ ] Script to download and preprocess the GENIA dataset.
- [ ] Script to download and preprocess the ACE relation dataset.
- [ ] Script to download and preprocess the ACE event dataset.

## Pretrained models

For each data set, let's release a single pre-trained model that does reasonable well on all the IE tasks for that dataset. Need to think about the right way to release these models. Should they be part of the package or be in a separate download?

## Config files

Clean up the config file structure and include configs to re-run the experiments in the paper.

## Training scripts

Include scripts to reproduce (approximately) the results in the paper.

## Demo

Maybe have a web demo?
