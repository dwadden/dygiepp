Things needed for code release

## README instructions

- [ ] Instructions on how to use a pre-trained model to make predictions on a new dataset.
- [ ] Instructions on how to train a new model.

## Data

- [ ] ULME Dump all datasets in the gdrive folder that Dave shared. I (Dave) need to make sure that my preprocessing scripts are doing the right thing.
- [ ] Script to download the scierc dataset.
- [ ] Script to download and preprocess the WLPC dataset.
- [ ] Script to download and preprocess the GENIA dataset.
- [ ] Script to download and preprocess the ACE relation dataset.
- [ ] Script to download and preprocess the ACE event dataset.

## Pretrained models

For each data set, let's release a single pre-trained model that does reasonable well on all the IE tasks for that dataset. Need to think about the right way to release these models. Should they be part of the package or be in a separate download?

## Config files

- [ ] Get down to a single `template.jsonnet` file.
    - [ ] ULME I think that we should be able to use `template_dw` for everything. Can you diff my template against yours to make sure this will work and let me know if it looks OK? Then I'll delete the rest of the templates.
- [ ] Simplify the configuration structure. Right now it's pretty opaque...
- [ ] Make it so we have one config file per model.

## Training scripts

Include scripts to reproduce (approximately) the results in the paper.

- [ ] ULME add whatever scripts you used to kick off model training.

## Demo

Maybe have a web demo?
