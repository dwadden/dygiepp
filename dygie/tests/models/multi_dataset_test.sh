# This isn't a formal test; just a training script to make sure that
# multi-dataset training works for a single epoch without breaking.

# Usage (from root of project):
# bash dygie/tests/models/multi_dataset_test.sh

tmpdir=dygie/tests/tmp

if [[ -d $tmpdir ]]
then
    rm -r $tmpdir
fi

mkdir -p $tmpdir

allennlp train "training_config/multi_dataset.jsonnet" \
    --serialization-dir $tmpdir \
    --include-package dygie


# Remove tmpdir once training has finished.
rm -r $tmpdir
