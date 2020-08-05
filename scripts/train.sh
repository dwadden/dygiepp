# USAGE: `bash train.sh [config_name]`
#
# The `config_name` is the name of one of the `jsonnet` config files in the
# `training_config` directory, for instance `scierc`. The result of training
# will be placed under `models/[config_name]`.

config_name=$1

allennlp train "training_config/${config_name}.jsonnet" \
    --serialization-dir "models/${config_name}" \
    --include-package dygie
