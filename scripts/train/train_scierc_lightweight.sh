# Train DyGIE++ model on the scierc data set.
# Usage: bash scripts/train/train_scierc.sh [gpu-id]
# gpu-id can be an integer GPU ID, or -1 for CPU.

experiment_name="scierc-lightweight"
config_file="./training_config/scierc_lightweight.jsonnet"

# Train model.
    allennlp train $config_file \
    --serialization-dir ./models/$experiment_name \
    --force \
    --include-package dygie \
