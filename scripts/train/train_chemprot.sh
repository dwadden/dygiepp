# Train DyGIE++ model on the chemprot data set.
# Usage: bash scripts/train/train_chemprot.sh [gpu-id]
# gpu-id can be an integer GPU ID, or -1 for CPU.

experiment_name="chemprot"
data_root="./data/chemprot/processed_data"
config_file="./training_config/chemprot.jsonnet"
cuda_device=$1

# Train model.
ie_train_data_path=$data_root/training.jsonl \
    ie_dev_data_path=$data_root/development.jsonl \
    ie_test_data_path=$data_root/test.jsonl \
    cuda_device=$cuda_device \
    allennlp train $config_file \
    --cache-directory $data_root/cached \
    --serialization-dir ./models/$experiment_name \
    --include-package dygie
