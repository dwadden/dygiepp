# Train DyGIE++ model on the scierc data set.
# Usage: bash scripts/train_scierc.sh [gpu-id]

experiment_name="scierc"
data_root="./data/scierc/processed_data/json"
config_file="./training_config/ace05_best_ner_bert.jsonnet"
cuda_device=$1

ie_train_data_path=$data_root/train.json \
    ie_dev_data_path=$data_root/dev.json \
    ie_test_data_path=$data_root/test.json \
    cuda_device=$cuda_device \
    allennlp train $config_file \
    --cache-directory $data_root/cached \
    --serialization-dir ./models/$experiment_name \
    --include-package dygie \
    --force
