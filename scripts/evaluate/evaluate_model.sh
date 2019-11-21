# Evaluate performance of a pretrained DyGIE model.
# Usage: `bash scripts/pretrained/evaluate_dygiepp_pretrained.sh [model-file] [data-path] [cuda-device]`.

model_file=$1
data_path=$2
cuda_device=$3

allennlp evaluate \
    $model_file \
    $data_path \
    --cuda-device $cuda_device \
    --include-package dygie
