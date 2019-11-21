# Evaluate performance of pretrained scierc model on scierc test set.

data_root="./data/scierc/processed_data/json"
cuda_device=$1

allennlp evaluate \
    ./pretrained/scierc.tar.gz \
    $data_root/test.json \
    --cuda-device $cuda_device \
    --include-package dygie
