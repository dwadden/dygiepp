# Make a prediction on the SciERC test set using the pretrained SciERC model.

# Make prediction directory if it doesn't exist.
if [ ! -d "./predictions" ]
then
    mkdir "./predictions"
fi

data_root="./data/scierc/processed_data/json"

python ./dygie/commands/predict_dygie.py \
    ./pretrained/scierc.tar.gz \
    ./$data_root/test.json \
    ./predictions/scierc_test.json \
    -1
