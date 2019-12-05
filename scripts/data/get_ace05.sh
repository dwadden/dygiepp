# Run preprocessing scripts and rearrange to put data in the right place.

raw_path=$1
original_path=$(pwd)
process_path=scripts/data/ace05/preprocess
out_path=data/ace05
mkdir -p $out_path/processed-data/json
mkdir -p $out_path/raw-data

# Run the preprocessing scripts.
cp -r $raw_path/*/English $process_path
cd $process_path
zsh run.zsh
python ace2json.py

# Move all the intermediate files to the `raw-data` folder if the output path.
cd $original_path
for folder in corpus English fixed result text
do
    mv $process_path/$folder $out_path/raw-data
done
