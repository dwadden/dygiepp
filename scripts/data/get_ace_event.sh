# Usage: `bash ./scripts/data/get_ace_event.sh [path-to-ace-data]`.

ace_dir=$1

# Collect all files into one folder.
bash ./scripts/data/ace-event/collect_ace_event.sh $ace_dir

# Run parsing script.
python ./scripts/data/ace-event/parse_ace_event.py
