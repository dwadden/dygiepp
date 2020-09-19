# Download the raw and processed scierc dataset and put them in the `data`
# folder.
# Usage: From main project folder, run `bash scripts/data/get_scierc.sh`

out_dir=data/scierc
mkdir $out_dir

# Download.
wget http://nlp.cs.washington.edu/sciIE/data/sciERC_raw.tar.gz -P $out_dir
wget http://nlp.cs.washington.edu/sciIE/data/sciERC_processed.tar.gz -P $out_dir

# Decompress.
tar -xf $out_dir/sciERC_raw.tar.gz -C $out_dir
tar -xf $out_dir/sciERC_processed.tar.gz -C $out_dir

# Clean up.
rm $out_dir/*.tar.gz

# Normalize by adding dataset name.
python scripts/data/shared/normalize.py \
    $out_dir/processed_data/json \
    $out_dir/normalized_data/json \
    --file_extension=json \
    --max_tokens_per_doc=0 \
    --dataset=scierc

# Collate for more efficient non-coref training.
python scripts/data/shared/collate.py \
    $out_dir/processed_data/json \
    $out_dir/collated_data/json \
    --file_extension=json \
    --dataset=scierc
