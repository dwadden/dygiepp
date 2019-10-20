# Download the GENIA dataset and preprocess into the JSON form required by
# DyGIE++.
# Usage: From main project folder, run `bash scripts/data/get_genia.sh`
# Most of the challenge here is in aligning the GENIA coref annotations with the
# ner annotations.

# Makes use of code from this repository:
# https://gitlab.com/sutd_nlp/overlapping_mentions/tree/master/data/GENIA.

# Because this script is fairly involved, I put the heavy lifting into
# functions.

out_dir=data/genia
log_dir=$out_dir/logs
raw_dir=$out_dir/raw-data
ner_dir=$raw_dir/GENIAcorpus3.02p

download_raw() {
    # Download the data.
    mkdir $raw_dir

    # Download the entities.
    wget -nv http://www.nactem.ac.uk/GENIA/current/GENIA-corpus/Part-of-speech/GENIAcorpus3.02p.tgz \
        -P $raw_dir

    # Download the coreference data.
    wget -nv http://www.nactem.ac.uk/GENIA/current/GENIA-corpus/Coreference/GENIA_MedCo_coreference_corpus_1.0.tar.gz \
        -P $raw_dir

    # Decompress and cleanup
    mkdir $ner_dir
    tar -xf $raw_dir/GENIAcorpus3.02p.tgz -C $ner_dir
    tar -xf $raw_dir/GENIA_MedCo_coreference_corpus_1.0.tar.gz -C $raw_dir
    rm $raw_dir/*.tgz $raw_dir/*.tar.gz
}

download_sutd() {
    # Download preprocessed data from this project:
    # https://gitlab.com/sutd_nlp/overlapping_mentions/tree/master.
    # We preprocess the data to align our folds with theirs.
    out_dir=$raw_dir/sutd-original
    mkdir $out_dir

    url_base=https://gitlab.com/sutd_nlp/overlapping_mentions/raw/master/data/GENIA/scripts

    for fold in train dev test
    do
        url=$url_base/$fold.data
        wget -nv $url -O $out_dir/$fold.data
    done
}

convert_sutd() {
    # Run script from https://gitlab.com/sutd_nlp/overlapping_mentions/tree/master/data/GENIA
    # to format the entity data. I've modified the script so that each article
    # gets dumped in its own file.

    corpus_name=$ner_dir/GENIAcorpus3.02.merged.xml

    sutd_base=$raw_dir/sutd-article
    tmpdir="$sutd_base/tmp"
    finaldir="$sutd_base/correct-format"

    mkdir $sutd_base
    mkdir $tmpdir
    mkdir $finaldir

    python ./scripts/data/genia/genia_xml_to_inline_sutd.py $corpus_name $tmpdir

    ls $tmpdir | grep "tok\.5types\.no_disc\.data" |
        while read file
        do
            newname=$(echo $file | sed -e "s/.tok.5types.no_disc.data//")
            cp $tmpdir/$file $finaldir/$newname.data
        done
}

####################

mkdir $out_dir
mkdir $log_dir

echo "Downloading raw GENIA data."
download_raw > $log_dir/00-download-genia.log

echo "Downloading preprocessed GENIA data from https://gitlab.com/sutd_nlp/overlapping_mentions."
download_sutd > $log_dir/01-download-sutd.log

echo "Converting SUTD-formatted GENIA data."
convert_sutd > $log_dir/02-convert-sutd.log

echo "Splitting GENIA docs into folds."
python ./scripts/data/genia/split_folds.py > $log_dir/03-split-folds.log

echo "Resolving differences between GENIA original version and SUTD version."
python ./scripts/data/genia/resolve_differences.py > $log_dir/04-resolve-differences.log

echo "Converting to JSON form."
python ./scripts/data/genia/convert_to_json.py > $log_dir/05-convert-to-json.log

echo "Aligning article ID's from NER and coref version."
python ./scripts/data/genia/align_articles.py > $log_dir/06-align-articles.log

# To include the 10 training documents with off-by-one errors, add the flag
# --keep-excluded below. For instance,
# python ./scripts/data/genia/merge_coref.py --keep-excluded.
echo "Merging coreference annotations into NER data."
python ./scripts/data/genia/merge_coref.py > $log_dir/07-merge-coref-all.log
python ./scripts/data/genia/merge_coref.py --ident-only > $log_dir/08-merge-coref-ident-only.log
