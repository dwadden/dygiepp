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
raw_dir=$out_dir/raw-data
ner_dir=$raw_dir/GENIAcorpus3.02p

download_raw() {
    # Download the data.

    mkdir $out_dir
    mkdir $raw_dir

    # Download the entities.
    wget http://www.nactem.ac.uk/GENIA/current/GENIA-corpus/Part-of-speech/GENIAcorpus3.02p.tgz \
        -P $raw_dir

    # Download the coreference data.
    wget http://www.nactem.ac.uk/GENIA/current/GENIA-corpus/Coreference/GENIA_MedCo_coreference_corpus_1.0.tar.gz \
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
        wget $url -O $out_dir/$fold.data
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

# download_raw
# download_sutd
# convert_sutd
# python ./scripts/data/genia/genia_split_doc_by_fold.py
# python ./scripts/data/genia/fix_genia_article.py
# python ./scripts/data/genia/format_genia_sutd.py
# python ./scripts/data/genia/genia_align_ids.py
python ./scripts/data/genia/genia_add_coref.py
