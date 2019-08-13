# Run the SUTD conversion of the GENIA data.
# I've altered the conversion so that each document gets its own file.
# Will need a follow-on script to get the splits for train, test, and dev data.

corpus_name="/data/dave/proj/scierc_coref/data/genia/GENIAcorpus3.02p/GENIAcorpus3.02.merged.xml"

out_base="/data/dave/proj/scierc_coref/data/genia/sutd-article"
tmpdir="$out_base/tmp"
finaldir="$out_base/final"

mkdir $tmpdir
mkdir $finaldir

python genia_xml_to_inline_sutd.py $corpus_name $tmpdir

ls $tmpdir | grep "tok\.5types\.no_disc\.data" |
    while read file
    do
        newname=$(echo $file | sed -e "s/.tok.5types.no_disc.data//")
        cp $tmpdir/$file $finaldir/$newname.data
    done
