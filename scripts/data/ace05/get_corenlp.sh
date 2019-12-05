# Download Stanford CoreNLP libararies.

out_dir=./scripts/data/ace05/common
wget --directory-prefix $out_dir http://nlp.stanford.edu/software/stanford-corenlp-full-2015-04-20.zip
wget --directory-prefix $out_dir http://nlp.stanford.edu/software/stanford-postagger-2015-04-20.zip

for name in stanford-corenlp-full-2015-04-20 stanford-postagger-2015-04-20
do
    unzip $out_dir/${name}.zip -d $out_dir
    rm $out_dir/${name}.zip
done
