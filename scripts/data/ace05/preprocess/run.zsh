#!/bin/zsh
# ace apf, sgm to ann, txt
mkdir result
for i in English/*/timex2norm/*.sgm
do
    echo $i
    python3 ace2ann.py `echo $i | sed -e 's/.sgm/.apf.xml/g'` $i result/`basename $i .sgm`.txt >! result/`basename $i .sgm`.ann
done #&>! log
# split & parse
mkdir text
java -cp ".:../common/stanford-corenlp-full-2015-04-20/*" edu.stanford.nlp.pipeline.StanfordCoreNLP -props ../common/props_ssplit -outputFormat conll -filelist train_list -outputDirectory text/
# conll to text
for i in text/*.conll
do
    python3 ../common/conll2txt.py $i >! text/`basename $i .txt.conll`.split.txt
done
# adjust offsets
for i in text/*.split.txt
do
    echo $i && python3 ../common/standoff.py result/`basename $i .split.txt`.txt result/`basename $i .split.txt`.ann $i >! text/`basename $i .split.txt`.split.ann
done #&>! log
# fix sentence split errors
mkdir fixed
for i in text/*.split.txt
do
    python3 ../common/fix_sentence_break.py $i text/`basename $i .split.txt`.split.ann fixed/`basename $i .split.txt`.txt
done
# parse ssplit-fixed text
java -cp ".:../common/stanford-corenlp-full-2015-04-20/*" edu.stanford.nlp.pipeline.StanfordCoreNLP -props ../common/props_fixed -outputFormat conll -filelist train_list_fixed -outputDirectory fixed/
for i in fixed/*.conll
do
    python3 ../common/conll2txt.py $i >! fixed/`basename $i .txt.conll`.split.txt
done
# collect data
mkdir corpus
cd corpus
cp ../result/*.txt .
cp ../result/*.ann .
cp ../fixed/*.split.txt .
cd ..
# adjust offsets
for i in corpus/*.split.txt
do
    echo $i && python3 ../common/standoff.py corpus/`basename $i .split.txt`.txt corpus/`basename $i .split.txt`.ann $i >! corpus/`basename $i .split.txt`.split.ann
done #&>! log &
# conll to so
for i in fixed/*.split.txt
do
    echo $i && perl ../common/dep2so.prl fixed/`basename $i .split.txt`.txt.conll $i >! corpus/`basename $i .txt`.stanford.so
done
# split into train, dev, test
mkdir corpus/train corpus/dev corpus/test
for i in `cat split/train`
do
mv corpus/$i* corpus/train
done
for i in `cat split/dev`
do
mv corpus/$i* corpus/dev
done
for i in `cat split/test`
do
mv corpus/$i* corpus/test
done
