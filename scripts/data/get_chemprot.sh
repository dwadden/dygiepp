# Download chemprot and format for model training.
# From https://biocreative.bioinformatics.udel.edu/news/corpora/chemprot-corpus-biocreative-vi/.

out_dir=data/chemprot
mkdir $out_dir

# Download and unzip.
mkdir $out_dir/raw_data
wget https://biocreative.bioinformatics.udel.edu/media/store/files/2017/ChemProt_Corpus.zip -P $out_dir/raw_data
unzip $out_dir/raw_data/ChemProt_Corpus.zip -d $out_dir/raw_data

unzip_dir=$out_dir/raw_data/ChemProt_Corpus
ls $unzip_dir | while read name
do
    unzip -q $unzip_dir/$name -d $unzip_dir
done

rm -r $unzip_dir/__MACOSX
rm -r $unzip_dir/*.zip
rm -r $out_dir/raw_data/ChemProt_Corpus.zip

# Get rid of the `_gs` suffix on the test data.
python scripts/data/chemprot/01_rename_test.py

# Run formatting.
mkdir $out_dir/processed_data
python scripts/data/chemprot/02_chemprot_to_input.py
