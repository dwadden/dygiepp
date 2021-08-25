# Download and unzip the MECHANIC datset

mkdir -p tmp

wget -P tmp https://ai2-s2-mechanic.s3-us-west-2.amazonaws.com/data/data.zip
unzip tmp/data.zip -d tmp
mv tmp/data data/mechanic

# Cleanup. Remove tmpdir if it wasn't there before.
rm tmp/data.zip
if [ -z "$(ls -A tmp)" ]
then
   rmdir tmp
fi
