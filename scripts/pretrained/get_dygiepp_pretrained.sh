# Download pretrained DyGIE++ models from AWS bucket and put the result in
# `./pretrained/`. Only download models if they aren't already there.

# Usage: bash scripts/pretrained/get_dygiepp_pretrained.sh.



for name in scierc scierc-lightweight genia genia-lightweight chemprot ace05-relation ace05-event
do
    if [ ! -f pretrained/$name.tar.gz ]
    then
        wget --directory-prefix=./pretrained \
            "https://s3-us-west-2.amazonaws.com/ai2-s2-research/dygiepp/master/${name}.tar.gz"
    fi
done
