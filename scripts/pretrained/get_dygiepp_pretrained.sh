# Download pretrained DyGIE++ models from AWS bucket and put the result in
# `./pretrained/`.

# Usage: python scripts/pretrained/get_dygiepp_pretrained.py.

# SciERC.
wget --directory-prefix=./pretrained \
    https://s3-us-west-2.amazonaws.com/ai2-s2-research/dygiepp/scierc.tar.gz

# SciERC lightweight.
wget --directory-prefix=./pretrained \
    https://s3-us-west-2.amazonaws.com/ai2-s2-research/dygiepp/scierc-lightweight.tar.gz


# Genia
wget --directory-prefix=./pretrained \
    https://s3-us-west-2.amazonaws.com/ai2-s2-research/dygiepp/genia.tar.gz


# Genia lightweight
wget --directory-prefix=./pretrained \
    https://ai2-s2-research.s3-us-west-2.amazonaws.com/dygiepp/genia-lightweight.tar.gz


# ChemProt
wget --directory-prefix=./pretrained \
    https://ai2-s2-research.s3-us-west-2.amazonaws.com/dygiepp/chemprot.tar.gz


# ACE05-Event
wget --directory-prefix=./pretrained \
    https://ai2-s2-research.s3-us-west-2.amazonaws.com/dygiepp/ace05-event.tar.gz
