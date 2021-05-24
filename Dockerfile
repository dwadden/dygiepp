# Set-up docker image for DYGIE++.
FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

# Datasets will be downloaded to the /dygiepp root directory in image.
# Please mount source code project dir at /dygiepp for using default paths.
RUN mkdir /dygiepp

# Required-base: set-up shared DYGIE++ modeling environment.
# GCC and make needed to compile python deps. SQLite3 for Optuna hyperparameter optimization.
RUN apt-get update && \
    apt-get -y install gcc make sqlite3
RUN conda create --name dygiepp python=3.7 -y
SHELL ["conda", "run", "-n", "dygiepp", "/bin/bash", "-c"]
# jsonnet has a conflict when installed with pip for now, install from conda.
RUN conda install -c conda-forge jsonnet -y
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# ACE05-EVENT: set-up environment for pre-processing.
SHELL ["/bin/bash", "-c"]
RUN conda create --name ace-event-preprocess python=3.7 -y
SHELL ["conda", "run", "-n", "ace-event-preprocess", "/bin/bash", "-c"]
COPY scripts/data/ace-event/requirements.txt /tmp/ace-prep-requirements.txt
RUN pip install -r /tmp/ace-prep-requirements.txt
RUN python -m spacy download en

# ACE05 dataset creation: Install CoreNLP (requires Java 1.8+) and zsh.
SHELL ["/bin/bash", "-c"]
RUN apt-get install openjdk-8-jdk openjdk-8-jre wget unzip -y
COPY scripts/data/ace05/get_corenlp.sh /tmp/get_corenlp.sh
RUN cd /dygiepp/ && bash /tmp/get_corenlp.sh
RUN conda install -c conda-forge zsh -y

# SciERC, GENIA, ChemProt: Download data.
# Downloader scripts require wget, unzip, and shared parsing code.
RUN apt-get install unzip wget -y
COPY scripts/data/shared /dygiepp/scripts/data/shared
# SciERC
COPY scripts/data/get_scierc.sh /tmp/get_scierc.sh
COPY dygie /dygiepp/dygie
ENV PYTHONPATH="${PYTHONPATH}:/dygiepp"
SHELL ["conda", "run", "-n", "dygiepp", "/bin/bash", "-c"]
RUN cd /dygiepp && bash /tmp/get_scierc.sh
# GENIA
COPY scripts/data/get_genia.sh /tmp/get_genia.sh
COPY scripts/data/genia /dygiepp/scripts/data/genia
ENV PYTHONPATH="${PYTHONPATH}:/dygiepp"
SHELL ["conda", "run", "-n", "dygiepp", "/bin/bash", "-c"]
RUN cd /dygiepp && bash /tmp/get_genia.sh
# ChemPROT
COPY scripts/data/get_chemprot.sh /tmp/get_chemprot.sh
COPY scripts/data/chemprot /dygiepp/scripts/data/chemprot
ENV PYTHONPATH="${PYTHONPATH}:/dygiepp"
SHELL ["conda", "run", "-n", "dygiepp", "/bin/bash", "-c"]
RUN pip install scispacy https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.5/en_core_sci_sm-0.2.5.tar.gz
RUN cd /dygiepp && bash /tmp/get_chemprot.sh

# Pretrained-models-all-DYGIEPP: Download pre-trained models for all DYGIEPP tasks.
RUN apt-get install wget -y
COPY scripts/pretrained/get_dygiepp_pretrained.sh /tmp/get_dygiepp_pretrained.sh
RUN cd /dygiepp && bash /tmp/get_dygiepp_pretrained.sh

# Required-base: cleanup.
RUN rm -rf /tmp /dygiepp/{scripts,dygie}

# Required-base: on run, ensure conda env is activated and /dygiepp is workdir.
WORKDIR /dygiepp/
SHELL ["/bin/bash", "-c"]
RUN conda init bash
RUN echo "conda activate dygiepp" >> ~/.bashrc
ENV PATH /opt/conda/envs/dygiepp/bin:$PATH
ENV CONDA_DEFAULT_ENV dygiepp
