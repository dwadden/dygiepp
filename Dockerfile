# Setup docker image for DYGIE++
FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-devel
COPY . /dygiepp

# setup DYGIE++ modeling environment
RUN conda create --name dygiepp python=3.7 -y
SHELL ["conda", "run", "-n", "dygiepp", "/bin/bash", "-c"]
RUN pip install -r /dygiepp/requirements.txt

# set-up environment for pre-processing ACE05-EVENT
SHELL ["/bin/bash", "-c"]
RUN conda create --name ace-event-preprocess python=3.7 -y
SHELL ["conda", "run", "-n", "ace-event-preprocess", "/bin/bash", "-c"]
RUN pip install -r /dygiepp/scripts/data/ace-event/requirements.txt
RUN python -m spacy download en

# For ACE05 dataset creation: Install CoreNLP (requires Java 1.8+) and zsh
SHELL ["/bin/bash", "-c"]
RUN apt-get update && apt-get install openjdk-8-jdk openjdk-8-jre -y
RUN bash /dygiepp/scripts/data/ace05/get_corenlp.sh
RUN conda install -c conda-forge zsh -y

# Download data for SciERC, GENIA, ChemProt (downloader scripts require wget and unzip)
RUN apt-get update && apt-get install unzip wget -y
RUN cd /dygiepp && bash ./scripts/data/get_scierc.sh
RUN cd /dygiepp && bash ./scripts/data/get_genia.sh
RUN cd /dygiepp && bash ./scripts/data/get_chemprot.sh

# Download pre-trained models for all tasks
RUN cd /dygiepp && bash ./scripts/pretrained/get_dygiepp_pretrained.sh