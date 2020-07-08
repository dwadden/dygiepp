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