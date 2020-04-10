# Ubuntu Linux as the base imag
#FROM ubuntu:16.04
FROM python:3.7-slim

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Add the files
ADD . /QA
WORKDIR /QA

# Install basic packages
RUN apt-get -y update && \
    apt-get -y upgrade

# Pytorch
RUN pip3 install torch torchvision

# nltk
RUN pip3 install -U nltk

# download nltk data
RUN chmod +x nltk_download_models
RUN ./nltk_download_models

# pattern library for python3
RUN apt-get install default-libmysqlclient-dev
RUN pip3 install https://github.com/clips/pattern/archive/python3.zip

# spacy
RUN pip3 install -U spacy
RUN pip3 install -U spacy-lookups-data
RUN python3 -m spacy download en_core_web_sm


# other dependencies
RUN pip3 install -r requirement.txt


# download infersent dependecies
RUN chmod +x infersent_install.sh
RUN ./infersent_install.sh

CMD ["chmod 777 ask"]
CMD ["chmod 777 answer"]

ENTRYPOINT ["/bin/bash", "-c"]