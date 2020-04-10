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


# Download standford NLP
RUN wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip
RUN wget http://nlp.stanford.edu/software/stanford-ner-2018-10-16.zip
RUN unzip stanford-corenlp-full-2018-10-05.zip; \
unzip stanford-ner-2018-10-16.zip; \
mv stanford-corenlp-full-2018-10-05 CoreNLP; \
cd CoreNLP; \
export CLASSPATH=""; for file in `find . -name "*.jar"`; \
do export CLASSPATH="$CLASSPATH:`realpath $file`"; done

# Expose port 9090 for standford corenlp
ENV PORT 9090

EXPOSE 9090
ENTRYPOINT [ "./start_stanford_corenlp.sh" ]

# ENTRYPOINT ["/bin/bash", "-c"]
