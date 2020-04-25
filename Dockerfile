FROM ubuntu:18.04

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN apt-get -y update && \
apt-get -y upgrade

# install python 3.7
RUN apt-get -y install software-properties-common
# RUN apt-get -y install --reinstall ca-certificates
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt -y install python3.7
RUN apt-get -y install python3-pip
RUN apt-get -y install python3.7-dev && \
apt-get install -y wget

RUN apt install -y default-jdk
RUN apt-get install zip unzip
RUN apt install -y curl

# Add the files
ADD . /QA
WORKDIR /QA

# Pytorch
RUN python3.7 -m pip install torch torchvision

# nltk
RUN python3.7 -m pip install -U nltk

# download nltk data
RUN chmod +x nltk_download_models
RUN ./nltk_download_models

# pattern library for python3
RUN apt-get -y install default-libmysqlclient-dev
ENV PYTHONPATH "${PYTHONPATH}:$(pwd)/site-packages/"

# RUN pip3 install https://github.com/clips/pattern/archive/python3.zip

# spacy
RUN python3.7 -m pip install -U spacy==2.1.0
RUN python3.7 -m pip install -U spacy-lookups-data
RUN python3.7 -m spacy download en_core_web_sm
RUN python3.7 -m spacy download en_core_web_lg

# other dependencies
RUN python3.7 -m pip install -r requirement.txt

# install neuralcoref
RUN python3.7 -m pip install neuralcoref --no-binary neuralcoref

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
ENV PORT 9000

EXPOSE 9000
RUN chmod +x start_stanford_corenlp.sh
ENTRYPOINT [ "./start_stanford_corenlp.sh" ]

# ENTRYPOINT ["/bin/bash", "-c"]
