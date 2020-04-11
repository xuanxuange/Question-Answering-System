FROM ubuntu:18.04

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN apt-get -y update && \
apt-get -y upgrade && \
apt-get -y install python3-pip python3-dev && \
apt-get install -y wget

RUN apt install -y default-jdk
RUN apt-get install zip unzip
RUN apt install -y curl

# Add the files
ADD . /QA
WORKDIR /QA

# Pytorch
RUN pip3 install torch torchvision

# nltk
RUN pip3 install -U nltk

# download nltk data
RUN chmod +x nltk_download_models
RUN ./nltk_download_models

# pattern library for python3
RUN apt-get -y install default-libmysqlclient-dev
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
ENV PORT 9000

EXPOSE 9000
ENTRYPOINT [ "./start_stanford_corenlp.sh" ]

# ENTRYPOINT ["/bin/bash", "-c"]
