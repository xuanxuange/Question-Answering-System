#!/bin/bash
nohup java -mx4g -cp "CoreNLP/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,ner,parse,depparse -status_port 9090 -port 9090 -timeout 15000 &
exec "$@"