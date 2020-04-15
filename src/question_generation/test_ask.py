#!/usr/bin/env python3

import re
import sys
import os
import nltk
from nltk.parse import corenlp


def format_question(question):
    question = question.replace('-LRB- ', '(')
    question = question.replace(' -RRB-', ')')
    question = re.sub(r"\s([,;:\.\?!\'\"])", r"\1", question)
    question = question[0].upper() + question[1:]
    return question


def blockPrint():
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


if __name__ == "__main__":
    article = sys.argv[1]
    nquestions = int(sys.argv[2])
    blockPrint()
    from src.question_generation.question_gen import *
    from src.question_generation.question_gen_preprocess import preprocess

    parser = corenlp.CoreNLPParser(url='http://localhost:9000', tagtype='pos')

    parsed_list = []
    with open(article) as f:
        line = f.readline()
        while line:
            if len(line.split()) > 0:  # check for empty line

                parsed_iter = parser.parse_text(line,
                                                timeout=5000)  # native output is a listiterator over detected sentences

                while True:
                    try:
                        next_item = next(parsed_iter)
                        parsed_list.append(next_item)
                    except StopIteration:
                        break
            line = f.readline()
        f.close()

    preprocessed_list = preprocess(parsed_list)
    question_list = []

    for parse in preprocessed_list:
        question_list += getWhoWhat(parse) + getBinarySimple(parse) + getBinaryAuxiliary(parse)
    
    print(question_list)

    return
    i = 0
    enablePrint()
    while i < len(question_list) and i < nquestions:
        print(format_question(question_list[i]))
        i += 1
