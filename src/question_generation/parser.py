import nltk
import pickle
import os
from os import path
from nltk.parse import corenlp


# cd ~/Documents/11-411/stanford-corenlp-full-2018-10-05; java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer

sample = 'a4'

debug = True

parser = corenlp.CoreNLPParser(url='http://localhost:9000', tagtype='pos')

corefparser = corenlp.CoreNLPParser(url='http://localhost:9000')
corefparser.parser_annotator='pos,ner,parse'

tree_list = []
if path.exists(sample + '.pkl') and not debug:
    print("loading from pkl")
    tree_list = pickle.load(open(sample + '.pkl', 'rb'))
else:
    print("pkl not found")
    with open('../../data/development/set1/' + sample + '.txt') as f:
        line = f.readline()
        while line:
            if len(line.split()) > 0:  # check for empty line
                parsed_iter = parser.parse_text(line, timeout=5000)  # native output is a listiterator over detected sentences
                res = corefparser.api_call(line, timeout=5000)
                # res["parse"]

                print(line)

                for sentence in res["sentences"]:
                    print(sentences["parse"])

                while True:
                    try:
                        next_item = next(parsed_iter)
                        print(next_item)
                        tree_list.append(next_item)
                    except StopIteration:
                        break 
                
                print("\n\n")
            line = f.readline()
        f.close()
    pickle.dump(tree_list, open(sample + '.pkl', 'wb'))
    print("pkl dumped")

print("sample loaded")


# Navigating the tree
#
# tree is of tree.Tree format ?
# len(T) = number of children
# T.height() = tree height
# T.label() = node label
# T.pos() = list of POS tags extracted
# T.leaves() = tree leaves
# T.subtrees() <== generates all subtrees
# T[i] for children indexing