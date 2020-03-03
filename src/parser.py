import nltk
import pickle
import os
from os import path
from nltk.parse import corenlp

sample = 'a1'

parser = corenlp.CoreNLPParser(tagtype='pos')

tree_list = []
if path.exists(sample + '.pkl'):
    print("loading from pkl")
    tree_list = pickle.load(open(sample + '.pkl', 'rb'))
else:
    print("pkl not found")
    with open('../data/development/set1/' + sample + '.txt') as f:
        line = f.readline()
        while line:
            if len(line.split()) > 0:  # check for empty line
                parsed_iter = parser.parse_text(line)  # native output is a listiterator over detected sentences

                while True:
                    try:
                        next_item = next(parsed_iter)

                        tree_list.append(next_item)
                    except StopIteration:
                        break 
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