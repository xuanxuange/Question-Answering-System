import nltk
from nltk.parse import corenlp

text = 'He came around the corner, and he spotted the crimson bird .'
tokens = text.split()

parser = corenlp.CoreNLPParser(tagtype='pos')

with open('../data/development/set1/a1.txt') as f:
    line = f.readline()
    while line:
        if len(line.split()) > 0:  # check for empty line
            parsed_iter = parser.parse_text(line)  # native output is a listiterator over detected sentences

            while True:
                try:
                    next_item = next(parsed_iter)

                    next_item.pretty_print()
                except StopIteration:
                    break 
        line = f.readline()
    f.close()

# Navigating the tree
#
# tree is of tree.Tree format ?
# len(T) = number of children
# T.height() = tree height
# T.label() = node label
# T.pos() = list of POS tags extracted
# T.leaves() = tree leaves
# T.subtrees() <== generates subtrees (children pointers)