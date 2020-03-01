import nltk
from nltk.parse import corenlp

text = 'He came around the corner, and he spotted the crimson bird .'
tokens = text.split()

parser = CoreNLPParser(tagtype='pos')

pos_list = parser.tag(tokens) # Gives standard list of (token, POS) tuples
tree, = parser.raw_parse(text)  # native output is a listiterator over detected sentences
test_pos_list = tree.pos()

tree.pretty_print()

# Navigating the tree
#
# tree is of tree.Tree format ?
# len(T) = number of children
# T.height() = tree height
# T.label() = node label
# T.pos() = list of POS tags extracted
# T.leaves() = tree leaves
# T.subtrees() <== generates subtrees (children pointers)