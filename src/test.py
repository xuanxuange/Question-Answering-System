# pip3 install https://github.com/huggingface/neuralcoref-models/releases/download/en_coref_lg-3.0.0/en_coref_lg-3.0.0.tar.gz

from qgen import *
from parser import *

print("modules imported")

results = []
for tree in tree_list:
    results += getWhoWhat(tree) + getBinarySimple(tree) + getBinaryAuxiliary(tree)

for result in results:
    if ',' not in result:
        print(result)