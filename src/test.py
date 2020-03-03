from qgen import *
from parser import *

print("modules imported")

results = []
for tree in tree_list:
    results += getWhoWhat(tree) + getBinarySimple(tree) + getBinaryAuxiliary(tree)

for result in results:
    print(result)