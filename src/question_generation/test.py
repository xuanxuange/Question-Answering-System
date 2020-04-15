# pip3 install https://github.com/huggingface/neuralcoref-models/releases/download/en_coref_lg-3.0.0/en_coref_lg-3.0.0.tar.gz

# pip3 install stanfordcorenlp

from stanfordcorenlp import StanfordCoreNLP
from os import path
import pickle


debug = False
sample = 'a1'
corenlp_folder = '/Users/Thomas/Documents/11-411/CoreNLP'

doctext = []
if path.exists(sample + '_text.pkl') and not debug:
    print("loading from pkl")
    doctext = pickle.load(open(sample + '_text.pkl', 'rb'))
else:
    print("pkl not found")
    with open('../../data/development/set1/' + sample + '.txt') as f:
        line = f.readline()
        while line:
            doctext.append(line)
            line = f.readline()
        f.close()
    pickle.dump(doctext, open(sample + '_text.pkl', 'wb'))
    print("pkl dumped")
print("sample loaded")


nlp = StanfordCoreNLP(corenlp_folder)

testline = doctext[4]

parse_tree = nlp.parse(testline)
ner_data = nlp.ner(testline)
print(ner_data)
nlp.close()