# setup step following: https://github.com/nltk/nltk/wiki/Stanford-CoreNLP-API-in-NLTK

from nltk.parse import CoreNLPParser
from nltk.parse.corenlp import CoreNLPDependencyParser
pos_parser = CoreNLPParser(url='http://localhost:9000', tagtype='pos')
ner_parser = CoreNLPParser(url='http://localhost:9000', tagtype='ner')
dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')

def parse_raw_text(text):
    """
    parse raw string
    :param text: raw string
    :return:nltk.tree.Tree, the parse Tree
    """
    result = list(pos_parser.raw_parse(text))
    return result[0]

def pos_tagging(tokens):
    """
    POS tag raw tokens
    :param tokens: list[str], tokenized text
    :return:list[(token, tag))]
    """
    return list(pos_parser.tag(tokens))


def ner_tagging(tokens):
    """
    NER tag raw tokens
    :param tokens: list[str], tokenized text
    :return:list[(token, tag))]
    """
    return list(ner_parser.tag(tokens))

def dep_parse(text):
    result = dep_parser.parse(text.split())
    return result


if __name__ == "__main__":
    tree = parse_raw_text("Apple is looking at buying U.K. startup for $1 billion")
    #print(str(parse_raw_text("he where Arthur find in Beijing is not a male")[0][0]))
    print(str(parse_raw_text("isn't it true that he loves sports?")))
    print(str(parse_raw_text("was he killed?")))
    print(str(tree))


