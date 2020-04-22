import spacy
from spacy import displacy
nlp = spacy.load("en_core_web_sm")

def parse_raw_text(text):
    """
    :param text: raw text
    :return: spacy parse tree
    """
    return nlp(text)




if __name__ == "__main__":
    print(str([item.dep_ for item in parse_raw_text("she is beautiful")]))
    print(str([item.dep_ for item in parse_raw_text("he who traveled in china is granted access of the company by john")]))
    displacy.serve(parse_raw_text("he who traveled in china is granted access of the company by john"), style="dep")