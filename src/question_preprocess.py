from rake_nltk import Rake
import spacy
from spacy.symbols import nsubj, VERB, dobj, NOUN, PROPN, PRON


"""
Question types:
1. binary
2. what, who, which (questions on Noun)
3: why question
4: when, where, how (questions on ADV)
5: either-or
6. unknown
"""
BINARY = "BINARY"
UNKNOWN_TYPE = "unknown"
WH_N = "WH_N"
WH_ADV = "WH_ADV"
WHY = "WHY"
EITHER_OR = "EITHER_OR"

class q_preprocess:
    def __init__(self, question):
        self.quesiton = question
        self.whn_words = {'what', 'who', 'which'}
        self.whadv_words = {'where', 'when', 'how'}
        self.bi_words = {'is', 'does', 'are', 'were', 'was', 'did', 'do', 'should', 'will', 'would', 'had', 'has', 'have'}

    def qtype_recognition(self):
        """
        Recognize the type of the question
        Arg: question - the question to process
        Return: a label (str)
        """
        words = self.quesiton.split()
        first_word = words[0].lower()
        q_type = UNKNOWN_TYPE
        if first_word in self.whn_words:
            q_type = WH_N
        elif first_word in self.whadv_words:
            q_type = WH_ADV
        elif first_word in self.bi_words:
            q_type = BINARY
        elif first_word == 'why':
            q_type = WHY
        elif 'or' in words:
            q_type = EITHER_OR
        return q_type
           
    def get_sov(self):
        """
        Return the subj, v, obj of the sentence
        """
        nlp = spacy.load("en_core_web_sm")
        tree = nlp(self.quesiton)
        sov = {}
        for item in tree:
            # print(item.text, ' ', item.dep_, ' ', item.pos_, ' ', item.head)
            if item.dep == nsubj and item.head.pos == VERB:
                sov['verb'] = item.head
            if item.dep == nsubj and (item.pos == NOUN or item.pos == PROPN or item.pos == PRON) and item.head.pos == VERB:
                sov['subj'] = item
            if item.dep == dobj and (item.pos == NOUN or item.pos == PROPN or item.pos == PRON) and item.head.pos == VERB:
                sov['obj'] = item
        return sov           

    def generate_keywords(self):
        """
        Generate a list of keywords for the sentence
        Arg: sentence(str) - the sentence to parse
        Return: a list of string
        """
        keywords = []
        r = Rake()
        r.extract_keywords_from_text(self.quesiton)
        keywords = r.get_ranked_phrases()
        if not keywords:
            print('No keywords generated. Please rephrase your question.')
        if len(keywords) == 1:
            splited = keywords[0].split()
        return keywords

    def preprocess(self):
        """
        Generate a list of keywords, the main target and the question type for the question
        Arg: question(str) - the question to process
        Return: tuple(keywords(list), type(str), sov(dict))
        """
        q_type = self.qtype_recognition()
        keywords = self.generate_keywords()
        sov = self.get_sov()
        return (keywords, q_type, sov)


if __name__ == "__main__":
    question = 'How did the scholars construct a history of the 4th-6th Dynasties of Egypt?'
    question = "Who expanded Egypt's army and wielded it with great success to consolidate the empire created by his predecessors?"
    preprocess = q_preprocess(question)
    print(preprocess.preprocess())
    
