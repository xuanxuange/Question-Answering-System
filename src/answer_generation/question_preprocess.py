from rake_nltk import Rake
import spacy
from spacy.symbols import nsubj, VERB, dobj, NOUN, PROPN, PRON
from src.parser.nltk_stanford_parser import *

"""
Preprocess the question, generate keywords, question type and processed complete question
Could be used to find most relavant sentence, generate answers
Question types:
1. binary
2. what, who, which (questions on Noun)
3: why question
4: when, where, how (questions on ADV)
5: either-or
6. unknown
"""
BINARY = "BINARY"
UNKNOWN_TYPE = "UNK"
WH_N = "WH_N"
WH_ADV = "WH_ADV"
WHY = "WHY"
HOW = "HOW"
EITHER_OR = "EITHER_OR"

class q_preprocess:
    def __init__(self, question):
        self.quesiton = question
        self.whn_words = {'what', 'who', 'which'}
        self.whadv_words = {'where', 'when', 'how'}
        self.how_words = {'many', 'long'}
        self.bi_words = {'is', 'does', 'are', 'were', 'was', 'did', 'do', 'should', 'will', 'would', 'had', 'has', 'have', "doesn't", "haven't", "hasn't", "hadn't"}
        # for tree parser
        self.wh_forms = ["WHNP", "WHADVP", "WHADJP", "WHPP"]
        self.bi_clause = ["SINV", "SQ", "S"]
        self.wh_clause = ["SBAR", "SBARQ"]

    def  qtype_recognition(self):
        """
        Recognize the type of the question and remove extra wh/binary words at the beginning of the original question
        Arg: question - the question to process
        Return: a label (str), a modified question (str)
        """
        print("question: {}".format(self.quesiton))
        words = self.quesiton.split()
        if words[1] == 'or' or words[1] == '/':
            words = words[2 :]
        elif '/' in words[0]:
            slash_pos = words[0].find('/')
            words[0] = words[0][slash_pos+1 :]
        first_word = words[0].lower()
        second_word = words[1].lower()
        q_type = UNKNOWN_TYPE
        if first_word in self.whn_words:
            q_type = WH_N
        elif first_word in self.whadv_words:
            if second_word not in self.how_words:
                q_type = WH_ADV
            else:
                q_type = (first_word + ' ' + second_word).upper()
        elif 'or' in words:
            q_type = EITHER_OR
        elif first_word in self.bi_words:
            q_type = BINARY
        elif first_word == 'why':
            q_type = WHY
        question = ' '.join(words)
        return q_type, question
        
    def expand_question(self, question):
        """
        Split EITHEROR question into two questions
        Return: list[str], a list of questions
        """
        tmp1 = []
        tmp2 = []
        tmp_list = []
        q_list = []
        flag = False
        tmp_tree = parse_raw_text(quesiton)
        for i in range(len(tmp_tree)):
            if tmp_tree[i].label() in self.bi_clause:
                candidate = tmp_tree[i]
                for i in range(len(candidate)):
                    curr = candidate[i]
                    if 'or' not in curr.leaves():
                        tmp1.append(curr.leaves())
                        if not flag:
                            tmp2 = tmp1.copy()
                        else:
                            tmp2.append(curr.leaves())
                        continue
                    for i in range(len(curr)):
                        if curr[i].leaves()[0] != 'or':
                            tmp1.append(curr[i].leaves())
                        else:
                            tmp2.append(curr[i+1].leaves())
                            flag = True
                            break
        flat_list1 = [item for sublist in tmp1 for item in sublist]
        flat_list2 = [item for sublist in tmp2 for item in sublist]
        q_list.append(' '.join(flat_list1))
        q_list.append(' '.join(flat_list2))
        return q_list
           
    def shorten_question(self, q_type, question):
        """
        Remove extra clause before/after the main questioning part
        Return: list[str], a list of complete main questions
        """
        q_list = []
        final_qlist = [] # len=2 if either/or question, otherwise len=1
        # split question to two binary question
        if q_type == EITHER_OR:
            q_list = self.expand_question(question)
        else:
            q_list.append(question)
        
        # process each question in the list
        for i in q_list:
            result_question = []
            tree = parse_raw_text(i)
            for i in range(len(tree)):
                candidate = tree[i]
                # direct binary-q
                if candidate.label() in self.bi_clause:
                    result_question.append(candidate.leaves())
                        
                # wh-q or question w/ prefix sentence
                elif candidate.label() in self.wh_clause:
                    for j in range(len(candidate)):
                        if candidate[j].label() in self.wh_forms or candidate[j].label() == 'SQ':
                            result_question.append(candidate[j].leaves())

            result = [item for sublist in result_question for item in sublist]
            if result[-1] != '?':
                result.append('?')
            result = ' '.join(result)
            final_qlist.append(result)          
        return final_qlist
                        

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
        q_type, question = self.qtype_recognition()
        keywords = self.generate_keywords()
        q_shortened = self.shorten_question(q_type, question)
        return (keywords, q_type, q_shortened)


if __name__ == "__main__":
    question = 'How did the scholars construct a history of the 4th-6th Dynasties of Egypt?'
    question = "Who expanded Egypt's army and wielded it with great success to consolidate the empire created by his predecessors?"
    question = "How many hieroglyphs does Middle Egyptian make use of?"
    question = "How long did he spend in Spain?"
    question = "who did not use pegs , treenails,  or metal fasteners , but relied on rope to keep their ships assembled?"
    question = "Who or what had two principal functions : to ensure an ordered existence and to defeat death by preserving life into the next world?"
    question = "Did he run away to an island or stay here, at the end of story ?"
    question = "Has/Have Internal disorders set in during the incredibly long reign of Pepi II -LRB- 2278 -- 2184 BC -RRB- towards the end of the dynasty?"
    question = "among all prof in cmu, which prof is most famous?"
    preprocess = q_preprocess(question)
    print(preprocess.preprocess())
    
