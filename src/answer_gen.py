from Parser.nltk_stanford_parser import *
import spacy
import sys
import string
from spacy.symbols import *

def find_sv(word):
    if word.dep == nsubj and (word.head.pos == VERB or word.head.pos == AUX or (len(rights) > 0 and word.head.pos == NOUN and rights[0] == ADP)):
        return True
    else:
        return False


def answer_whn(question, rel_sentence):
    """
    Generate answer for whn questions which focus on nouns
    Return: the answer in str, if fail to process, return rel_sentence
    """
    answer = []
    sent_tokens = rel_sentence.split()
    q_tokens = question.split()
    main_subj = ''
    main_obj = ''
    main_v = ''
    main_v_index = 0
    root_index = 0
    flag = False 

    # find main v in question
    nlp = spacy.load("en_core_web_sm")
    q_dep = nlp(question)
    s_dep = nlp(rel_sentence)
    for i, word in enumerate(q_dep):
        rights = [token.pos for token in word.rights]
        if find_sv(word):
            main_v = word.head
            main_subj = word
            main_v_index = q_tokens.index(main_v.text)
    # find matched v in sentence
    for i, word in enumerate(s_dep):
        if word.head == word:
            root_index = i     
            root = word
        if word.head.head == word.head and word.lemma_ == main_v.lemma_:
            root_index = i
        
    # check if main subj is the wh word
    if q_tokens[0] == main_subj.text:
        meaningful_pos = [NOUN, VERB, ADJ, ADV, AUX, PROPN]
        flag = True
        curr = main_v.children
        for child in curr:
            if child.dep == dobj or child.dep == pobj or child.pos in meaningful_pos:
                main_obj = child
                break
            if child.dep == prep:
                for cc in child.children:
                    if cc.dep == dobj or cc.dep == pobj or cc.pos in meaningful_pos:
                        main_obj = cc
                        break
        main_obj_index = sent_tokens.index(main_obj.text)
                
    else:
        if main_subj.text not in sent_tokens and not flag:
            return rel_sentence
        main_subj_index = sent_tokens.index(main_subj.text)

    # append answer
    if not flag:
        if main_subj_index > root_index:
            answer.append(sent_tokens[ : root_index])
        else: 
            answer.append(sent_tokens[root_index + 1 : ])
    else:
        if main_obj_index > root_index:
            answer.append(sent_tokens[ : root_index])
        else: 
            answer.append(sent_tokens[root_index + 1 : ])
    flat_answer = [item for sub in answer for item in sub]

    # add/change the last element to be period
    if flat_answer[-1] not in string.punctuation: 
        flat_answer.append('.')
    else:
        flat_answer[-1] = '.'
    return ' '.join(flat_answer)

def answer_how(question, rel_sentence):
    """
    Generate answer for how many, how long and how much questions
    Return: the answer in str, if cannot process, return rel_sentence
    """
    q_tokens = question.split()
    nlp = spacy.load("en_core_web_sm")
    answer = []
    sent = nlp(rel_sentence)
    if q_tokens[1] == 'many':
        main = q_tokens[2]
        for i, ent in enumerate(sent.ents):
            if ent.label_ == "CARDINAL":
                answer.append(ent.text)
    elif q_tokens[1] == 'long':
        for i, ent in enumerate(sent.ents):
            if ent.label_ == "DATE":
                answer.append(ent.text)
    elif q_tokens[1] == 'much':
        for i, ent in enumerate(sent.ents):
            if ent.label_ == "MONEY":
                answer.append(ent.text)
    else:
        return rel_sentence
    if len(answer) == 0:
        return rel_sentence
    answer.append('.')
    return ' '.join(answer)

            

if __name__ == "__main__":
    # question = "What is the primary weapon of Egyptian armies during the new Kingdom ?"
    question = "What leads to the death of the princess ?"
    sent = "Bow and arrow was the principal weapon of the Egyptian army ;"
    # sent = "The principal weapon of the Egyptian army was bow and arrow."
    # sent = "The weapon leads to the death of the princess."
    # sent = "The main thing that leads to the death of the princess was apple."
    answer_whn(question, sent)