from Parser.nltk_stanford_parser import *
import spacy
import sys
from spacy.symbols import *

def get_vn(self, curr, result):
    verbforms = ["VBD", "VBG", "VBN", "VBP", "VBZ"]
    pforms = ['VP', 'NP']
    if not curr.leaves():
        return result
    if curr.label() in verbforms or curr.label() == 'TO':
        result.append(curr)
        if curr.leaves():
            result.append(curr.leaves())
        return result
    elif curr.label() in pforms:
        pass
    for i in range(len(curr)):
        return get_vn(curr[i], result)

def answer_whn(question, rel_sentence):
    """
    Generate answer for whn questions, focus on nouns
    Return: the answer in str, if fail to process, return rel_sentence
    """
    answer = []
    sent_tokens = rel_sentence.split()
    q_tokens = question.split()
    main_subj = ''
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
        if word.dep == nsubj and (word.head.pos == VERB or word.head.pos == AUX or (len(rights) > 0 and word.head.pos == NOUN and rights[0] == ADP)):
            main_v = word.head
            main_subj = word
            main_v_index = q_tokens.index(main_v.text)
            break

    # find matched v in sentence
    for i, word in enumerate(s_dep):
        if word.head == word:
            root_index = i
            
    # check if main subj is the wh word
    if q_tokens[0] == main_subj.text:
        flag = True
        main_subj_index = sys.maxsize
    else:
        if main_subj.text not in sent_tokens and not flag:
            return rel_sentence
        main_subj_index = sent_tokens.index(main_subj.text)

    # append answer
    if main_subj_index < root_index:
        answer.append(sent_tokens[root_index + 1 : ])
    else:
        if main_v.text == sent_tokens[root_index]:
            answer.append(sent_tokens[ : root_index])
        else:
            answer.append(sent_tokens[root_index + 1 : ])
    answer.append(q_tokens[main_v_index : ])
    flat_answer = [item for sub in answer for item in sub]
    flat_answer[-1] = '.'
    return ' '.join(flat_answer)

if __name__ == "__main__":
    # question = "What is the primary weapon of Egyptian armies during the new Kingdom ?"
    question = "What leads to the death of the princess ?"
    sent = "Bow and arrow was the principal weapon of the Egyptian army ;"
    # sent = "The principal weapon of the Egyptian army was bow and arrow."
    # sent = "The weapon leads to the death of the princess."
    # sent = "The main thing that leads to the death of the princess was apple."
    answer_whn(question, sent)