from Parser.nltk_stanford_parser import *
import spacy
import sys
import string
from spacy.symbols import *

def find_sv(word):
    """
    Check if given node is the main subject
    Return: T if the node is main subject, F otherwise
    """
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
        
    # check if main subj is the wh word (first word in question)
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
    if flat_answer[-1] not in string.punctuation and flat_answer[-1][-1] not in string.punctuation: 
        flat_answer.append('.')
    
    return ' '.join(flat_answer)

def answer_howx(question, rel_sentence):
    """
    Generate answer for how many, how long and how much questions
    Return: the answer in str, if cannot process, return rel_sentence
    """
    q_tokens = question.split()
    nlp = spacy.load("en_core_web_sm")
    answer = []
    sent = nlp(rel_sentence)
    if q_tokens[1] == 'many' or q_tokens[1] == 'tall':
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

def answer_where(question, rel_sentence):
    """
    Generate answer for where questions
    Return: the answer in str, if fail to process, return rel_sentence
    """
    nlp = spacy.load("en_core_web_sm") 
    sent = nlp(rel_sentence)
    q = nlp(question)
    pos_answer = []
    answer = ''
    for ent in sent.ents:
        if ent.label_ == 'LOC':
            pos_answer.append(ent.text)
    if len(pos_answer) == 1:
        answer = pos_answer[0] + '.'
    elif len(pos_answer) < 1:
        return rel_sentence

    # > 1 possible answer
    else:
        q_tokens = question.split()
        s_tokens = rel_sentence.split()
        q_subj = ''
        # find the main subj in question
        for word in q:
            if find_sv(word):
                q_subj = word.text
        if not q_subj or not (q_subj in s_tokens):
            return rel_sentence
        # traverse to find the LOC NP related to q_subj
        else:
            for word in sent:
                if word.text == q_subj:
                    curr = word
                while(curr.head and not answer):
                    if curr.head.dep == prep:
                        r = [w.text for w in curr.head.rights]
                        if len(r) == 0: continue
                        for a in pos_answer:
                            if r[0] in a:
                                answer = a + '.'
                                break       
            if not answer:
                answer = rel_sentence # or return pos_answer[0]?
    return answer

def answer_when(question, rel_sentence):
    """
    Generate answer for when questions
    Return: the answer in str, if fail to process, return rel_sentence
    """
    ner = ['DATE', 'TIME']
    nlp = spacy.load("en_core_web_sm") 
    sent = nlp(rel_sentence)
    q = nlp(question)
    pos_answer = []
    answer = ''
    for ent in sent.ents:
        if ent.label_ in ner:
            pos_answer.append(ent.text)
    # check if the ner text is part of a noun chunk
    for i in range(len(pos_answer)):
        for chunk in sent.noun_chunks:
            if pos_answer[i] in chunk.text:
                pos_answer[i] = chunk.text
                break
    if len(pos_answer) == 1:
        answer = pos_answer[0] + '.'
    elif len(pos_answer) < 1:
        return rel_sentence
    # > 1 possible answer, return the one closest to the main verb in question
    else:
        s_tokens = rel_sentence.split()
        for word in q:
            if find_sv(word):
                main_v = word.head.text
        if not main_v or main_v not in rel_sentence:
            return rel_sentence
        
        answer = find_closest_answer(pos_answer, main_v, rel_sentence) + '.'
        if not answer:
            return rel_sentence
    return answer
        
def dfs_tree(curr, label, pos_answer, visited):
    """
    Traverse the given tree and save the leaves of given label to pos_answer
    Return: None
    """
    visited.append(curr)
    for child in curr:
        if not isinstance(child, str) and child.label() == label:
            adv = child.leaves()
            # remove punctuations
            if adv[-1] in string.punctuation:
                adv = adv[ : -1]
            pos_answer.append(adv)
            
        if child not in visited:
            dfs_tree(child, label, pos_answer, visited)
    return 

def find_closest_answer(pos_answer, main_v, rel_sentence):
    """
    Given a list of possible answer, return the one that is closest to the main verb in rel_sentence
    Return: a string
    """
    main_v_index = rel_sentence.find(main_v)
    min_dist = sys.maxsize
    answer = ''
    for a in pos_answer:
        a_index = rel_sentence.find(a)
        curr_dist = abs(a_index - main_v_index)
        if curr_dist < min_dist:
            answer = a
            min_dist = curr_dist
    return answer

def answer_other_adv(question, rel_sentence):
    """
    Generate answer for questions besides where/when which focus on adv (e.g. how) by returning a meaning adv phrase
    Return: the answer in str, if fail to process, return rel_sentence
    """
    ADV_STOP_LIST = ['almost', 'also', 'further', 'generally', 'greatly',
    'however', 'just', 'later', 'longer', 'often', 'only', 'typically', 
    'similarly', 'initially', 'for', 'basically', 'already', 'literally', 'largely']
    nlp = spacy.load("en_core_web_sm") 
    sent = nlp(rel_sentence)
    q = nlp(question)
    answer = ''
    main_v = ''
    # find the main verb in question
    for word in q:
        if find_sv(word):
            main_v = word.head
    if (main_v.text not in rel_sentence and main_v.lemma_  not in rel_sentence) or not main_v:
        return rel_sentence
    for word in sent:
        if word.head.lemma_ == main_v.lemma_ and word.dep == advmod and word.text.lower() not in ADV_STOP_LIST:
            answer = word.text + ' '
        if word.head.lemma_ == main_v.lemma_ and (word.dep == advcl or word.dep == prep):
            span = str(sent[word.left_edge.i : word.right_edge.i+1])
            answer += span
    if answer:
        answer += '.'
    # find the advp in parse tree
    else:
        tree = parse_raw_text(rel_sentence)
        pos_answer = []
        visited = []
        dfs_tree(tree[0], 'ADVP', pos_answer, visited)
        # flatten pos_answer and remove non-meaningful answer
        pos_answer = [a for sublist in pos_answer for a in sublist]
        pos_answer = [a for a in pos_answer if a.lower() not in ADV_STOP_LIST]

        if not pos_answer:
            return rel_sentence
        elif len(pos_answer) == 1:
            answer = pos_answer[0].lower() + '.'
        else:
            answer = find_closest_answer(pos_answer, main_v.text, rel_sentence) + '.'

    if not answer:
        return rel_sentence
    return answer


def answer_whadv(question, rel_sentence):
    """
    Wrapper for wh_adv questions
    Return: the answer in str, if fail to process, return rel_sentence
    """
    q_tokens = question.split()
    if q_tokens[0].lower() == 'where':
        answer = answer_where(question, rel_sentence)
    elif q_tokens[0].lower() == 'when':
        answer = answer_when(question, rel_sentence)
    else:
        answer = answer_other_adv(question, rel_sentence)
    return answer

def answer_why(question, rel_sentence):
    """
    Generate answer for why questions
    Return: the answer in str, if fail to process, return rel_sentence
    """
    pass


if __name__ == "__main__":
    # question = "What is the primary weapon of Egyptian armies during the new Kingdom ?"
    # question = "What leads to the death of the princess ?"
    # question = "How many people are there in the room ?"
    # question = "How much does it take for you to finish the task ?"
    # question = "What is the Egyptian Empire ?"
    # question = "Where was the tomb for sons of Ramesses II?"
    # question = "Where is the Osirian temple?"
    # sent = "the tomb he built for his sons, many of whom he outlived, in the Valley of the Kings has proven to be the largest funerary complex in Egypt ."
    # sent = "In the Osirian temple at Denderah, an inscription (translated by Budge, Chapter XV, Osiris and the Egyptian Resurrection) describes in detail the making of wheat paste models of each dismembered piece of Osiris to be sent out to the town where each piece is discovered by Isis."
    # sent = "It took me $100,000 to finish ."
    # sent = "There are one hundred people in the room."
    # sent = "Bow and arrow was the principal weapon of the Egyptian army;"
    # sent = "The principal weapon of the Egyptian army was bow and arrow."
    # sent = "The weapon leads to the death of the princess."
    # sent = "The main thing that leads to the death of the princess was apple."
    # sent = "The apple leads to the death of the princess ."
    sent = "The 4th-6th Dynasties of Egypt, are scarce and historians regard the history of the era as literally 'written in stone' and largely architectural in that it is through the monuments and their inscriptions peacefully that scholars have been able to construct a history."
    question = "How did the scholars construct a history of the 4th-6th Dynasties of Egypt ?"
    # question = "How did he complete the task? "
    # sent = "Under the bridge, he completed the task peacefully under her help."
    # question = "When did the Old Kingdom and its power reach a zenith ?"
    # sent = "The Old Kingdom and its royal power reached a zenith under the Fourth Dynasty (26132494 BC), which began with Sneferu (26132589 BC)."
    print(answer_whadv(question, sent))
    # print(answer_whn(question, sent))
    # print(answer_howx(question, sent))