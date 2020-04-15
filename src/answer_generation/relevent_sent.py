import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from rake_nltk import Rake
import re
from src.answer_generation.tokenizer import *


def generate_keywords(sentence):
    """
    Generate a list of keywords for the sentence
    Arg: sentence(str) - the sentence to parse
    Return: a list of string
    """
    keywords = []
    r = Rake()
    r.extract_keywords_from_text(sentence)
    keywords = r.get_ranked_phrases()
    if not keywords:
        print('No keywords generated. Please rephrase your question.')
    if len(keywords) == 1:
        splited = keywords[0].split()
    return keywords

def cal_similarities(sentence1, sentence2):
    """
    Calculate the similarities of two sentences using cosine similarity
    Arg: sentence1, sentence2 - sentences to compare
    Return: the similariy (float)
    """
    list1 = word_tokenize(sentence1)
    list2 = word_tokenize(sentence2)
    l1 = []; l2 = []
    c = 0
    sw = stopwords.words('english')
    set1 = {w for w in list1 if not w in sw}  
    set2 = {w for w in list1 if not w in sw} 
    rvector = set1.union(set2)
    for w in rvector:
        if w in set1:
            l1.append(1)
        else:
            l1.append(0)
        if w in set2:
            l2.append(1)
        else:
            l2.append(0)
    for i in range(len(rvector)):
        c += l1[i] * l2[i]
    cosine = c / float((sum(l1)*sum(l2))**0.5)
    return cosine


def get_span(interval):
    """
    Calculate the span of an interval given
    Arg: interval(tuple/list) - an interval of the positions of keywords in the doc
    Return: the span (int)
    """
    span = interval[-1] - interval[0]
    return span

def check_kw_match(sentences, keywords):
    """
    Return the sentence containing most keywords
    Arg: sentences(list) - a list of sentences
         keywords(list) - a list of keywords
    Return: a string
    """
    rel_sentence = ''
    max_keywords = 0
    contain_keywords = 0
    for s in sentences:
        for keyword in keywords:
            if keyword in s:
                contain_keywords += 1
        if contain_keywords > max_keywords:
            rel_sentence = s
    return rel_sentence

def find_complete_psg(doc, content):
    pattern = r"\.?[\s\n]+[\w\s,;:_\'\"\-\–\(\)]*" + re.escape(content) + "[\w\s,;:_\'\"\–\-\(\)]*\.[\s\n]+"
    match = re.finditer(pattern, doc, re.IGNORECASE)
    psg = []

    for m in match:
        psg.append(doc[m.start()+1:m.end()].strip())
    return psg 

def get_most_relevent_sent(question, doc):
    # get keywords from the question
    keywords = generate_keywords(question)
    print('keywords: ', keywords)
    pos_to_keyword = {} # key: start pos of keyword, value: index in pos_list
    pos_list = [] # list of list, each list records the start pos of occurrences of a keyword
    min_interval = (0, len(doc))
    rel_sentence = '' # return value
    # construct list of start positions for each keyword
    for i in range(len(keywords)):
        first = True
        # find all occurrence of keyword
        matches = re.finditer(keywords[i], doc, re.IGNORECASE)
        # append pos to list, record pos-list_index
        pos_list.append(list())
        for match in matches:          
            pos_list[i].append(match.start())
            pos_to_keyword[match.start()] = i

    # record the keywords found
    found_keywords = []
    for i in range(len(pos_list)):
        if pos_list[i]:
            found_keywords.append(keywords[i])

    # if none of the keyword is found, output error msg
    if not found_keywords:
        print("Keywords not found in the document given.")
        psg = text_to_sentence(doc)
        min_dist = sys.maxsize
        for i in psg:
            if cal_similarities(i, question) < min_dist:
                rel_sentence = i
        # rel_sentence = "Keywords not found in the document given."
        return rel_sentence

    # print('pos-listindex: ', pos_to_keyword)
    # print('list of pos: ', pos_list)

    # if only one keyword, return the sentence with highest cos similarity to the question
    if len(found_keywords) == 1:
        psg = find_complete_psg(doc, found_keywords[0])
        if not psg:
            print("Content not found in the document given.")
            rel_sentence = "Content not found in the document given."
        min_sim = sys.maxsize
        for sentence in psg:
            similarity = cal_similarities(sentence, question)
            if similarity < min_sim:
                rel_sentence = sentence
                min_sim = similarity
        return rel_sentence

    # if more than one keywords, find the min interval containing all keywords
    tmp = [] # current interval with all keywords pos
    for l in pos_list:
        if not l:
            continue
        tmp.append(l.pop(0))
        tmp = sorted(tmp)
    min_interval = (tmp[0], tmp[-1])
    while len(pos_list[pos_to_keyword[tmp[0]]]) > 0:
        leftmost = pos_list[pos_to_keyword[tmp[0]]]
        p = leftmost.pop(0)
        q = tmp[1]
        # add the next element to tmp
        tmp.pop(0)
        tmp.append(p)
        tmp = sorted(tmp)
        if p > min_interval[1]:
            if get_span(min_interval) >= get_span(tmp):
                min_interval = (tmp[0], tmp[-1])
        else:
            r = min_interval[1]
            min_interval = (min(p,q), r)

    # find the complete sentences span the interval
    content = doc[min_interval[0]: min_interval[1]]
    print(content)
    # print(content)
    psg = find_complete_psg(doc, content)
    if not psg:
        print("Content not found in the document given.")
        rel_sentence = "Content not found in the document given."
        return rel_sentence
    # print(psg)

    # if multiple sentences found, return the one with most keywords
    sentences = text_to_sentence(psg[0])
    # print(sentences)
    if len(sentences) > 1:
        rel_sentence = check_kw_match(sentences, keywords)
    else:
        rel_sentence = sentences[0]
    
    return rel_sentence
    

if __name__ == "__main__":
    # file_rmextra('../data/development/set1/a6.txt')
    path = '../../data/Questions_set1(1-10).txt'
    psg_path = '../../data/development/set1/a1.txt'
    paren_pattern = r"\(.*\)"
    subtitle_pattern = r"a?\d+\.(txt)?\s?"
    doc = file_rmextra(path)
    doc = re.sub(paren_pattern, '', doc)
    doc = re.sub(subtitle_pattern, '', doc).strip()
    lines = doc.split('\n')
    # print(lines)
    # print(psg_path[:24])
    counter = 0
    psg = file_rmextra(psg_path)
    for i in range(len(lines)):
        if len(lines[i]) < 5:
            continue
        if counter % 5 == 0:
            # print("should change to " + str(counter//5+1))
            psg_path = psg_path[:24]
            psg_path += '/a' + str(counter//5 + 1) + '.txt'
            psg = file_rmextra(psg_path)
        line = lines[i].strip()
        sentence = get_most_relevent_sent(line, psg)
        print(sentence, psg_path)
        with open ('./test_result.txt', 'a') as out:
            out.write(str(counter+1) + '. ' + sentence + '\n')
            out.write('\n')

        counter += 1



