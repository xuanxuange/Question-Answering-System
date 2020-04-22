import nltk.tree as t
from src.parser.nltk_stanford_parser import parse_raw_text
from src.parser.word_processor import get_synonyms, word_lemmatize
from nltk.corpus import wordnet

def binary_question_transform(question):
    """
    :param question: raw string
    :return: raw generated sentence and boolean (indicates whether this question is actually antonym, for ex: isn't she beautiful?)
    """
    question.strip()
    question = question.lower()

    parse_tree = parse_raw_text(question)

    # find the left most VBZ(is)
    node = parse_tree
    while(type(node) is t.Tree and type(node[0]) is t.Tree and node[0].label() != 'VBZ' and node[0].label() != 'VBD'):
        node = node[0]

    #print(node[0].label())
    if type(node) is not t.Tree or type(node[0]) is not t.Tree:
        raise Exception("cannot transfer binary question \"{}\"".format(question))

    if type(node[1]) is t.Tree:
        if node[1].label() == 'RB':
            # for question like "isn't ...", it is actually the same as "is ..."
            del node[1]

        if type(node[1] is t.Tree):
            # switch node[1] and node[0]
            node[0], node[1] = node[1], node[0]
        else:
            raise Exception("cannot transfer binary question \"{}\"".format(question))
    else:
        raise Exception("cannot transfer binary question \"{}\"".format(question))

    # delete question mark
    if parse_tree[0][-1].label() == '.' and question[-1] == '?':
        del parse_tree[0][-1]

    return ' '.join(parse_tree.leaves())



def check_two_sentence_semantically_equal(sentence1, sentence2):
    """
    :param sentence1: str
    :param sentence2: str
    :return: True/False
    """
    # this function checks whether the meaning of sentence1 is included in sentence2, by search sentence 2 to see if it
    # contains all the noun, verb, adjective, and 'not'
    tree1 = parse_raw_text(sentence1)
    tree2 = parse_raw_text(sentence2)

    word_set1 = set()
    word_set2 = set()

    pos_tags1 = {}
    pos_tags2 = {}

    _recursively_get_keywords_parse_trees(tree1, word_set1, pos_tags1)
    _recursively_get_keywords_parse_trees(tree2, word_set2, pos_tags2)

    for word in word_set1:
        exists = False
        synonyms = get_synonyms(word, pos_tags1[word])
        synonyms.add(word)

        for synonym in synonyms:
            if synonym in word_set2:
                exists = True
                break

        if not exists:
            #print(word)
            return False

    return True



def _recursively_get_keywords_parse_trees(tree, word_set, pos_tag_values):
    if type(tree) is not t.Tree:
        return

    if type(tree[0]) is str:
        word = tree[0]
        current_label = tree.label()
        pos = ""
        if current_label.startswith('NN'):
            # noun
            pos = wordnet.NOUN
            word = word_lemmatize(word, pos)
            word_set.add(word)
            pos_tag_values[word] = pos
        elif current_label.startswith('VB'):
            # verb
            pos = wordnet.VERB
            word = word_lemmatize(word, pos)
            word_set.add(word)
            pos_tag_values[word] = pos
        elif current_label.startswith('JJ'):
            # adj
            pos = wordnet.ADJ
            word_set.add(word)
            pos_tag_values[word] = pos
        elif current_label == 'RB':
            # 'not, n't'
            pos = wordnet.ADV
            word_set.add(word)
            pos_tag_values[word] = pos
        else:
            for child in tree:
                _recursively_get_keywords_parse_trees(child, word_set, pos_tag_values)
    else:
        for child in tree:
            _recursively_get_keywords_parse_trees(child, word_set, pos_tag_values)



if __name__ == "__main__":
    tree= check_two_sentence_semantically_equal("apple is buying a U.K. startup", "apple was looking at buying U.K. startup for $1 billion")
    print(str(tree))
