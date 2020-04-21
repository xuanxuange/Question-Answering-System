from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer


def get_synonyms(word, pos_tag):
    """
    :param word: str word
    :param part_of_speech: ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'
    :return: set of synonyms, set(str)
    """
    res = set()
    for word in wordnet.synsets(word, pos=pos_tag):
        for lemma in word.lemmas():
            res.add(lemma.name().replace('_', ' '))
    return res

def word_lemmatize(word, pos_tag):
    """
    :param word:
    :return: lemmatized word
    """
    return WordNetLemmatizer().lemmatize(word, pos_tag)




if __name__ == "__main__":
    print(get_synonyms("kill", wordnet.VERB))
    print(word_lemmatize("apples", wordnet.NOUN))