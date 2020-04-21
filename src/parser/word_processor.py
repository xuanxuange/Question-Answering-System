from nltk.corpus import wordnet

def get_synonyms(word, part_of_speech):
    """
    :param word: str word
    :param part_of_speech: ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'
    :return: set of synonyms, set(str)
    """
    res = set()
    for word in wordnet.synsets(word, pos=part_of_speech):
        for lemma in word.lemmas():
            res.add(lemma.name().replace('_', ' '))
    return res



if __name__ == "__main__":
    print(get_synonyms("killed", wordnet.VERB))