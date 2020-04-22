from nltk.corpus import wordnet as wn

def word_similarity(word1, word2, pos1=None, pos2=None):
	try:
		synSet1=wn.synsets(word1,pos=pos1)[0]
		synSet2=wn.synsets(word2,pos=pos2)[0]
	except:
		return -1.0
	return synSet1.path_similarity(synset2)

def get_word_dist_to_root(word1,pos=None):
	try:
		synSet = wn.synsets(word,pos=pos)[0]
	except:
		return -1
	dist=0
	while(True):
		hypers = synSet.hypernyms()
		if not hypers or dist >= 100:
			return dist
		else:
			synSet = hypers[0]
			dist += 1

#Reaches up n layers to get hypernym
def get_nth_hypernyms(word, pos):
	try:
		synSet = wn.synset(word+"."+pos+".01")
	except:
		return set(),0
	for i in range(n):
		nextSynSet = synSet.hypernyms()[0]
		if not nestSynSet:
			break
		synSet = nextSynSet[0]
	out = set()
	for lemma in synSet:
		out.add(lemma.name())
	return out,i

def synonym_get(word,pos):
	results = {}
	try:
		synSet = wn.synset(word+"."+pos+".01")
	except:
		return result

	results["synonyms"] = set()
	results["hypernyms"] = set()
	results["meronyms"] = set()
	results["holonyms"] = set()
	for lemma in synSet.lemmas():
	    results["synonyms"].add(lemma.name())
	for lemma in synSet.hypernyms():
		results["hypernyms"].add(lemma.name()) #This right here. Use this whenever possible
	for lemma in synSet.member_holonyms():
		results["holonyms"].add(lemma.name())
	for lemma in synSet.part_holonyms():
		results["holonyms"].add(lemma.name())
	for lemma in synSet.substance_holonyms():
		results["holonyms"].add(lemma.name())
	for lemma in synSet.member_meronyms():
		results["meronyms"].add(lemma.name())
	for lemma in synSet.part_meronyms():
		results["meronyms"].add(lemma.name())
	for lemma in synSet.substance_meronyms():
		results["meronyms"].add(lemma.name())
	return results

def antonym_get(word,pos=None):
	results = {}
	try:
		synSet = wn.synset(word+"."+pos+".01")
	except:
		return result

	results["antonyms"] = set()
	results["hyperantonyms"] = set()

	for lemma in synSet.lemmas():
		for antolemma in lemma.antonyms():
			results["antonyms"].add(antolemma.name())
	for lemma in synSet.hypernyms():
		for hyperantolemma in lemma.antonyms():
			results["hyperantonyms"].add(hyperantolemma.name())
	return results