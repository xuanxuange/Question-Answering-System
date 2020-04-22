from nltk.corpus import wordnet as wn
from src.question_generation.nym_utils import antonym_get, synonym_get, get_nth_hypernyms
import pattern.en
import random

def postprocess(qlist, tlist, parser):
	results = []
	for i in range(len(qlist)):
		t = tlist[i]
		if t == "AP":
			if random.uniform(0,1) > 0.5:
				results.append(qlist[i].replace("apt descriptor", "accurate or inaccurate descriptor"))
			else:
				results.append(qlist[i])
			continue
		try:
			p = parser.parse_text(qlist[i],timeout=5)
			qtree = next(p)
		except:
			results.append(qlist[i])
			continue
		qpos = qtree.pos()
		new_sent = []
		for j in range(len(qpos)):
			word, wpos = qpos[j]
			if wpos in ["NN", "NNS", "JJ"] and random.uniform(0,1) > 1.5:#Disabled for now
				is_plural = False
				if wpos=="NNS":
					newWord = wn.morphy(word)
				else:
					newWord = word
				if wpos in ["NN", "NNS"]:
					hypernyms,_ = get_nth_hypernyms(newWord,pos='n',n=2)
					wordset = synonym_get(newWord,pos='n')["synonyms"]
					wordset.update(hypernyms)
					wordset = list(wordset)
				else:
					wordset = list(synonym_get(newWord,pos='a')["synonyms"])

				if wordset:
					newWord = random.choice(wordset)
					if is_plural:
						newWordP = pattern.en.pluralize(word)
						if newWord != newWordP:
							word = newWord
							if new_sent[-1] in ["an" or "a"]:
								new_sent[-1] = get_an_a(word)
					else:
						word = newWord
						if new_sent[-1] in ["an" or "a"]:
							new_sent[-1] = get_an_a(word)
			new_sent.append(word)

		if t in ["BA", "BS"] and random.uniform(0,1) > 0.5:
			cardinals = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth", "Ninth", "Tenth", "Eleventh", "Twelfth", "Thirteenth", "Fourteenth", "Fifteenth", "Sixteenth", "Seventeenth", "Eighteenth", "Nineteenth", "Twentieth"]
			cardinals_lower = [x.lower() for x in cardinals]
			if set(cardinals) & set(qtree.leaves()) or set(cardinals_lower) & set(qtree.leaves()):
				for i in range(len(new_sent)):
					replacement_set = None
					if new_sent[i] in cardinals:
						replacement_set = cardinals
					elif new_sent[i] in cardinals_lower:
						replacement_set = cardinals_lower
					if replacement_set:
						idx = replacement_set.index(new_sent[i])
						replacement_subset = [k for k in [idx-1, idx+1] if k >= 0 and k < len(replacement_set)]
						new_sent[i] = replacement_set[random.choice(replacement_subset)]

			elif len(qtree)==1 and qtree.height() > 3 and qtree[0].label() in ["S", "SINV"] and len(qtree[0]):
				posList = [b[:2] for _,b in qtree.pos()][1:]
				if posList.count("VB") == 1:
					idx = posList.index("VB")+1
					new_sent = new_sent[:idx]+['not']+new_sent[idx:]
				else:
					idx = 0
					vp_found = False
					add_not = False
					if qtree[0].height() > 2:
						for i in range(len(qtree[0])):
							if not vp_found:
								if qtree[0][i].label() == "VP":
									vp_found = True
									if qtree[0][i].pos()[0][1][:2] == "VB":
										add_not = True
									else:
										add_not = False
								else:
									idx += len(qtree[0][i].leaves())
							elif qtree[0][i].label() == "VP":
								add_not = False
								break
					if add_not and idx > 0:
						new_sent = new_sent[:idx]+['not']+new_sent[idx:]
		#print(qlist[i])
		#print(" ".join(new_sent))
		results.append(" ".join(new_sent))
	return results

def get_an_a(word):
	if word[0] in "aeiouAEIOU":
		return "an"
	else:
		return "a"