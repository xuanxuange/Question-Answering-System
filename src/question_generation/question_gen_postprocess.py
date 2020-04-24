from nltk.corpus import wordnet as wn
from src.question_generation.nym_utils import antonym_get, synonym_get, get_nth_hypernyms
import pattern.en
import random

def postprocess(qlist, tlist, parser):
	results = []
	cardinals = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth", "Ninth", "Tenth", "Eleventh", "Twelfth", "Thirteenth", "Fourteenth", "Fifteenth", "Sixteenth", "Seventeenth", "Eighteenth", "Nineteenth", "Twentieth"]
	cardinals_lower = [x.lower() for x in cardinals]

	for i in range(len(qlist)):
		t = tlist[i]
		if t == "AP":
			sent = qlist[i]
			try:
				qwords = qlist[i].split()
				if (set(cardinals) & set(qwords)) or (set(cardinals_lower) & set(qwords)) and random.uniform(0,1) >= 0:
					for j in range(len(qwords)):
						replacement_set = None
						if qwords[j] in cardinals:
							replacement_set = cardinals
						elif qwords[j] in cardinals_lower:
							replacement_set = cardinals_lower
						if replacement_set:
							idx = replacement_set.index(qwords[j])
							replacement_subset = [k for k in [idx-1, idx+1] if k >= 0 and k < len(replacement_set)]
							qwords[j] = replacement_set[random.choice(replacement_subset)]
					qlist[i] = " ".join(qwords)
				if random.uniform(0,1) > 0.5:
					qlist[i] = qlist[i].replace("apt descriptor", "accurate or inaccurate descriptor")
			except:
				results.append(sent)
				continue

		try:
			p = parser.parse_text(qlist[i],timeout=5)
			qtree = next(p)
		except:
			results.append(qlist[i])
			continue

		qpos = qtree.pos()
		new_sent = []
		try:
			for j in range(len(qpos)):
				word, wpos = qpos[j]
				if t == "AP" and wpos in ["NN", "NNS", "JJ"] and word not in ["accurate", "inaccurate", "apt", "descriptor"] and word not in  cardinals and word not in cardinals_lower and random.uniform(0,1) > 0.9:
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

					wordset = [w.split("_") for w in wordset]

					if is_plural:
						wordset = [w for w in wordset if len(w) == 1]

					if wordset:
						newWord = random.choice(wordset)
						if word[0].isupper():
							for i in range(len(newWord)):
								newWord[i] = newWord[i].capitalize()
						newWord = " ".join(newWord)
						if is_plural:
							newWordP = pattern.en.pluralize(word)
							if newWord != newWordP:
								word = newWord
								if new_sent[-1] in ["an", "a"]:
									new_sent[-1] = get_an_a(word)
						else:
							word = newWord
							if new_sent[-1] in ["an", "a"]:
								new_sent[-1] = get_an_a(word)
				new_sent.append(word)
		except:
			new_sent = qtree.leaves()

		if t in ["BA", "BS"] and random.uniform(0,1) > 0.5:
			sent = " ".join(new_sent)
			try:
				if (set(cardinals) & set(qtree.leaves())) or (set(cardinals_lower) & set(qtree.leaves())):
					for j in range(len(new_sent)):
						replacement_set = None
						if new_sent[j] in cardinals:
							replacement_set = cardinals
						elif new_sent[j] in cardinals_lower:
							replacement_set = cardinals_lower
						if replacement_set:
							idx = replacement_set.index(new_sent[j])
							replacement_subset = [k for k in [idx-1, idx+1] if k >= 0 and k < len(replacement_set)]
							new_sent[j] = replacement_set[random.choice(replacement_subset)]

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
							for j in range(len(qtree[0])):
								if not vp_found:
									if qtree[0][j].label() == "VP":
										vp_found = True
										if qtree[0][j].pos()[0][1][:2] == "VB":
											add_not = True
										else:
											add_not = False
									else:
										idx += len(qtree[0][j].leaves())
								elif qtree[0][j].label() == "VP":
									add_not = False
									break
						if add_not and idx > 0:
							new_sent = new_sent[:idx]+['not']+new_sent[idx:]
			except:
				new_sent = [sent]

		#print(qlist[i])
		#print(" ".join(new_sent))
		new_sent = " ".join(new_sent).replace(" ''","\"").replace("`` ","\"")
		results.append(new_sent)
	return results

def get_an_a(word):
	if word[0] in "aeiouAEIOU":
		return "an"
	else:
		return "a"