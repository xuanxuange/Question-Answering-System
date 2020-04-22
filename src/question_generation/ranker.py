from src.question_generation.nym_utils import get_word_dist_to_root
#init with target sentence length and weights, optionally
#input list of parsetrees to ranker, optionally with list of ints representing value of question type
class Ranker:
	def __init__(self, parser, target_length=14, weights={"height": 0.5, "length": 1, "noun_to_root": 2.0, "sbars": 4.0, "type": 10.0}, type_weights={}):
		self.target_length = target_length
		self.type_weights = type_weights
		self.weights = weights
		self.parser = parser

	def top_n_qtrees(self, tl, n=None, typelist=None):
		if n:
			return self.rank_treelist(tl, typelist=typelist)[:n]
		else:
			return self.rank_treelist(tl, typelist=typelist)

	def rank_treelist(self, tl, typelist=None):
		if not typelist:
			typelist = [" "]*len(tl)
		res = [[self.score_tree(tl[i],typelist[i]),tl[i]] for i in range(len(tl))]
		res.sort(reverse=True)
		return res

	def score_tree(self, t, qtype=0.0):
		score = 0.0
		score -= self.weights["height"]*t.height()**2
		score -= self.weights["length"]*((len(t.leaves()) - self.target_length)**2) #distance ^2
		score += self.weights["sbars"]*self.count_sbars(t)
		score += self.weights["noun_to_root"]*self.avg_dist_to_root(t)
		score += self.type_weights.get(qtype,0.0)*self.weights["type"]
		return score

	def count_sbars(self, t):
		count = 0
		for s in t.subtrees():
			if s.label() == "SBAR":
				count += 1
		return count

	def avg_dist_to_root(self, t):
		total = 0
		nouns = 0
		for word,pos in t.pos():
			if pos in ["NN","NNS","NNP","NNPS"]:
				dist = get_word_dist_to_root(word,pos="n")
				if dist >= 0:
					total += dist
					nouns += 1
		if nouns:
			return total/nouns
		else:
			return 100

	def top_n_qstr(self, qsl,n=None, typelist=None):
		qtrees = []
		for q in qsl:
			try:
				qtree = next(self.parser.parse_text(q, timeout=5000))
				qtrees.append(qtree)
			except:
				pass
		if not typelist:
			typelist = [" "]*len(qsl)
		res = [[self.score_tree(qtrees[i],typelist[i]),qsl[i]] for i in range(len(qtrees))]
		res.sort(reverse=True)
		if n:
			return res[:n]
		else:
			return res

# import nltk
# from nltk.lm.preprocessing import padded_everygram_pipeline
# from nltk.lm import MLE
# brown = nltk.corpus.brown
# train_sentences = brown.sents()
# tokenized_text = [list(map(str.lower, sent)) 
#                 for sent in train_sentences]
# n = 2
# train_data, padded_vocab = padded_everygram_pipeline(n, tokenized_text)
# model = MLE(n)
# model.fit(train_data, padded_vocab)

# test_sentences = ['an apple', 'an ant']
# tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(sent))) 
#                 for sent in test_sentences]

# test_data, _ = padded_everygram_pipeline(n, tokenized_text)
# for test in test_data:
#     print ("MLE Estimates:", [((ngram[-1], ngram[:-1]),model.score(ngram[-1], ngram[:-1])) for ngram in test])

# test_data, _ = padded_everygram_pipeline(n, tokenized_text)
# print(list(test_data))
# for test_sentence in test_sentences:
#   print("PP({0}):{1}".format(test_sentence, model.perplexity(test_sentence.split())))
# for i, test in enumerate(test_data):
#   print("PP({0}):{1}".format(test_sentences[i], model.perplexity(test)))