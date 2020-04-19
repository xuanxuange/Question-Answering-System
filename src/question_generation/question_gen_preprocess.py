from collections import defaultdict
import re
from queue import Queue, LifoQueue
import neuralcoref
import spacy

nlp = spacy.load('en_core_web_lg')
neuralcoref.add_to_pipe(nlp)

class SenTree:
	def __init__(self, t, parser, prevST=None, nextST=None, ner=None, parent=None):
		self.t = t
		self.spacy = nlp
		self.metadata = None
		if parent is not None:
			self.metadata = parent.metadata
		self.ner = ner
		self.parser = parser
		self.fulltext = reconstitute_sentence(t.leaves()) if t else ""
		self.text = t.leaves()
		self.type = None
		self.flags = defaultdict(lambda:False)
		self.parentheticals = None
		self.turns = None
		self.children = defaultdict(list)
		self.prevST = prevST
		self.nextST = nextST
		self.to_be_swapped = False
		return

	#1 Replace <NN_> <PRP> turns of phrase

	#2 Replace <has been> <___> <to be> turns of phrase
	def tobe_turn_of_phrase(self, text=None, pos=None,turns = []):
		if text is None:
			text = self.text
		if pos is None:
			pos = [p[1] for p in self.t.pos()]
		stage_num = 2
		first = [("has", "been"),("have", "been"), ("had", "been")]
		last = [["to", "be"],["as"]]
		restext = None
		respos = None
		if len(text) > 4:
			for i in range(len(text)-4):
				if tuple(text[i:i+2]) in first and (tuple(text[i+3:i+5]) == ("to", "be") or text[i+3] == "as"):
					start = i
					replace = "is"
					if i > 0 and text[i] == "have" and text[i-1] == "will":
						start -= 1
						replace = "will be"
					elif text[i] == "have":
						replace = "are"
					elif text[i] == "had":
						replace = "was"
						revpos = pos[:i:-1]
						a = revpos.index("NNS") + 1
						b = revpos.index("NNPS") + 1
						c = revpos.index("NN") + 1
						d = revpos.index("NNP") + 1
						if ((a or b) and (c or d) and min(a,b) < min(c,d)) or ((a or b) and not (c or d)):
							replace = "were"

					end = i+5
					if text[i+3] == "as":
						end -= 1
					turns.append((start,text[i+2]))
					restext = text[:start]+[replace]+text[end:]
					respos = pos[:start]+["REPLACED"]+pos[end:]
					break
		if restext:
			return self.tobe_turn_of_phrase(restext, respos, turns)
		elif turns:
			newtree = self.parser.raw_parse(reconstitute_sentence(text), timeout=5000)
			newtree = next(newtree)
			child = SenTree(newtree, self.parser, prevST=self.prevST, nextST=self.nextST, ner=self.ner)
			child.type = stage_num
			child.flags = self.flags
			child.turns = self.turns
			child.flags["turns_removed"] = True
			self.children[stage_num] = [child]
			if self.prevST is not None:
				self.prevST.nextST = child
			if self.nextST is not None:
				self.nextST.prevST = child
			return True
		return False

	#3 Fix/Separate SINV phraseology
	def fix_sinv(self):
		return False

	#4 Parenthetical removal
	def parenthetical_removal(self):
		result = []
		count = 0
		stage_num = 4
		start = None
		parentheticals = []
		# print(self.text)
		for word in self.text:
			if word == "-LRB-":
				if count == 0:
					parentheticals.append({"index" : len(result), "text":"("})
				else:
					parentheticals[-1]["text"] += " ("
				count += 1
			elif word == "-RRB-":
				count -= 1
				parentheticals[-1]["text"] += " )"
			elif count == 0:
				result.append(word)
			else:
				parentheticals[-1]["text"] += (" " + word)
		if parentheticals:
			if len(result) > 0:
				temp = reconstitute_sentence(result)
				# print(temp)
				newtree = self.parser.raw_parse(temp, timeout=5000)
				newtree = next(newtree)
				child = SenTree(newtree, self.parser, prevST=self.prevST, nextST=self.nextST, ner=self.ner)
				# print(newtree.leaves())
				child.type = stage_num
				child.flags = self.flags
				child.parentheticals = parentheticals
				self.children[stage_num] = [child]
				child.flags["parentheticals_removed"] = True
				if self.prevST is not None:
					self.prevST.nextST = child
				if self.nextST is not None:
					self.nextST.prevST = child
				return True
		return False

	#5 Run apositive removal/manipulation
	def appositive_removal(self, immediate_questions):
		# can generate "IS <A> an apt descriptor for <B>?"
		return False

	#6 Remove NP-prefixed SBAR
	def sbar_remove(self, immediate_questions):
		stage_num = 6
		for i in range(len(self.t)-1):
			if self.t[i].label() == "NP" and self.t[i+1].label() == "VP" and self.t[i+1][0].label()[:2] == "VB" and self.t[i+1][1].label() == "NP" and self.t[i+1][2].label() == "SBAR":
				self.flags["SBAR_removal_applied"] = True
				new_sent = ""
				for k in range(i+1):
					new_sent += reconstitute_sentence(self.t[k].leaves()) + " "
				new_sent += reconstitute_sentence(self.t[i+1].leaves()[:2]+self.t[i+1].leaves()[3:]) + " "
				for m in range(i+2,len(self.t)):
					new_sent += reconstitute_sentence(self.t[m].leaves()) + " "
				new_sent = new_sent[:-1]
				newtree = self.parser.raw_parse(new_sent)
				newtree = next(newtree)
				child = SenTree(newtree, self.parser, prevST=self.prevST, nextST=self.nextST, ner=self.ner)
				child.type = stage_num
				child.flags = self.flags
				self.children[stage_num] = [child]
				if self.prevST is not None:
					self.prevST.nextST = child
				if self.nextST is not None:
					self.nextST.prevST = child
				# print(reconstitute_sentence(self.t[i+1][2].leaves()))

				temp = "What did " + reconstitute_sentence(self.t[i].leaves()) + " " + reconstitute_sentence(self.t[i+1][0].leaves()) + " " + reconstitute_sentence(self.t[i+1][2].leaves()) + "?"
				pattern = re.compile(r'\.\s*\?')
				immediate_questions.append(pattern.sub('?', temp))
				return True
			elif self.t[i].label() == "NP" and self.t[i+1].label() == "," and self.t[i+2].label() == "SBAR" and self.t[i+2][0].label()[:4] == "WHMP" and self.t[i+2][1].label() == "S" and self.t[i+2][1][-1].label() == "VP":
				self.flags["SBAR_removal_applied"] = True
				print(reconstitute_sentence(self.t[i+1][2].leaves()))
				temp = "Who or what " + reconstitute_sentence(self.t[i+2][1][-1].leaves()) + "?"
				pattern = re.compile(r'\.\s*\?')
				immediate_questions.append(pattern.sub('?', temp))
				return True
		return False
	
	#7 Separate root-level <S> <CC> <S>
	def s_cc_s_separation(self):
		stage_num = 7
		for i in range(len(self.t[0])-2):
			if (valid_s(self.t[0][i]) and self.t[0][i+1].label() == "CC" and valid_s(self.t[0][i+2])):
				newtree1 = self.parser.raw_parse(reconstitute_sentence(self.t[0][i].leaves() + ['.']), timeout=5000)
				newtree1 = next(newtree1)
				newtree2 = self.parser.raw_parse(reconstitute_sentence(self.t[0][i+2].leaves() + ['.']), timeout=5000)
				newtree2 = next(newtree2)
				child1 = SenTree(newtree1, self.parser, prevST=self.prevST, nextST=self.nextST, ner=self.ner)
				child1.type = stage_num
				child1.flags = self.flags
				child2 = SenTree(newtree2, self.parser, prevST=self.prevST, nextST=self.nextST, ner=self.ner)
				child2.type = stage_num
				child2.flags = self.flags
				self.children[stage_num] = [child1, child2]
				if self.prevST is not None:
					self.prevST.nextST = child1
				if self.nextST is not None:
					self.nextST.prevST = child2
				child1.nextST = child2
				child2.prevST = child1
				return True
			elif i+3 < len(self.t[0]) and valid_s(self.t[0][i]) and self.t[0][i+1].label() == "," and self.t[0][i+2].label() == "CC" and valid_s(self.t[0][i+3]):
				newtree1 = self.parser.raw_parse(reconstitute_sentence(self.t[0][i].leaves() + ['.']), timeout=5000)
				newtree1 = next(newtree1)
				newtree2 = self.parser.raw_parse(reconstitute_sentence(self.t[0][i+3].leaves() + ['.']), timeout=5000)
				newtree2 = next(newtree2)
				child1 = SenTree(newtree1, self.parser, prevST=self.prevST, nextST=self.nextST, ner=self.ner)
				child1.type = stage_num
				child1.flags = self.flags
				child2 = SenTree(newtree2, self.parser, prevST=self.prevST, nextST=self.nextST, ner=self.ner)
				child2.type = stage_num
				child2.flags = self.flags
				self.children[stage_num] = [child1, child2]
				if self.prevST is not None:
					self.prevST.nextST = child1
				if self.nextST is not None:
					self.nextST.prevST = child2
				child1.nextST = child2
				child2.prevST = child1
				return True
		return False
	
	#8 NER coreference resolution
	def run_ner(self):
		stage_num = 8
		if self.ner is None and self.text[-1] == ".":
			count,document_fulltext = self.update_ner()
			if self.ner._.has_coref:
				doc_info = self.ner

				test = [token.text for token in doc_info]
				threshold = 0
				if count > 1:
					print(test)
					threshold = len(test) - 1 - test[:-1][::-1].index('.')
					print("Detected threshold: " + str(threshold) + " -- " + str(doc_info[threshold].text))
				# test = list(self.parser.tokenize(test))
				original_pos = [token.pos_ for token in doc_info][threshold:]
				# print("Found coref in: ")
				for st in document_fulltext.split('. '):
					# print(st)
					pass
				# print("------------")

				replace_operations = []
				generic_descriptors = ["he", "his", "her", "hers", "she", "him", "they", "their", "them", "theirs", "it", "its"]
				for cluster in doc_info._.coref_clusters:
					for mention in cluster.mentions:
						if mention.start >= threshold and mention.text != cluster.main.text and cluster.main.text.lower() not in generic_descriptors:
							main_ner = [token.ent_type_ for token in cluster.main]
							mention_pos = self.corenlp_pos(mention)
							mention_text = [token.text for token in mention]
							mention_ner = [token.ent_type_ for token in mention]

							main_text = cluster.main.text
							main_pos = self.corenlp_pos(cluster.main)
							if not self.match_ner(main_ner, mention_ner, main_pos, mention_pos, mention_text, cluster.main, mention.start):
								main_text = self.custom_ner(mention_ner, mention_text, mention.start)

							corrected_main = main_text
							if corrected_main[-2:] == "\'s":
								main_text = corrected_main[:-2]
							elif corrected_main[-2:] == "s\'":
								main_text = corrected_main[:-1]
							elif corrected_main[-1] == "s":
								corrected_main += "\'"
							else:
								corrected_main += "\'s"

							print("metadata:")
							print("Main position: (" + str(cluster.main.start) + ", " + str(cluster.main.end) + ")")
							# print(mention.start - threshold)
							# print(mention.end - threshold)

							# print(threshold)
							# print(original_pos)

							# print(cluster.main.text)
							# print(main_ner)
							# print(main_pos)
							print(mention.text)
							# print(mention_ner)
							print(mention_pos)
							# print([token.text for token in doc_info][threshold:])
							if 'PRP$' in mention_pos or mention_pos[-1] == 'POS':
								if 'PRP$' in main_pos:
									print("Chosen replacement: " + main_text)
									replace_operations.append((mention.start - threshold, mention.end - threshold, main_text))
								else:
									print("Chosen replacement: " + corrected_main)
									replace_operations.append((mention.start - threshold, mention.end - threshold, corrected_main))
							elif len(original_pos) > mention.end - threshold and original_pos[mention.end - threshold][1] == "POS":
								if 'PRP$' in main_pos:
									print("Chosen replacement: " + main_text)
									replace_operations.append((mention.start - threshold, mention.end - threshold + 1, main_text))
								else:
									print("Chosen replacement: " + corrected_main)
									replace_operations.append((mention.start - threshold, mention.end - threshold + 1, corrected_main))
							else:
								print("Chosen replacement: " + main_text)
								replace_operations.append((mention.start - threshold, mention.end - threshold, main_text))
							print(main_pos)
							print("------------------------\n")

				acc = 0
				test = test[threshold:]
				print("detected last sentencece")
				print(test)
				for operation in replace_operations:
					start = operation[0] + acc
					end_p = operation[1] + acc
					replace_text = operation[2].split()
					# print(str(start) + ", " + str(end_p) + ", " + operation[2])
					replace_size = len(replace_text)

					test = test[:start] + replace_text + test[end_p:]

					acc += replace_size - (end_p - start)

				test = reconstitute_sentence(test)
				test = test.replace(" - ", "-")
				print("post_reconstitution: " + test)

				newtree = self.parser.raw_parse(test)
				newtree = next(newtree)
				child = SenTree(newtree, self.parser, prevST=self.prevST, nextST=self.nextST, ner=self.ner)
				if self.prevST is not None:
					self.prevST.nextST = child
				if self.nextST is not None:
					self.nextST.prevST = child
				self.children[stage_num] = [child]
				# print("YAY: [" + self.fulltext + "]\n====> [" + child.fulltext + "]\n")
				return True
		return False

	#9 Rearrange <SBAR/PP>, <S> into <S> <SBAR/PP>
	def sbarpp_s_rearrange(self):
		stage_num = 9
		#if self.t[0].label() in ["PP", "SBAR"] and valid_s(self.t[1]):
		#	newtree1 = self.parser.raw_parse(reconstitute_sentence(self.t[1].leaves())+" "+reconstitute_sentence(self.t[0].leaves()), timeout=5000)
		#	newtree1 = next(newtree1)
		#	newtree2 = self.parser.raw_parse(reconstitute_sentence(self.t[1].leaves()), timeout=5000)
		#	newtree2 = next(newtree2)
		#	child1 = SenTree(newtree1, self.parser, prevST=self.prevST, nextST=self.nextST, ner=self.ner)
		#	child1.type = stage_num
		#	child1.flags = self.flags
		#	child2 = SenTree(newtree2, self.parser, prevST=self.prevST, nextST=self.nextST, ner=self.ner)
		#	child2.type = stage_num
		#	child2.flags = self.flags
		#	self.children[stage_num] = [child1, child2]
		#	if self.prevST is not None:
		#		self.prevST.nextST = child1
		#	if self.nextST is not None:
		#		self.nextST.prevST = child2
		#	return True
		#el
		if self.t[0].label() in ["PP", "SBAR"] and self.t[1].label() == "," and valid_s(self.t[2]):
			newtree1 = self.parser.raw_parse(reconstitute_sentence(self.t[2].leaves())+" "+reconstitute_sentence(self.t[0].leaves()), timeout=5000)
			newtree1 = next(newtree1)
			newtree2 = self.parser.raw_parse(reconstitute_sentence(self.t[2].leaves()), timeout=5000)
			newtree2 = next(newtree2)
			child2 = SenTree(newtree1, self.parser, prevST=self.prevST, nextST=self.nextST, ner=self.ner)
			child1.type = stage_num
			child1.flags = self.flags
			child1 = SenTree(newtree2, self.parser, prevST=self.prevST, nextST=self.nextST, ner=self.ner)
			child2.type = stage_num
			child2.flags = self.flags
			self.children[stage_num] = [child1, child2]
			if self.prevST is not None:
				self.prevST.nextST = child1
			if self.nextST is not None:
				self.nextST.prevST = child2
			child1.nextST = child2
			child2.prevST = child1
			return True
		return False

	#10 Rearrange <NP> <VP>'s <NP> components based on complexity if verb is "to be" - top-level only atm
	def to_be_equiv(self):
		stage_num = 10
		if self.flags["to_be_swapped"]:
			return
		else:
			self.flags["to_be_swapped"] = True
			self.children[stage_num] = []
		to_be_conj = ["be", "am", "is", "are", "was", "were", "been", "being"]
		for i in range(len(self.t)-1):
			if valid_np(self.t[i]) and valid_vp(self.t[i+1]) and len(self.t[i+1] > 1) and self.t[i+1][0].label()[:2] == "VB" and valid_np(self.t[i+1][1]):
				if reconstitute_sentence(self.t[i+1][0].leaves()) in to_be_conj:
					new_sent = ""
					for k in range(i):
						new_sent += reconstitute_sentence(self.t[k].leaves()) + " "
					new_sent += reconstitute_sentence(self.t[i+1][1].leaves()) + " "
					new_sent += reconstitute_sentence(self.t[i+1][1].leaves()) + " "
					new_sent += reconstitute_sentence(self.t[i].leaves()) + " "
					for m in range(i+2,len(self.t)):
						new_sent += reconstitute_sentence(self.t[m].leaves()) + " "
					new_sent = new_sent[:-1]
					newtree = self.parser.raw_parse(new_sent)
					newtree = next(newtree)
					child = SenTree(newtree, self.parser, prevST=self.prevST, nextST=self.nextST, ner=self.ner)
					child.type = stage_num
					child.flags = self.flags
					self.children[stage_num].append(child)
		return False

	#11 Include every valid <NP> <VP> combo in the sentence
	def npvp_combo(self):
		stage_num = 11
		subs = self.t.subtrees()
		self.flags["npvp_extracted"] = True
		self.children[stage_num] = []
		for sub in subs:
			if valid_s(sub):
				newtree = self.parser.raw_parse(reconstitute_sentence(sub.leaves()))
				newtree = next(newtree)
				child = SenTree(newtree, self.parser, prevST=self.prevST, nextST=self.nextST, ner=self.ner)
				child.type = stage_num
				child.flags = self.flags
				self.children[stage_num].append(child)
		return False
	
	# Wrapper to handle stage progression
	def handle_stage(self, stage, immediate_questions):
		if stage == 1:
			# Replace <NN_> <PRP> turns of phrase
			# NOT IMPLEMENTED
			pass
		elif stage == 2:
			# Replace <has been> <___> <to be> turns of phrase
			return self.tobe_turn_of_phrase()
		elif stage == 3:
			# Fix/Separate SINV phraseology
			# NOT IMPLEMENTED
			return self.fix_sinv()
		elif stage == 4:
			# Parenthetical removal
			return self.parenthetical_removal()
		elif stage == 5:
			# Run apositive removal/manipulation
			# NOT IMPLEMENTED
			return self.appositive_removal(immediate_questions)
		elif stage == 6:
			# Remove NP-prefixed SBAR
			return self.sbar_remove(immediate_questions)
		elif stage == 7:
			# Separate root-level <S> <CC> <S>
			return self.s_cc_s_separation()
		elif stage == 8:
			# Run coreference to replace ambiguous sentences
			return self.run_ner()
		elif stage == 9:
			# Rearrange <SBAR/PP>, <S> into <S> <SBAR/PP>
			return self.sbarpp_s_rearrange()
		elif stage == 10:
			# Rearrange <NP> <VP>'s <NP> components based on complexity if verb is "to be"
			return self.to_be_equiv()
		elif stage == 11:
			# Include every valid <NP> <VP> combo in the sentence
			return self.npvp_combo()
		return False

	# SpaCY has a coarser POS tagger. Here, we fetch the SpaCY mention's POS tags, according to the Stanford CoreNLP tagset
	def corenlp_pos(self, mention):
		# corenlp_tokens = next(self.parser.raw_parse(reconstitute_sentence([token.text for token in mention]).replace(" - ", "-")))
		doc_len = len(self.ner)
		ind = doc_len - 2
		count = 0
		last_per = doc_len - 1
		while ind >= mention.end:
			if self.ner[ind].text == ".":
				last_per = ind
				count += 1
			ind -= 1
		target_sent = self
		for i in range(count):
			target_sent = target_sent.prevST
			while target_sent is not None and target_sent.text[-1] != ".":
				target_sent = target_sent.prevST
		hyp_loc = max(0, len(target_sent.text) - 1 - max(0, (last_per - mention.start)))

		# print("Searching for: [" + reconstitute_sentence([token.text for token in mention]).replace(" - ", "-") + "] in [" + target_sent.fulltext + "]")

		fixed_spacy = []
		ind = 0
		while ind < len(mention):
			test = mention[ind].text
			if test == '-' and ind + 1 < len(mention):
				fixed_spacy[-1] += '-' + mention[ind + 1].text
				ind += 1
			else:
				fixed_spacy.append(test)
			ind += 1

		found_loc = None
		match_ind = 0
		# Search forward
		print("Looking forward in range: " + str(hyp_loc) + ", " + str(len(target_sent.text)))
		for i in range(hyp_loc, len(target_sent.text)):
			test = target_sent.text[i]
			if test == fixed_spacy[match_ind]:
				match_ind += 1
			if match_ind == len(fixed_spacy):
				found_loc = i - len(fixed_spacy) + 1
				break

		# Search backwards
		for i in range(hyp_loc, -1, -1):
			test = target_sent.text[i]
			if test == fixed_spacy[0]:
				success = True
				for j in range(1, len(fixed_spacy)):
					if fixed_spacy[j]!= target_sent.text[i+j]:
						success = False
						break
				if success:
					test_dist = abs(i - hyp_loc)
					if found_loc is None or test_dist < abs(found_loc - hyp_loc):
						found_loc = i
					break
		if found_loc is not None:
			return [pair[1] for pair in target_sent.t.pos()[found_loc: found_loc + len(fixed_spacy)]]
		else:
			print("Could not find :")
			print(fixed_spacy)
			print("in:")
			print(target_sent.text)
			return None

	# Check if spacy neuralcoref worked properly
	# Sometimes, it's not the first NP that needs to match, but only a part of it.
	# Ex: Akhenaten's wife corresponds with his, whereas the part that should correspond is just Akhenaten
	def match_ner(self, main_ner, mention_ner, main_pos, mention_pos, mention_text, main, mention_start):
		# PERSON		People, including fictional.
		# NORP			Nationalities or religious or political groups.
		# FAC			Buildings, airports, highways, bridges, etc.
		# ORG			Companies, agencies, institutions, etc.
		# GPE			Countries, cities, states.
		# LOC			Non-GPE locations, mountain ranges, bodies of water.
		# PRODUCT		Objects, vehicles, foods, etc. (Not services.)
		# EVENT			Named hurricanes, battles, wars, sports events, etc.
		# WORK_OF_ART	Titles of books, songs, etc.
		# LAW			Named documents made into laws.
		# LANGUAGE		Any named language.
		# DATE			Absolute or relative dates or periods.
		# TIME			Times smaller than a day.
		# PERCENT		Percentage, including ”%“.
		# MONEY			Monetary values, including unit.
		# QUANTITY		Measurements, as of weight or distance.
		# ORDINAL		“first”, “second”, etc.
		# CARDINAL		Numerals that do not fall under another type.
		# print(main.label)
		# print(main_ner)
		# print(mention_ner)
		if len(mention_text) == 1:
			if mention_text[0] in ['it', 'its']:
				# Should not be a person
				return True
			elif mention_text[0] in ['they', 'them', 'their', 'theirs']:
				# Check if origin is: NORP, ORG, PERSON (Plural)
				# print(pos)
				pass
			elif mention_text[0] in ['he', 'his', 'him']:
				# Check if origin is PERSON (Singular, Male)
				# print(pos)
				pass
			elif mention_text[0] in ['she', 'hers', 'her']:
				# Check if origin is PERSON (Singular, Female)
				# print(pos)
				pass
			elif mention_text[0] in ['I', 'me', 'my', 'mine', 'we', 'our', 'us', 'ours', 'you', 'your', 'yours']:
				return True
		return True

	# Custom NER for when the provided coref fails
	def custom_ner(self, mention_ner, mention_text, mention_start):
		return reconstitute_sentence(mention_text)

	# Update the NER data of this SenTree
	def update_ner(self, lookback=10):
		document_stack = LifoQueue()
		document_stack.put_nowait(self)
		curr = self.prevST
		count = 1
		while curr is not None and count <= lookback:
			if curr.text[-1] == ".":
				document_stack.put_nowait(curr)
				count += 1
			curr = curr.prevST

		document_fulltext = ""
		while not document_stack.empty():
			curr = document_stack.get_nowait()
			document_fulltext += curr.fulltext + " "
		pattern = re.compile(r'([^ ])\. ')
		self.ner = nlp(pattern.sub(r"\1 . ", document_fulltext))
		return count,document_fulltext

	# Assumes NER data is up to date for this SenTree
	# Gets you a 
	def align_ner(self):
		# Match the subtree things
		test_spacy = self.ner.text.split('.')[-2].split() + ['.']
		separate_inds = []
		replacements = []

		to_print = False
		if len(test_spacy) != len(self.text):
			to_print = True
		else:
			for i in range(len(test_spacy)):
				if test_spacy[i] != self.text[i]:
					to_print = True
					break

		if to_print:
			print("COMPARE SPACY to CoreNLP")
			print(test_spacy)
			print(self.text)
			print("=======================================\n")
		return


def reconstitute_sentence(raw):
	found_quote = False
	found_single = False
	reconstruct_quotes = ""
	for elem in raw:
		space = " "
		
		if (elem == '\"' and not found_quote) or elem == "``":
			elem = '\"'
			space = ""
			found_quote = True
		elif elem == "`":
			elem = '\"'
			space = ""
			found_single = True
		elif elem == '\"' or elem == "\'\'": # already saw a ", but found_quote = False, so now it's True ==> ending quote
			elem = '\"'
			reconstruct_quotes = reconstruct_quotes[:-1] # eliminate space before the quotation
			found_quote = False
		elif elem == "\'" and found_single:
			elem = '\"'
			reconstruct_quotes = reconstruct_quotes[:-1] # eliminate space before the quotation
			found_single = False
		reconstruct_quotes += elem + space
	if reconstruct_quotes[-2:] == ". ":
		reconstruct_quotes = reconstruct_quotes[:-1]
	pattern1 = re.compile(r' \.')
	pattern2 = re.compile(r' \'s ')
	pattern3 = re.compile(r' ,')
	return pattern1.sub('.', pattern2.sub('\'s ', pattern3.sub(',', reconstruct_quotes))).replace(' \' ', '\' ')

def valid_np(t):
	if t.label() != "NP":
		return False
	else:
		return has_valid_np(t)

def valid_vp(t):
	if t.label() != "VP":
		return False
	else:
		return has_valid_vp(t)

def has_valid_np(t):
	pos_trunc = [p[1][:2] for p in t.pos()]
	return "NN" in pos_trunc

def has_valid_vp(t):
	pos_trunc = [p[1][:2] for p in t.pos()]
	return "VB" in pos_trunc

def valid_s(t):
	#Maybe remove this
	if t.label() == "ROOT":
		return valid_s(t[0])
	elif t.label() != "S":
		return False
	else:
		np_found = False
		for i in range(len(t)):
			if np_found or valid_np(t[i]):
				np_found = True
			if np_found and valid_vp(t[i]):
				return True
	return False

def validate_s(t):
	if t.label() == "ROOT":
		return valid_s(t[0])
	elif t.label() != "S":
		return False
	else:
		return has_valid_np(t) and has_valid_vp(t)
	return False

def acc_stage(stage):
	test = stage + 1
	if stage == 10:
		return True,3
	else:
		return False,test

def preprocess(treelist, parser):
	preprocessed_questions = []
	preprocessed_trees = []
	root_list = []

	FrontierQueue = Queue()

	ind = 0
	for i in range(len(treelist)):
		tree = treelist[i]
		if validate_s(tree):
			root = SenTree(tree, parser)
			root_list.append(root)

			if ind > 0:
				root.prevST = root_list[ind - 1]
				root_list[ind - 1].nextST = root
			ind += 1

	updated_root = None
	for root in root_list:

		node_list = [root]
		keep_bools = [True]

		FrontierQueue.put_nowait((0, 1))

		while not FrontierQueue.empty():
			node_id,stage = FrontierQueue.get_nowait()
			curr_node = node_list[node_id]
			print("Stage " + str(stage) + ":")
			print(curr_node.text)
			print(curr_node.fulltext)
			print("-----------------------\n")
			# print(str(stage) + ": " + reconstitute_sentence(curr_node.t.leaves()))
			full_replace = curr_node.handle_stage(stage, preprocessed_questions)
			if full_replace:
				# print("What: " + str(stage))
				keep_bools[node_id] = False
			if len(curr_node.children[stage]) > 0:
				# print("Ho: " + str(stage))
				rollover, new_stage = acc_stage(stage)
				# Currently only does one passthrough
				if not rollover:
					for child in curr_node.children[stage]:
						FrontierQueue.put_nowait((len(node_list), new_stage))
						node_list.append(child)
						keep_bools.append(True)
			else:
				rollover, new_stage = acc_stage(stage)
				if not rollover:
					FrontierQueue.put_nowait((node_id, new_stage))

		preprocessed_trees += [node_list[i] for i in range(len(node_list)) if keep_bools[i]]
	for tree in preprocessed_trees:
		if tree.text[-1] == ".":
			tree.update_ner()
			tree.align_ner()
	return preprocessed_trees, preprocessed_questions