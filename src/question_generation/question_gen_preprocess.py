from collections import defaultdict
import re
from queue import Queue, LifoQueue

import neuralcoref
import spacy

import nltk
from nltk.tag import StanfordNERTagger

from pattern.en import lemma


#st = StanfordNERTagger('/Users/Thomas/Documents/11-411/NER/classifiers/english.all.3class.distsim.crf.ser.gz', '/Users/Thomas/Documents/11-411/NER/stanford-ner.jar', encoding='utf-8')
st = None
nlp = spacy.load('en_core_web_lg')
# nlp = spacy.load('en')
neuralcoref.add_to_pipe(nlp)

document_metadata = {}
document_metadata["coref"] = []

generic_NP = [['it', 'its'],
	['they', 'them', 'their', 'theirs'],
	['he', 'his', 'him'],
	['she', 'hers', 'her'],
	['i', 'me', 'my', 'mine', 'we', 'our', 'us', 'ours', 'you', 'your', 'yours']]


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

	def update_text(self):
		self.text = self.t.leaves()
		self.filltext = reconstitute_sentence(t.leaves()) if self.t else ""

	#1 Replace <NN_> <PRP> turns of phrase

	#2 Replace <has been> <___> <to be> turns of phrase
	def tobe_turn_of_phrase(self, text=None, pos=None, turns=[], stage_num=2):
		if text is None:
			text = self.text
		if pos is None:
			pos = [p[1] for p in self.t.pos()]
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
			newtree = self.parser.parse_text(reconstitute_sentence(text), timeout=5000)
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

	#3 Remove removable prefixes
	def remove_prefix(self, stage_num=3):
		if self.t[0].label() == "S":
			S = self.t[0]
			if len(S) >= 5 and S[0].label() != "NP" and S[1].label() == ",":
				found_NP = False
				validated = False
				for i in range(2, len(S)):
					if found_NP or has_valid_np(S[i]):
						found_NP = True
					if found_NP and has_valid_vp(S[i]):
						validated = True
				if validated:
					print("REMOVED PREFIX TEXT [" + S[0].label() + "]: " + reconstitute_sentence(S[0].leaves()))

					result = []
					for i in range(2, len(S)):
						result += S[i].leaves()

					temp = reconstitute_sentence(result)

					newtree = self.parser.parse_text(temp, timeout=5000)
					newtree = next(newtree)
					child = SenTree(newtree, self.parser, prevST=self.prevST, nextST=self.nextST)
					# print(newtree.leaves())
					child.type = stage_num
					child.flags = self.flags
					self.children[stage_num] = [child]
					if self.prevST is not None:
						self.prevST.nextST = child
					if self.nextST is not None:
						self.nextST.prevST = child
					return True
		return False

	#4 Parenthetical removal
	def parenthetical_removal(self, stage_num=4):
		result = []
		count = 0
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
				newtree = self.parser.parse_text(temp, timeout=5000)
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
	def appositive_removal(self, immediate_questions, stage_num=5):
		# can generate "IS <A> an apt descriptor for <B>?"
		delims = [";", ":", ",", "."]
		allowables = ["NP", "PP", "SBAR", "S", "NN"]
		retval = False
		tree_string = " ".join(self.t.leaves())
		for s in list(self.t.subtrees()):
			if s.label() == "NP" and s.height() > 2:
				if len(s) >= 4:
					a = 0
					is_list = False
					state = 0
					while(a < len(s)):
						if state == 0 :
							if s[a].label() in ["NP", "NN"]:
								a += 1
								state = 1
							else:
								break
						elif state == 1:
							if s[a].label() == ",":
								a += 1
								state = 2
							elif s[a].label() == "CC":
								a += 1
								state = 3
							else:
								break
						elif state == 2:
							if s[a].label() == "CC":
								a += 1
								state = 3
							elif s[a].label() in ["NP", "NN"]:
								a += 1
								state = 1
							else:
								break
						else:
							if s[a].label() in ["NP", "NN"]:
								is_list = True
								print("SKIPPING, since FOUND LIST in :")
								s.pretty_print()
								break
					i = 0
					appositives_and_delims = []
					while not is_list and (i < len(s)-3):
						if s[i].label() in ["NP", "NN"] and s[i+1].label() in delims and s[i+2].label() in allowables and s[i+3].label() in delims:
							if s[i+2].pos()[0][1] != "CC":
								appositives_and_delims.append(i+1)
								appositives_and_delims.append(i+2)
								print(" ".join(s[i].leaves()) + " is NP to the appositive " + " ".join(s[i+2].leaves()))
								immediate_questions.append("Is "+" ".join(s[i+2].leaves()) + " an apt descriptor for" + " ".join(s[i].leaves())+"?")
								retval = True
							else:
								print("SKIPPING child %s, since FOUND LIST in child %s of :" % str(i+2),str(i+2))
								s.pretty_print()
						i += 1

					if not is_list and len(s) > 2 and s[-3].label() in ["NP", "NN"] and s[-2].label() in delims and s[-1].label() in allowables:
						s_idx = -1
						try:
							s_idx = tree_string.index(" ".join(s.leaves()))
						except:
							pass
						if s_idx >= 0:
							next_idx = s_idx + len(" ".join(s.leaves()))+1
						if s[-1].pos()[0][1] != "CC" and next_idx < len(tree_string) and tree_string[next_idx] in delims:
							appositives_and_delims.append(len(s)-1)
							appositives_and_delims.append(len(s)-2)
							print(" ".join(s[-3].leaves()) + " is NP to the appositive " + " ".join(s[-1].leaves()))
							immediate_questions.append("Is "+" ".join(s[-1].leaves()) + " an apt descriptor for" + " ".join(s[-3].leaves())+"?")
							retval = True

					appositives_and_delims.sort(reverse=True)
					for idx in appositives_and_delims:
						s.__delitem__(idx)

		for s in self.t.subtrees():
			if s.height() > 1 and s[0] != str(s[0]):
				for i in range(len(s)-1,-1,-1):
					if not s[i].leaves():
						s.__delitem__(i)

		self.update_text()
		return retval

	#6 Remove NP-prefixed SBAR
	# Should only remove trailing SBAR, and only when no CC, because that tends to mean there's a list, so the parser messed up
	def sbar_remove(self, immediate_questions, stage_num=6):
		curr = self.t[0]
		acc_left = []
		if len(curr) >= 3:
			for i in range(len(curr) - 2):
				acc_left += curr[i].leaves()
			curr = curr[-2]
			prior_NP = True
			while curr.height() > 2:
				found_np = False
				for i in range(len(curr) - 1):
					if found_np or curr[i].label() == "NP":
						found_np = True
					acc_left += curr[i].leaves()
				if found_np and len(curr) >= 3 and curr[-3].label() == "NP" and curr[-2].label() == "," and curr[-1].label() == "SBAR":
					test_pos = [pair[1] for pair in curr[-1].pos()]
					# Check for no CC
					if "CC" not in test_pos:
						if acc_left[-1] == ",":
							acc_left = acc_left[:-1]
						immediate_questions += getSBARQuestion(curr, self.t)

						temp = reconstitute_sentence(acc_left + ["."])

						newtree = self.parser.parse_text(temp, timeout=5000)
						newtree = next(newtree)
						child = SenTree(newtree, self.parser, prevST=self.prevST, nextST=self.nextST)
						# print(newtree.leaves())
						child.type = stage_num
						child.flags = self.flags
						self.children[stage_num] = [child]
						if self.prevST is not None:
							self.prevST.nextST = child
						if self.nextST is not None:
							self.nextST.prevST = child
						

						print("REMOVED TRAILING SBAR. CHANGE [" + self.fulltext + "]")
						print("=====> [" + child.fulltext + "]\n")

						return True
				curr = curr[-1]
		return False
	
	#7 Separate root-level <S> <CC> <S>
	def s_cc_s_separation(self, stage_num=7):
		for i in range(len(self.t[0])-2):
			if (valid_s(self.t[0][i]) and self.t[0][i+1].label() == "CC" and valid_s(self.t[0][i+2])):
				newtree1 = self.parser.parse_text(reconstitute_sentence(self.t[0][i].leaves() + ['.']), timeout=5000)
				newtree1 = next(newtree1)
				newtree2 = self.parser.parse_text(reconstitute_sentence(self.t[0][i+2].leaves() + ['.']), timeout=5000)
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
				newtree1 = self.parser.parse_text(reconstitute_sentence(self.t[0][i].leaves() + ['.']), timeout=5000)
				newtree1 = next(newtree1)
				newtree2 = self.parser.parse_text(reconstitute_sentence(self.t[0][i+3].leaves() + ['.']), timeout=5000)
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
	# Wrapper so we only do this part once (removed rollover concept)
	# Assumes this is the very first ner doc
	def do_all_ner(self, awaiting_ner, node_list, keep_bools, stage_num=8):
		finished_ner = []

		sentree_list = []
		all_doc_list = []
		document_fulltext = ""
		count = 0
		curr = self
		while curr is not None:
			document_fulltext += curr.fulltext + " "
			sentree_list.append(curr)
			all_doc_list.append([])
			curr = curr.nextST
			count += 1
		if count != len(awaiting_ner):
			print("Detected doctext:")
			print(document_fulltext + "\n")
			print("Detected Node List:")
			curr = self
			ind = 0
			while curr is not None:
				if ind < len(awaiting_ner):
					if curr == node_list[awaiting_ner[ind]]:
						print("[" + str(ind) + "] OK")
					else:
						print("[" + str(ind) + "] <== WRONG SENTENCE FOUND.")
						print("\t\t Expecting | " + str(curr.fulltext))
						print("\t\t Found | " + str(node_list[awaiting_ner[ind]].fulltext))
					ind += 1
				else:
					print("[" + str(ind) + "] <== MISSING FROM awaiting_ner. Expecting:\n\t | " + curr.fulltext)
				curr = curr.nextST
			raise ValueError

		pattern = re.compile(r'([^a-zA-Z0-9 ])\.(\s*)')
		self.ner = nlp(pattern.sub(r"\1 .\2", document_fulltext))
		tokenized_doc = [token.text for token in self.ner]

		# Get all sentence threhsolds at once
		thresholds_list = [0]
		curr_size = 0
		for sent in self.ner.sents:
			curr_size += len(sent)
			thresholds_list.append(curr_size)
		
		# error check here
		if len(thresholds_list) != len(sentree_list) + 1:
			print(thresholds_list)
			print(len(sentree_list))
			i = 0

			last_thresh = 0
			while i < len(sentree_list) and i < len(thresholds_list) - 1:
				new_thresh = thresholds_list[i+1]
				print(sentree_list[i].text)
				print(self.ner[last_thresh:new_thresh])
				print("")
				i += 1
			raise ValueError

		# Update ner on all ST before running coref
		for ST in sentree_list:
			ST.update_ner(self.ner)

		if self.ner._.has_coref:
			replace_operations = []
			generic_descriptors = ["he", "his", "her", "hers", "she", "him", "they", "their", "them", "theirs", "it", "its"]
			original_pos = [token.pos_ for token in self.ner]
			for cluster in self.ner._.coref_clusters:
				document_metadata["coref"].append(cluster)

				main_ner = [token.ent_type_ for token in cluster.main]
				main_id = get_sent_num(thresholds_list, cluster.main.start)
				main_ST = sentree_list[main_id]
				main_pos = main_ST.corenlp_pos(cluster.main, thresholds_list[main_id + 1] - 1, main_ST)

				if main_pos is not None and "CC" in main_pos:
					# Choose a new main, this one is a list
					generic_list = [item for sublist in generic_NP for item in sublist]
					success = False
					for mention in cluster:
						if mention.text.lower() != cluster.main.text.lower() and mention.text not in generic_list:
							test_id = get_sent_num(thresholds_list, mention.start)
							test_ST = sentree_list[main_id]
							test_pos = main_ST.corenlp_pos(mention, thresholds_list[test_id + 1] - 1, test_ST)

							if test_pos is not None and "CC" not in test_pos:
								cluster.main = mention
								main_ner = [token.ent_type_ for token in mention]
								main_id = test_id
								main_ST = test_ST
								main_pos = test_pos
								success = True
							break
					if not success:
						main_pos = None

				if cluster.main.text.lower() not in generic_descriptors:
					main_text = cluster.main.text
					if not main_ST.check_proper(cluster.main):
						main_text = main_text[0].lower() + main_text[1:]

					for mention in cluster.mentions:
						tar_id = get_sent_num(thresholds_list, mention.start)
						target_ST = sentree_list[tar_id]
						# print("TENTATIVE: [" + target_ST.fulltext + "]")
						# print("CONFiRMED: [" + mention.sent.text + "]")
						if mention.text.lower() != cluster.main.text.lower():
							mention_pos = target_ST.corenlp_pos(mention, thresholds_list[tar_id + 1] - 1, target_ST)
							mention_text = [token.text for token in mention]
							mention_ner = [token.ent_type_ for token in mention]

							if not target_ST.match_ner(main_ner, mention_ner, main_pos, mention_pos, cluster.main, mention):
								if not target_ST.check_proper(mention):
									main_text = target_ST.custom_ner(mention_ner, mention_text, mention.start)
								else:
									main_pos = None

							# TODO: add safety check for main_pos is not None in final version. Leave out now for debug
							if main_pos is not None:
								replace_operations = all_doc_list[tar_id]
								threshold = thresholds_list[tar_id]

								corrected_main = main_text
								if corrected_main[-2:] == "\'s":
									main_text = corrected_main[:-2]
								elif corrected_main[-2:] == "s\'":
									main_text = corrected_main[:-1]
								elif corrected_main[-1] == "s":
									corrected_main += "\'"
								else:
									corrected_main += "\'s"

								# print("metadata:")
								# print("Main position: (" + str(cluster.main.start) + ", " + str(cluster.main.end) + ")")
								# print(mention.start - threshold)
								# print(mention.end - threshold)

								# print(threshold)
								# print(original_pos)

								# print(cluster.main.text)
								# print(main_ner)
								# print([ent.label_ for ent in list(cluster.main.ents)])
								# print([ent.text for ent in list(cluster.main.ents)])
								# print(main_pos)
								# print(mention.text)
								# print(mention_ner)
								# print(mention_pos)
								# print([token.text for token in doc_info][threshold:])

								if 'PRP$' in mention_pos or mention_pos[-1] == 'POS':
									if 'PRP$' in main_pos:
										# print("Chosen replacement: " + main_text)
										replace_operations.append((mention.start - threshold, mention.end - threshold, main_text))
									else:
										# print("Chosen replacement: " + corrected_main)
										replace_operations.append((mention.start - threshold, mention.end - threshold, corrected_main))
								elif len(original_pos) > mention.end and original_pos[mention.end][1] == "POS":
									if 'PRP$' in main_pos:
										# print("Chosen replacement: " + main_text)
										replace_operations.append((mention.start - threshold, mention.end - threshold + 1, main_text))
									else:
										# print("Chosen replacement: " + corrected_main)
										replace_operations.append((mention.start - threshold, mention.end - threshold + 1, corrected_main))
								else:
									# print("Chosen replacement: " + main_text)
									replace_operations.append((mention.start - threshold, mention.end - threshold, main_text))
								# print(main_pos)
								# print("------------------------\n")
			for ST_id in range(len(all_doc_list)):
				replace_operations = all_doc_list[ST_id]
				threshold = thresholds_list[ST_id]
				next_threshold = None
				if ST_id < len(thresholds_list) - 1:
					next_threshold = thresholds_list[ST_id + 1]
				target_ST = sentree_list[ST_id]

				if len(replace_operations) > 0:
					test = tokenized_doc
					if next_threshold is not None:
						test = tokenized_doc[threshold:next_threshold]
					else:
						test = tokenized_doc[threshold:]
					# print("detected target sentence")
					# print(test)

					acc = 0
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
					# print("post_reconstitution: " + test)

					newtree = self.parser.parse_text(test)
					newtree = next(newtree)
					child = SenTree(newtree, self.parser, prevST=target_ST.prevST, nextST=target_ST.nextST, ner=target_ST.ner)
					if target_ST.prevST is not None:
						target_ST.prevST.nextST = child
					if target_ST.nextST is not None:
						target_ST.nextST.prevST = child
					target_ST.children[stage_num] = [child]

					replaced_ST_id = awaiting_ner[ST_id]
					keep_bools[replaced_ST_id] = False

					finished_ner.append(len(node_list))

					node_list.append(child)
					keep_bools.append(True)
				else:
					finished_ner.append(awaiting_ner[ST_id])
		return finished_ner

	#9 Rearrange <SBAR/PP>, <S> into <S> <SBAR/PP>
	def sbarpp_s_rearrange(self, stage_num=9):
		if len(self.t) == 1 and self.t[0].label() == "S":
			S = self.t[0]
			if len(S) > 4 and S[0].label() in ["PP", "SBAR"] and S[1].label() == "," and S.height() > 2:
				found_NP = False
				valid_next = False
				for i in range(2, len(S)):
					if found_NP or has_valid_np(S[i]):
						found_NP = True
					if found_NP and has_valid_vp(S[i]):
						valid_next = True
						break
				if valid_next:
					rest_S = []
					for i in range(2, len(S) - 1):
						rest_S += S[i].leaves()
					fix_first = S[0].leaves()
					fix_first[0] = fix_first[0].lower()
					newtree1 = self.parser.parse_text(reconstitute_sentence(rest_S + fix_first + ["."]), timeout=5000)
					newtree1 = next(newtree1)
					child = SenTree(newtree1, self.parser, prevST=self.prevST, nextST=self.nextST, ner=self.ner)
					child.type = stage_num
					child.flags = self.flags
					self.children[stage_num] = [child]
					if self.prevST is not None:
						self.prevST.nextST = child
					if self.nextST is not None:
						self.nextST.prevST = child
					return True
		return False

	#10 Rearrange <NP> <VP>'s <NP> components based on complexity if verb is "to be" - top-level only atm
	def to_be_equiv(self, stage_num=10):
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
					newtree = self.parser.parse_text(new_sent)
					newtree = next(newtree)
					child = SenTree(newtree, self.parser, prevST=self.prevST, nextST=self.nextST, ner=self.ner)
					child.type = stage_num
					child.flags = self.flags
					self.children[stage_num].append(child)
		return False

	#11 Include every valid <NP> <VP> combo in the sentence
	def npvp_combo(self, stage_num=11):
		subs = self.t.subtrees()
		self.flags["npvp_extracted"] = True
		self.children[stage_num] = []
		for sub in subs:
			if valid_s(sub):
				newtree = self.parser.parse_text(reconstitute_sentence(sub.leaves()))
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
			# Remove removable prefixes
			return self.remove_prefix()
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
	def corenlp_pos(self, mention, mention_per, target_ST):
		# corenlp_tokens = next(self.parser.parse_text(reconstitute_sentence([token.text for token in mention]).replace(" - ", "-")))
		target_sent = self
		hyp_loc = max(0, len(target_sent.text) - 1 - max(0, (mention_per - mention.start)))

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
		# print("Looking forward in range: " + str(hyp_loc) + ", " + str(len(target_sent.text)))
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
			# print("Could not find :")
			# print(fixed_spacy)
			# print("in:")
			# print(target_sent.text)
			return None

	# Check if spacy neuralcoref worked properly
	# Sometimes, it's not the first NP that needs to match, but only a part of it.
	# Ex: Akhenaten's wife corresponds with his, whereas the part that should correspond is just Akhenaten
	def match_ner(self, main_ner, mention_ner, main_pos, mention_pos, main, mention):
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
		if main_pos is None or mention_pos is None:
			return False

		if self.check_proper(mention):
			return False

		if len(mention.text) == 1:
			if mention.text[0] in generic_NP[0]:
				# Should not be a person
				return True
			elif mention.text[0] in generic_NP[1]:
				# Check if origin is: NORP, ORG, PERSON (Plural)
				# print(pos)
				pass
			elif mention.text[0] in generic_NP[2]:
				# Check if origin is PERSON (Singular, Male)
				# print(pos)
				pass
			elif mention.text[0] in generic_NP[3]:
				# Check if origin is PERSON (Singular, Female)
				# print(pos)
				pass
			elif mention.text[0] in generic_NP[4]:
				return True
		return True

	# check if this mention is a proper noun
	def check_proper(self, mention, avoid_recurse=False):
		generic_list = [item for sublist in generic_NP for item in sublist]
		if mention.start > 0 and self.ner[mention.start - 1].text == ".":
			# This is start of sentence, unknown, return True
			if len(mention) > 1:	# If multi-length, and some of the second ones capital, this is proper
				for i in range(1, len(mention)):
					if mention[i].text[0].isupper():
						return True
				return False
			else:					# Singular word
				if self.ner[mention.start + 1].text[0] == '\'':	# We know this is possessive, so assume is proper noun
					return True
			
			# Check for other mentions of this. If any of them are proper, this is proper
			if not avoid_recurse:
				cluster = mention._.coref_cluster
				for c_mention in cluster:
					if self.check_proper(c_mention, avoid_recurse=True):
						return True

			# By default, assume is True unless otherwise
			return True
		elif len(mention) > 1:
			# Not start of sentence, multiple words ==> return True if there exists a capital letter
			for token in mention:
				if token.text[0].isupper():
					return True
			return False
		else:
			# Not start of sentence, singular word ==> return True if not a generic thing
			# Should not have singular things like "The x", or "A x"
			return mention[0].text.lower() not in generic_list

	# Custom NER for when the provided coref fails
	def custom_ner(self, mention_ner, mention_text, mention_start):
		return reconstitute_sentence(mention_text)

	# New function of NER: just set self.ner properly
	def update_ner(self, ner):
		self.ner = ner
		return

	# Update the NER data of this SenTree
	def old_update_ner(self, lookback=5):
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
		pattern = re.compile(r'([^a-zA-Z0-9 ])\.(\s*)')
		self.ner = nlp(pattern.sub(r"\1 .\2", document_fulltext))
		return count,document_fulltext

	# Assumes NER data is up to date for this SenTree
	# Gets you a 
	def align_ner(self):
		#tokens = st.tag(self.text)
		# print("Stanford NER: ")
		# print(tokens)
		# print("SpaCY NER: ")
		spacy_ents = (list(self.ner.sents)[-1].ents)
		ents_txt = "["
		for ent in spacy_ents:
			ents_txt += "(" + ent.text + ", " + ent.label_ + "), "
		if len(ents_txt) > 2:
			ents_txt = ents_txt[:-2] + "]"
		else:
			ents_txt = "[]"
		# print(ents_txt)
		# print("-----------------------\n")
		return

	# Old version still uses SpaCY
	def old_align_ner(self):
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
			# print("COMPARE SPACY to CoreNLP")
			# print(test_spacy)
			# print(self.text)
			# print("=======================================\n")
			pass
		return

# Given a list of sentence ending thresholds and mention, find which sentence it belongs to
def get_sent_num(thresholds_list, start):
	# print(thresholds_list)
	if start >= thresholds_list[-1]:
		return len(thresholds_list) - 1

	loind = 0
	hiind = len(thresholds_list) - 2

	while loind < hiind:
		loval = thresholds_list[loind]
		hival = thresholds_list[hiind + 1]
		approx_size = (hival - loval) // (hiind + 1 - loind)
		hyp = min(hiind, ((start - loval) // approx_size) + loind)
		# print(str(loval) + ", " + str(hival) + ", " + str(approx_size) + ", " + str(thresholds_list[hyp]) + ", " + str(start))

		if start >= thresholds_list[hyp]:
			if start < thresholds_list[hyp + 1]:
				hiind = hyp
				loind = hyp
			else:
				loind = hyp + 1
		else:
			if start >= thresholds_list[hyp - 1]:
				loind = hyp - 1
				hiind = hyp - 1
			else:
				hiind = hyp - 2
	# print("Found <" + str(start) + "> in bucket [" + str(thresholds_list[loind]) + ", " + str(thresholds_list[loind + 1]) + ")")

	return loind

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
	pattern1 = re.compile(r' ([\.,:;])')
	pattern2 = re.compile(r' \'s ')
	return pattern1.sub(r'\1', pattern2.sub('\'s ', reconstruct_quotes)).replace(' \' ', '\' ')

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
	if t.leaves()[-1] != ".":
		return False
	elif t.label() == "ROOT":
		return valid_s(t[0])
	elif t.label() != "S":
		return False
	else:
		return has_valid_np(t) and has_valid_vp(t)
	return False



def getSBARQuestion(SBAR, root):
    retlist = []
    if SBAR.height() > 2:
        lemmatizer = nltk.stem.WordNetLemmatizer()
        if SBAR[0].label() == "WHNP":
            # case who/what
            if SBAR[0][0].label() == "WDT":
                retlist.append(reconstitute_sentence(["What"] + SBAR.leaves()[1:] + ["?"]))
            elif SBAR[0][0].label() == "WP":
                retlist.append(reconstitute_sentence(["Who"] + SBAR.leaves()[1:] + ["?"]))
        elif SBAR[0].label() == "WHADVP":
            # case where/when
            if SBAR[0][0].label() == "WDT":
                retlist.append(reconstitute_sentence(["What"] + SBAR.leaves()[1:] + ["?"]))
            elif SBAR[0][0].label() == "WP":
                retlist.append(reconstitute_sentence(["Who"] + SBAR.leaves()[1:] + ["?"]))
            elif SBAR[0][0].label() == "WRB":
                S = SBAR[1]
                if S.label()[-1] == "S":
                    if S[0].label()[-2:] == "NP" and S[1].label()[-2:] == "VP":
                        if S[1][0].label()[:2] == "VB" and S[1][1].label()[:2] != "VB":
                            vbn = S[1][0].leaves()[0]
                            conj_verb = lemma(vbn)
                            verblvs = []
                            for i in range(1, len(S[1])):
                                verblvs += S[1][i].leaves()
                            retlist.append(reconstitute_sentence(SBAR[0][0].leaves() + ["did"] + S[0].leaves() + [conj_verb] + verblvs + ["?"]))
        elif SBAR[0].label() == "IN":
            # Case on that = what + invert, although = else
            if len(SBAR) > 1 and SBAR[1].label()[-1] == "S":
                if len(SBAR[1]) == 2:
                    retlist.append(reconstitute_sentence(["What"] + SBAR[1][1].leaves() + ["?"]))
                    # if has_valid_np(SBAR[1][0]) and has_valid_vp(SBAR[1][1]):
                    #     retlist.append(reconstitute_sentence(["What", "was"] + SBAR[1].leaves() + ["?"]))
                else:
                    retlist.append(reconstitute_sentence(["What"] + SBAR[1][0].leaves() + ["?"]))
        elif SBAR[0].label() == "S":
            if len(SBAR[0]) == 2:
                retlist.append(reconstitute_sentence(["What"] + SBAR[0][1].leaves() + ["?"]))
    return retlist


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

	updated_root = root_list[0]
	node_list = [root for root in root_list]
	keep_bools = [True for root in root_list]
	for i in range(len(node_list)):
		FrontierQueue.put_nowait((i, 1))

	not_run_ner = True
	ner_stage = 8
	not_finished = True
	awaiting_ner = []
	while not_finished:
		while not FrontierQueue.empty():
			node_id,stage = FrontierQueue.get_nowait()
			curr_node = node_list[node_id]
			# print("Stage " + str(stage) + ":")
			# print(curr_node.text)
			# print(curr_node.fulltext)
			# print("-----------------------\n")
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
						if child.prevST is None:
							updated_root = child
						if new_stage != ner_stage:
							FrontierQueue.put_nowait((len(node_list), new_stage))
						else:
							awaiting_ner.append(len(node_list))
						node_list.append(child)
						keep_bools.append(True)
			else:
				rollover, new_stage = acc_stage(stage)
				if not rollover:
					if new_stage != ner_stage:
						FrontierQueue.put_nowait((node_id, new_stage))
					else:
						awaiting_ner.append(node_id)
		if not_run_ner and len(awaiting_ner) > 0:
			not_run_ner = False

			finished_ner = updated_root.do_all_ner(awaiting_ner, node_list, keep_bools)

			for ind in finished_ner:
				FrontierQueue.put_nowait((ind, ner_stage + 1))

			updated_root = node_list[finished_ner[0]]
		else:
			not_finished = False

	preprocessed_trees = [node_list[i] for i in range(len(node_list)) if keep_bools[i]]
	# for tree in preprocessed_trees:
	# 	if tree.text[-1] == ".":
	# 		tree.update_ner()
	# 		tree.align_ner()
	for cluster in document_metadata["coref"]:
		# print(cluster)
		pass
	# print("==============================\n")
	return preprocessed_trees, preprocessed_questions