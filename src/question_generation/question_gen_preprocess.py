# -*- coding: utf-8 -*-
from collections import defaultdict
import re
from queue import Queue, LifoQueue
from pathlib import Path
import neuralcoref
import spacy

import nltk
from nltk.tag import StanfordNERTagger

from pattern.en import lemma
import pdb
from src.utils import get_project_root

import sys


safety = True
debug_print = False

english_classifiers_path = get_project_root() / Path("corenlp_ner/english.all.3class.distsim.crf.ser.gz")
ner_jar_path = get_project_root() / Path("corenlp_ner/stanford-ner.jar")
st = StanfordNERTagger(str(english_classifiers_path), path_to_jar=str(ner_jar_path), encoding='utf-8')
#st = StanfordNERTagger('/Users/Thomas/Documents/11-411/NER/classifiers/english.all.3class.distsim.crf.ser.gz', '/Users/Thomas/Documents/11-411/NER/stanford-ner.jar', encoding='utf-8')

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
		t2 = t
		while(t2.height() > 2):
			t2 = t2[0]
		if t2.label() not in ["NNP", "NNPS"]:
			t2[0] = t2[0].lower()
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
		self.fulltext = reconstitute_sentence(self.t.leaves()) if self.t is not None else ""

	#1 Replace <NN_> <PRP> turns of phrase

	#1 Replace <has been> <___> <to be> turns of phrase
	def tobe_turn_of_phrase(self, text=None, pos=None, turns=[], stage_num=1):
		try:
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
				newtree = None
				try:
					newtree = self.parser.parse_text(reconstitute_sentence(text), timeout=5)
					newtree = next(newtree)
				except:
					if debug_print:
						print("Failed new parse [" + str(stage_num) + "] on sentence [" + reconstitute_sentence(text) + "]\n", file=sys.stderr)
					raise
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
		except:
			if debug_print:
				print("Unknown error in stage [" + str(stage_num) + "], assuming did not compromise node integrity, continuing operation\n", file=sys.stderr)
			if not safety:
				raise
		return False

	#2 Remove removable prefixes
	def remove_prefix(self, stage_num=2):
		try:
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
						if debug_print:
							print("REMOVED PREFIX TEXT [" + S[0].label() + "]: " + reconstitute_sentence(S[0].leaves()), file=sys.stderr)

						result = []
						for i in range(2, len(S)):
							result += S[i].leaves()

						temp = reconstitute_sentence(result)
						newtree = None
						try:
							newtree = self.parser.parse_text(temp, timeout=5)
							newtree = next(newtree)
						except:
							if debug_print:
								print("Failed new parse [" + str(stage_num) + "] on sentence [" + temp + "]\n", file=sys.stderr)
							raise

						child = SenTree(newtree, self.parser, prevST=self.prevST, nextST=self.nextST)
						# print(newtree.leaves(), file=sys.stderr)
						child.type = stage_num
						child.flags = self.flags
						self.children[stage_num] = [child]
						if self.prevST is not None:
							self.prevST.nextST = child
						if self.nextST is not None:
							self.nextST.prevST = child
						return True
		except:
			if debug_print:
				print("Unknown error in stage [" + str(stage_num) + "], assuming did not compromise node integrity, continuing operation\n", file=sys.stderr)
			if not safety:
				raise
		return False

	#3-4 Parenthetical removal
	def parenthetical_removal(self, stage_num=3):
		try:
			result = []
			count = 0
			start = None
			parentheticals = []
			# print(self.text, file=sys.stderr)
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

					newtree = None
					try:
						newtree = self.parser.parse_text(temp, timeout=5)
						newtree = next(newtree)
					except:
						if debug_print:
							print("Failed new parse [" + str(stage_num) + "] on sentence [" + temp + "]\n", file=sys.stderr)
						raise

					child = SenTree(newtree, self.parser, prevST=self.prevST, nextST=self.nextST, ner=self.ner)
					# print(newtree.leaves(), file=sys.stderr)
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
		except:
			if debug_print:
				print("Unknown error in stage [" + str(stage_num) + "], assuming did not compromise node integrity, continuing operation\n", file=sys.stderr)
			if not safety:
				raise
		return False

	def parenthetical_child_removal(self, stage_num=4):
		try:
			result = []
			parentheticals = []
			t = self.t.copy(deep=True)
			self.recursive_paren_remover(t,result,parentheticals)
			if parentheticals and result:
				temp = reconstitute_sentence(result)

				newtree = None
				try:
					newtree = self.parser.parse_text(temp, timeout=5)
					newtree = next(newtree)
				except:
					if debug_print:
						print("Failed new parse [" + str(stage_num) + "] on sentence [" + temp + "]\n", file=sys.stderr)
					raise

				child = SenTree(newtree, self.parser, prevST=self.prevST, nextST=self.nextST, ner=self.ner)
				# print(newtree.leaves(), file=sys.stderr)
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
		except:
			if debug_print:
				print("Unknown error in stage [" + str(stage_num) + "], assuming did not compromise node integrity, continuing operation\n", file=sys.stderr)
			if not safety:
				raise
		return False	

	def recursive_paren_remover(self, t, result, parentheticals):
		if t.height() <= 2:
			result += t.leaves()
			return
		i = 0
		while(i < len(t)):
			if t[i].label() == "PRN":
				paren_text = [x if x != "-LRB-" else "(" for x in t[i].leaves()]
				paren_text = [x if x != "-RRB-" else ")" for x in t[i].leaves()]
				paren_text = " ".join(paren_text)
				parentheticals.append({"index" : len(result), "text": " ".join(paren_text)})
				t.__delitem__(i)
			else:
				self.recursive_paren_remover(t[i], result, parentheticals)
				i += 1
		if parentheticals:
			return


	def is_np_list(self, s, i=0):
		if s.label() == "NP" and s.height() > 2:
			a = i
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
						if debug_print:
							print("SKIPPING, since FOUND LIST in :", file=sys.stderr)
							s.pretty_print()
						return True
					else:
						break
		return False

	#5 Run apositive removal/manipulation
	def appositive_removal(self, immediate_questions, stage_num=5):
		# can generate "IS <A> an apt descriptor for <B>?"
		delims = [";", ":", ",", "."]
		allowables = ["NP", "PP", "SBAR", "S", "NN", "VP"]
		retval = False
		tree_string = " ".join(self.t.leaves())
		# self.t.pretty_print()
		for s in list(self.t.subtrees()):
			try:
				is_list = self.is_np_list(s)
				if not is_list:
					if len(s) >= 3 and s.height() > 2 and s.label() in ["NP", "VP"]:
						#print("GOING", file=sys.stderr)
						#s.pretty_print()
						i = 0
						appositives_and_delims = []
						#pdb.set_trace()
						while not is_list and (i < len(s)-3):
							if s[i].label() in ["NP", "NN"] and s[i+1].label() in delims and s[i+2].label() in allowables and s[i+3].label() in delims:
								#print("GOT ONE", file=sys.stderr)
								#s[i+2].pretty_print()
								#print(self.is_np_list(s, i=i+2), file=sys.stderr)
								if s[i+2].pos()[0][1] != "CC" and not self.is_np_list(s, i=i+2):
									u_s = use_second(s[i],s[i+1],s[i+2])
									if len(s[i+2].leaves()) > 1 and not u_s:
										appositives_and_delims.append(i+1)
										appositives_and_delims.append(i+2)
										if debug_print:
											print(" ".join(s[i].leaves()) + " is NP to the appositive " + " ".join(s[i+2].leaves()), file=sys.stderr)
										immediate_questions.append("AP: Is \""+" ".join(s[i+2].leaves()) + "\" an apt descriptor for " + " ".join(s[i].leaves())+"?")
										if s[i+2].label() == "SBAR":
											immediate_questions += getSBARQuestion(s[i+2], self.t)
									elif (len(s[i].leaves()) > 1 and s[i].label() == "NP" and s[i+2].label == "NP" and s[i+2].height() > 2 and s[i+2][0].label() in ["NNP","NNPS"]) or u_s:
										appositives_and_delims.append(i)
										appositives_and_delims.append(i+1)
										if debug_print:
											print(" ".join(s[i+2].leaves()) + " is NP to the appositive " + " ".join(s[i].leaves()), file=sys.stderr)
										immediate_questions.append("AP: Is \""+" ".join(s[i].leaves()) + "\" an apt descriptor for " + " ".join(s[i+2].leaves())+"?")
									# retval = True
								elif debug_print:
									print("SKIPPING child " + str(i+2) +", since FOUND LIST inside", file=sys.stderr)
									s.pretty_print()
							elif i > 0 and s[i-1].label() in ["NP", "NN"] and s[i].label() == "PP" and s[i+1].label() in delims and s[i+2].label() in allowables and s[i+3].label() in delims:
								if s[i+2].pos()[0][1] != "CC" and not self.is_np_list(s, i=i+2):
									u_s = use_second(s[i-1],s[i+1],s[i+2])
									if len(s[i+2].leaves()) > 1 and not u_s:
										appositives_and_delims.append(i+1)
										appositives_and_delims.append(i+2)
										if debug_print:
											print(" ".join(s[i-1].leaves()) + " is NP to the appositive " + " ".join(s[i+2].leaves()), file=sys.stderr)
										immediate_questions.append("AP: Is \""+" ".join(s[i+2].leaves()) + "\" an apt descriptor for " + " ".join(s[i-1].leaves())+"?")
										if s[i+2].label() == "SBAR":
											immediate_questions += getSBARQuestion(s[i+2], self.t)
									elif (len(s[i-1].leaves()) > 1 and s[i-1].label() == "NP" and s[i+2].label == "NP" and s[i+2].height() > 2 and s[i+2][0].label() in ["NNP","NNPS"]) or u_s:
										appositives_and_delims.append(i-1)
										appositives_and_delims.append(i)
										appositives_and_delims.append(i+1)
										if debug_print:
											print(" ".join(s[i+2].leaves()) + " is NP to the appositive " + " ".join(s[i-1].leaves())+" "+" ".join(s[i].leaves()), file=sys.stderr)
										immediate_questions.append("AP: Is \""+" ".join(s[i-1].leaves()) + "\" an apt descriptor for " + " ".join(s[i+2].leaves())+"?")
									# retval = True
								elif debug_print:
									print("SKIPPING child " + str(i+2) +", since FOUND LIST inside", file=sys.stderr)
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
								u_s = use_second(s[-3],s[i-2],s[-1])
								if len(s[-1].leaves()) > 1 and not u_s:
									appositives_and_delims.append(len(s)-1)
									appositives_and_delims.append(len(s)-2)
									if debug_print:
										print(" ".join(s[-3].leaves()) + " is NP to the appositive " + " ".join(s[-1].leaves()), file=sys.stderr)
									immediate_questions.append("AP: Is \""+" ".join(s[-1].leaves()) + "\" an apt descriptor for " + " ".join(s[-3].leaves())+"?")
									if s[-1].label() == "SBAR":
										immediate_questions += getSBARQuestion(s[-1], self.t)
								elif (len(s[-3].leaves()) > 1 and s[-3].label() == "NP" and s[-1].label == "NP" and s[-1].height() > 2 and s[-1][0].label() in ["NNP","NNPS"]) or u_s:
									appositives_and_delims.append(len(s)-3)
									appositives_and_delims.append(len(s)-2)
									if debug_print:
										print(" ".join(s[-1].leaves()) + " is NP to the appositive " + " ".join(s[-3].leaves()), file=sys.stderr)
									immediate_questions.append("AP: Is \""+" ".join(s[-3].leaves()) + "\" an apt descriptor for " + " ".join(s[i-1].leaves())+"?")

						elif not is_list and len(s) > 3 and s[-4].label() in ["NP", "NN"] and s[-3].label() == "PP" and s[-2].label() in delims and s[-1].label() in allowables:
							s_idx = -1
							try:
								s_idx = tree_string.index(" ".join(s.leaves()))
							except:
								pass
							if s_idx >= 0:
								next_idx = s_idx + len(" ".join(s.leaves()))+1
							if s[-1].pos()[0][1] != "CC" and next_idx < len(tree_string) and tree_string[next_idx] in delims:
								u_s = use_second(s[-4],s[-2],s[-1])
								if len(s[-1].leaves()) > 1 and not u_s:
									appositives_and_delims.append(len(s)-1)
									appositives_and_delims.append(len(s)-2)
									if debug_print:
										print(" ".join(s[-4].leaves()) + " is NP to the appositive " + " ".join(s[-1].leaves()), file=sys.stderr)
									immediate_questions.append("AP: Is \""+" ".join(s[-1].leaves()) + "\" an apt descriptor for " + " ".join(s[-4].leaves())+"?")
									if s[-1].label() == "SBAR":
										immediate_questions += getSBARQuestion(s[-1], self.t)						
									# retval = True
								elif (len(s[-4].leaves()) > 1 and s[-4].label() == "NP" and s[-1].label == "NP" and s[-1].height() > 2 and s[-1][0].label() in ["NNP","NNPS"]) or u_s:
									appositives_and_delims.append(len(s)-4)
									appositives_and_delims.append(len(s)-3)
									appositives_and_delims.append(len(s)-2)
									if debug_print:
										print(" ".join(s[-1].leaves()) + " is NP to the appositive " + " ".join(s[-4].leaves())+" "+" ".join(s[i-3].leaves()), file=sys.stderr)
									immediate_questions.append("AP: Is \""+" ".join(s[-4].leaves()) + "\" an apt descriptor for " + " ".join(s[i-1].leaves())+"?")

						appositives_and_delims.sort(reverse=True)
						for idx in appositives_and_delims:
							s.__delitem__(idx)
			except:
				if debug_print:
					print("...what just happed to appositive...")
		try:
			for s in self.t.subtrees():
				if s.height() > 1 and s[0] != str(s[0]):
					for i in range(len(s)-1,-1,-1):
						if not s[i].leaves():
							s.__delitem__(i)
		except:
			if debug_print:
				print("Appositive_removal node trimming failed!")

		self.update_text()
		#pdb.set_trace()
		return retval

	#6 Remove NP-prefixed SBAR
	# Should only remove trailing SBAR, and only when no CC, because that tends to mean there's a list, so the parser messed up
	def sbar_remove(self, immediate_questions, stage_num=6):
		try:
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

							newtree = None
							try:
								newtree = self.parser.parse_text(temp, timeout=5)
								newtree = next(newtree)
							except:
								if debug_print:
									print("Failed new parse [" + str(stage_num) + "] on sentence [" + temp + "]\n", file=sys.stderr)
								raise

							child = SenTree(newtree, self.parser, prevST=self.prevST, nextST=self.nextST)
							# print(newtree.leaves(), file=sys.stderr)
							child.type = stage_num
							child.flags = self.flags
							self.children[stage_num] = [child]
							if self.prevST is not None:
								self.prevST.nextST = child
							if self.nextST is not None:
								self.nextST.prevST = child
							
							if debug_print:
								print("REMOVED TRAILING SBAR. CHANGE [" + self.fulltext + "]", file=sys.stderr)
								print("=====> [" + child.fulltext + "]\n", file=sys.stderr)

							return True
					curr = curr[-1]
		except:
			if debug_print:
				print("Unknown error in stage [" + str(stage_num) + "], assuming did not compromise node integrity, continuing operation\n", file=sys.stderr)
			if not safety:
				raise
		return False
	
	#7 Separate root-level <S> <CC> <S>
	def s_cc_s_separation(self, stage_num=7):
		try:
			for i in range(len(self.t[0])-2):
				newtree1 = None
				newtree2 = None
				if (valid_s(self.t[0][i]) and self.t[0][i+1].label() == "CC" and valid_s(self.t[0][i+2])):
					try:
						newtree1 = self.parser.parse_text(reconstitute_sentence(self.t[0][i].leaves() + ['.']), timeout=5)
						newtree1 = next(newtree1)
					except:
						if debug_print:
							print("Failed new parse [" + str(stage_num) + "] on sentence [" + reconstitute_sentence(self.t[0][i].leaves() + ['.']) + "]\n", file=sys.stderr)
						raise

					try:
						newtree2 = self.parser.parse_text(reconstitute_sentence(self.t[0][i+2].leaves() + ['.']), timeout=5)
						newtree2 = next(newtree2)
					except:
						if debug_print:
							print("Failed new parse [" + str(stage_num) + "] on sentence [" + reconstitute_sentence(self.t[0][i+2].leaves() + ['.']) + "]\n", file=sys.stderr)
						raise

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
					try:
						newtree1 = self.parser.parse_text(reconstitute_sentence(self.t[0][i].leaves() + ['.']), timeout=5)
						newtree1 = next(newtree1)
					except:
						if debug_print:
							print("Failed new parse [" + str(stage_num) + "] on sentence [" + reconstitute_sentence(self.t[0][i].leaves() + ['.']) + "]\n", file=sys.stderr)
						raise
					try:
						newtree2 = self.parser.parse_text(reconstitute_sentence(self.t[0][i+3].leaves() + ['.']), timeout=5)
						newtree2 = next(newtree2)
					except:
						if debug_print:
							print("Failed new parse [" + str(stage_num) + "] on sentence [" + reconstitute_sentence(self.t[0][i+3].leaves() + ['.']) + "]\n", file=sys.stderr)
						raise
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
		except:
			if debug_print:
				print("Unknown error in stage [" + str(stage_num) + "], assuming did not compromise node integrity, continuing operation\n", file=sys.stderr)
			if not safety:
				raise
		return False

	#8 NER coreference resolution
	# Wrapper so we only do this part once (removed rollover concept)
	# Assumes this is the very first ner doc
	def do_coref(self, awaiting_ner, node_list, stage_num=8):
		sentree_list = self.do_corenlp_supersense()
		count = len(sentree_list)

		finished_coref = []
		all_doc_list = []
		document_fulltext = ""

		try:
			for ST in sentree_list:
				document_fulltext += ST.fulltext + " "
				all_doc_list.append([])

			if count != len(awaiting_ner):
				if debug_print:
					print("Detected doctext:", file=sys.stderr)
					print(document_fulltext + "\n", file=sys.stderr)
					print("Detected Node List:", file=sys.stderr)
					curr = self
					ind = 0
					while curr is not None:
						if ind < len(awaiting_ner):
							if curr == node_list[awaiting_ner[ind]]:
								print("[" + str(ind) + "] OK", file=sys.stderr)
							else:
								print("[" + str(ind) + "] <== WRONG SENTENCE FOUND.", file=sys.stderr)
								print("\t\t Expecting | " + str(curr.fulltext), file=sys.stderr)
								print("\t\t Found | " + str(node_list[awaiting_ner[ind]].fulltext), file=sys.stderr)
							ind += 1
						else:
							print("[" + str(ind) + "] <== MISSING FROM awaiting_ner. Expecting:\n\t | " + curr.fulltext, file=sys.stderr)
						curr = curr.nextST
				raise ValueError
			
			if debug_print:
				print("Detected doctext:", file=sys.stderr)
				for ST in sentree_list:
					print(ST.text, file=sys.stderr)

			pattern = re.compile(r'([^a-zA-Z0-9 ])\.(\s*)')
			spacy_input = pattern.sub(r"\1 .\2", document_fulltext)
			self.ner = nlp(spacy_input)
			tokenized_doc = [token.text for token in self.ner]

			# Get all sentence threhsolds at once
			thresholds = [0]
			curr_size = 0
			for i in range(len(tokenized_doc)):
				token = tokenized_doc[i]
				if token == ".":
					thresholds.append(i + 1)
			
			# error check here
			if len(thresholds) != len(sentree_list) + 1:
				if debug_print:
					print(thresholds, file=sys.stderr)
					print(len(sentree_list), file=sys.stderr)

					i = 0
					while i < len(sentree_list) and i < len(thresholds) - 1:
						new_thresh = thresholds[i+1]
						last_thresh = thresholds[i]
						print(sentree_list[i].text, file=sys.stderr)
						print([token.text for token in self.ner[last_thresh:new_thresh]], file=sys.stderr)
						print("", file=sys.stderr)
						i += 1
					
					print("SPACY INPUT:", file=sys.stderr)
					print(spacy_input, file=sys.stderr)
				raise ValueError

			# Update ner on all ST before running coref
			beginning = 0
			for i in range(1, len(thresholds)):
				end = thresholds[i]

				curr_slice = self.ner[beginning:end]

				ST = sentree_list[i-1]
				ST.update_ner(self.ner)
				ST.bubble_ner(curr_slice, corenlp=False)
				ST.align_ner()

				beginning = end

			if self.ner._.has_coref:
				replace_operations = []
				generic_descriptors = ["he", "his", "her", "hers", "she", "him", "they", "their", "them", "theirs", "it", "its"]
				original_pos = [token.pos_ for token in self.ner]
				for cluster in self.ner._.coref_clusters:
					document_metadata["coref"].append(cluster)

					main_ner = [token.ent_type_ for token in cluster.main]
					main_id = get_sent_num(thresholds, cluster.main.start)
					main_ST = sentree_list[main_id]
					main_pos = main_ST.corenlp_pos(cluster.main, thresholds[main_id + 1] - 1, main_ST)

					if main_pos is not None and "CC" in main_pos:
						# Choose a new main, this one is a list. Give preference to proper nouns
						generic_list = [item for sublist in generic_NP for item in sublist]
						success = False
						for mention in cluster:
							if mention.text.lower() != cluster.main.text.lower() and (mention.text not in generic_list):
								test_id = get_sent_num(thresholds, mention.start)
								test_ST = sentree_list[test_id]
								test_pos = test_ST.corenlp_pos(mention, thresholds[test_id + 1] - 1, test_ST)

								# Check that new one is also not a list
								if test_pos is not None and "CC" not in test_pos:
									cluster.main = mention
									main_ner = [token.ent_type_ for token in mention]
									main_id = test_id
									main_ST = test_ST
									main_pos = test_pos
									success = True

									# If this is a proper noun, then hooray we're done. Else, keep searching for a proper noun
									if main_ST.check_proper(mention):
										break
						if not success:
							main_pos = None

					# only want to do coref if we found a reference to something solid ==> not a generic thing
					if cluster.main.text.lower() not in generic_descriptors:
						main_text = cluster.main.text
						if not main_ST.check_proper(cluster.main):
							main_text = main_text[0].lower() + main_text[1:]

						# Assuming sentences are relatively clean/simplified, don't want to do coref if found prior descriptor is in the same sentence
						for mention in cluster.mentions:
							tar_id = get_sent_num(thresholds, mention.start)
							target_ST = sentree_list[tar_id]
							# print("TENTATIVE: [" + target_ST.fulltext + "]", file=sys.stderr)
							# print("CONFiRMED: [" + mention.sent.text + "]", file=sys.stderr)

							# Don't want to replcae if current mention is a proper noun
							if mention.text.lower() != cluster.main.text.lower() and main_ST != target_ST and not target_ST.check_proper(mention):
								mention_pos = target_ST.corenlp_pos(mention, thresholds[tar_id + 1] - 1, target_ST)
								mention_text = [token.text for token in mention]
								mention_ner = [token.ent_type_ for token in mention]

								if mention_pos is not None and not target_ST.match_ner(main_ner, mention_ner, main_pos, mention_pos, cluster.main, mention):
									main_text,main_pos = target_ST.custom_ner(mention_ner, mention_text, mention.start)

								# TODO: add safety check for main_pos is not None in final version. Leave out now for debug
								if mention_pos is not None and main_pos is not None:
									replace_operations = all_doc_list[tar_id]
									threshold = thresholds[tar_id]

									corrected_main = main_text
									if corrected_main[-2:] == "\'s":
										main_text = corrected_main[:-2]
									elif corrected_main[-2:] == "s\'":
										main_text = corrected_main[:-1]
									elif corrected_main[-1] == "s":
										corrected_main += "\'"
									else:
										corrected_main += "\'s"

									if 'PRP$' in mention_pos or mention_pos[-1] == 'POS':
										if 'PRP$' in main_pos:
											# print("Chosen replacement: " + main_text, file=sys.stderr)
											replace_operations.append((mention.start - threshold, mention.end - threshold, main_text))
										else:
											# print("Chosen replacement: " + corrected_main, file=sys.stderr)
											replace_operations.append((mention.start - threshold, mention.end - threshold, corrected_main))
									elif len(original_pos) > mention.end and original_pos[mention.end][1] == "POS":
										if 'PRP$' in main_pos:
											# print("Chosen replacement: " + main_text, file=sys.stderr)
											replace_operations.append((mention.start - threshold, mention.end - threshold + 1, main_text))
										else:
											# print("Chosen replacement: " + corrected_main, file=sys.stderr)
											replace_operations.append((mention.start - threshold, mention.end - threshold + 1, corrected_main))
									else:
										# print("Chosen replacement: " + main_text, file=sys.stderr)
										replace_operations.append((mention.start - threshold, mention.end - threshold, main_text))
									
									if debug_print:
										print("metadata:", file=sys.stderr)
										print("Main position: (" + str(cluster.main.start) + ", " + str(cluster.main.end) + ")", file=sys.stderr)
										print(mention.start - threshold, file=sys.stderr)
										print(mention.end - threshold, file=sys.stderr)

										print(threshold, file=sys.stderr)
										print(original_pos, file=sys.stderr)

										print(cluster.main.text, file=sys.stderr)
										print(main_ner, file=sys.stderr)
										print([ent.label_ for ent in list(cluster.main.ents)], file=sys.stderr)
										print([ent.text for ent in list(cluster.main.ents)], file=sys.stderr)
										print(main_pos, file=sys.stderr)
										print(mention.text, file=sys.stderr)
										print(mention_ner, file=sys.stderr)
										print(mention_pos, file=sys.stderr)
										print([token.text for token in doc_info][threshold:], file=sys.stderr)

										print("------------------------\n", file=sys.stderr)
				for ST_id in range(len(all_doc_list)):
					replace_operations = all_doc_list[ST_id]
					threshold = thresholds[ST_id]
					next_threshold = None
					if ST_id < len(thresholds) - 1:
						next_threshold = thresholds[ST_id + 1]
					target_ST = sentree_list[ST_id]

					if len(replace_operations) > 0:
						test = tokenized_doc
						if next_threshold is not None:
							test = tokenized_doc[threshold:next_threshold]
						else:
							test = tokenized_doc[threshold:]
						# print("detected target sentence", file=sys.stderr)
						# print(test, file=sys.stderr)

						acc = 0
						for operation in replace_operations:
							start = operation[0] + acc
							end_p = operation[1] + acc
							replace_text = operation[2].split()
							# print(str(start) + ", " + str(end_p) + ", " + operation[2], file=sys.stderr)
							replace_size = len(replace_text)

							test = test[:start] + replace_text + test[end_p:]

							acc += replace_size - (end_p - start)

						test = reconstitute_sentence(test)
						test = test.replace(" - ", "-")
						# print("post_reconstitution: " + test, file=sys.stderr)

						newtree = self.parser.parse_text(test)
						newtree = next(newtree)
						child = SenTree(newtree, self.parser, prevST=target_ST.prevST, nextST=target_ST.nextST, ner=target_ST.ner)
						if target_ST.prevST is not None:
							target_ST.prevST.nextST = child
						if target_ST.nextST is not None:
							target_ST.nextST.prevST = child
						target_ST.children[stage_num] = [child]

						replaced_ST_id = awaiting_ner[ST_id]

						finished_coref.append(len(node_list))

						node_list.append(child)
					else:
						finished_coref.append(awaiting_ner[ST_id])
		except:
			finished_coref = awaiting_ner
			if debug_print:
				print("Failed something, probably threshold creation")
			if not safety:
				raise
		return finished_coref

	#9 Rearrange <SBAR/PP>, <S> into <S> <SBAR/PP>
	def sbarpp_s_rearrange(self, stage_num=9):
		try:
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
						newtree = None
						try:
							newtree = self.parser.parse_text(reconstitute_sentence(rest_S + fix_first + ["."]), timeout=5)
							newtree = next(newtree)
						except:
							if debug_print:
								print("Failed new parse [" + str(stage_num) + "] on sentence [" + reconstitute_sentence(rest_S + fix_first + ["."]) + "]\n", file=sys.stderr)
							raise
						child = SenTree(newtree, self.parser, prevST=self.prevST, nextST=self.nextST, ner=self.ner)
						child.type = stage_num
						child.flags = self.flags
						self.children[stage_num] = [child]
						if self.prevST is not None:
							self.prevST.nextST = child
						if self.nextST is not None:
							self.nextST.prevST = child
						return True
		except:
			if debug_print:
				print("Unknown error in stage [" + str(stage_num) + "], assuming did not compromise node integrity, continuing operation\n", file=sys.stderr)
			if not safety:
				raise
		return False

	#10 Rearrange <NP> <VP>'s <NP> components based on complexity if verb is "to be" - top-level only atm
	def to_be_equiv(self, stage_num=10):
		try:
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
						newtree = None
						try:
							newtree = self.parser.parse_text(new_sent)
							newtree = next(newtree)
						except:
							if debug_print:
								print("Failed new parse [" + str(stage_num) + "] on sentence [" + new_sent + "]\n", file=sys.stderr)
							raise
						child = SenTree(newtree, self.parser, prevST=self.prevST, nextST=self.nextST, ner=self.ner)
						child.type = stage_num
						child.flags = self.flags
						self.children[stage_num].append(child)
						return True
		except:
			if debug_print:
				print("Unknown error in stage [" + str(stage_num) + "], assuming did not compromise node integrity, continuing operation\n", file=sys.stderr)
			if not safety:
				raise
		return False

	#11 Include every valid <NP> <VP> combo in the sentence
	def npvp_combo(self, stage_num=11):
		try:
			subs = self.t.subtrees()
			self.flags["npvp_extracted"] = True
			self.children[stage_num] = []
			changed = False
			for sub in subs:
				if valid_s(sub):
					for i in range(len(sub)-1):
						if sub.height() > 2 and valid_np(sub[i]) == "NP" and valid_vp(sub[i+1]):
							newtree = self.parser.parse_text(reconstitute_sentence(sub.leaves()))
							newtree = next(newtree)
							child = SenTree(newtree, self.parser, prevST=self.prevST, nextST=self.nextST, ner=self.ner)
							child.type = stage_num
							child.flags = self.flags
							self.children[stage_num].append(child)
							changed = True
			return changed
		except:
			if debug_print:
				print("Unknown error in stage [" + str(stage_num) + "], assuming did not compromise node integrity, continuing operation\n", file=sys.stderr)
			if not safety:
				raise
		return False
	
	# Wrapper to handle stage progression
	def handle_stage(self, stage, immediate_questions):
		if stage == 1:
			# Replace <has been> <___> <to be> turns of phrase
			return self.tobe_turn_of_phrase()
		elif stage == 2:
			# Remove removable prefixes
			return self.remove_prefix()
		elif stage == 3:
			# Parenthetical removal 1
			return self.parenthetical_removal()
		elif stage == 4:
			# Parenthetical removal 2
			return self.parenthetical_child_removal()
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
		try:
			# corenlp_tokens = next(self.parser.parse_text(reconstitute_sentence([token.text for token in mention]).replace(" - ", "-")))
			target_sent = self
			hyp_loc = max(0, len(target_sent.text) - 1 - max(0, (mention_per - mention.start)))

			# print("Searching for: [" + reconstitute_sentence([token.text for token in mention]).replace(" - ", "-") + "] in [" + target_sent.fulltext + "]", file=sys.stderr)

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
			# print("Looking forward in range: " + str(hyp_loc) + ", " + str(len(target_sent.text)), file=sys.stderr)
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
			elif debug_print:
				print("Could not find :", file=sys.stderr)
				print(fixed_spacy, file=sys.stderr)
				print("in:", file=sys.stderr)
				print(target_sent.text, file=sys.stderr)
		except:
			if debug_print:
				print("Failed to align SpaCY tokens to CoreNLP tokens", file=sys.stderr)
			if not safety:
				raise
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
		# PERCENT		Percentage, including symbol.
		# MONEY			Monetary values, including unit.
		# QUANTITY		Measurements, as of weight or distance.
		# ORDINAL		“first”, “second”, etc.
		# CARDINAL		Numerals that do not fall under another type.
		# print(main.label, file=sys.stderr)
		# print(main_ner, file=sys.stderr)
		# print(mention_ner, file=sys.stderr)
		try:
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
					# print(pos, file=sys.stderr)
					pass
				elif mention.text[0] in generic_NP[2]:
					# Check if origin is PERSON (Singular, Male)
					# print(pos, file=sys.stderr)
					pass
				elif mention.text[0] in generic_NP[3]:
					# Check if origin is PERSON (Singular, Female)
					# print(pos, file=sys.stderr)
					pass
				elif mention.text[0] in generic_NP[4]:
					return True
		except:
			if debug_print:
				print("Failed to match ner", file=sys.stderr)
			if not safety:
				raise
		return True

	# check if this mention is a proper noun
	def check_proper(self, mention, avoid_recurse=False):
		try:
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
		except:
			if debug_print:
				print("Somehow failed a proper noun check", file=sys.stderr)
			if not safety:
				raise
		return True

	# Custom NER for when the provided coref fails
	# Return <text for new match>, main_pos
	# Main_pos is None in case of failure
	def custom_ner(self, mention_ner, mention_text, mention_start):
		return reconstitute_sentence(mention_text),None

	# New function of NER: just set self.ner properly
	def update_ner(self, ner):
		self.ner = ner
		return

	# Starting from the root node, get and set corenlp supersense data for all nodes
	def do_corenlp_supersense(self):
		curr = self
		text_tokens = []
		ST_list = []
		thresholds = [0]
		while curr is not None:
			text_tokens += curr.text
			thresholds.append(len(text_tokens))
			ST_list.append(curr)
			curr = curr.nextST

		ner_tokens = st.tag(text_tokens)
		beginning = 0
		for i in range(1, len(thresholds)):
			end = thresholds[i]

			curr_slice = ner_tokens[beginning:end]
			ST_list[i-1].bubble_ner(curr_slice, corenlp=True)

			beginning = end
		return ST_list

	# Starting from the root node, get and set spacy supersense data for all nodes
	def do_spacy_supersense(self, sentree_list):
		try:
			all_doc_list = []
			document_fulltext = ""

			for ST in sentree_list:
				document_fulltext += ST.fulltext + " "
				all_doc_list.append([])

			print("Detected doctext:", file=sys.stderr)
			for ST in sentree_list:
				print(ST.text, file=sys.stderr)

			pattern = re.compile(r'([^a-zA-Z0-9 ])\.(\s*)')
			spacy_input = pattern.sub(r"\1 .\2", document_fulltext)
			self.ner = nlp(spacy_input)

			tokenized_doc = [token.text for token in self.ner]

			# Get all sentence threhsolds at once
			thresholds = [0]
			curr_size = 0
			for i in range(len(tokenized_doc)):
				token = tokenized_doc[i]
				if token == ".":
					thresholds.append(i + 1)
			
			# error check here
			if len(thresholds) != len(sentree_list) + 1:
				if debug_print:
					print(thresholds, file=sys.stderr)
					print(len(sentree_list), file=sys.stderr)

					i = 0
					while i < len(sentree_list) and i < len(thresholds) - 1:
						new_thresh = thresholds[i+1]
						last_thresh = thresholds[i]
						print(sentree_list[i].text, file=sys.stderr)
						print([token.text for token in self.ner[last_thresh:new_thresh]], file=sys.stderr)
						print("", file=sys.stderr)
						i += 1

					print("SPACY INPUT:", file=sys.stderr)
					print(spacy_input, file=sys.stderr)
				if not safety:
					raise ValueError

			# Update ner on all ST before running coref
			beginning = 0
			for i in range(1, len(thresholds)):
				end = thresholds[i]

				curr_slice = self.ner[beginning:end]

				ST = sentree_list[i-1]
				ST.update_ner(self.ner)
				ST.bubble_ner(curr_slice, corenlp=False)
				ST.align_ner()

				beginning = end
		except:
			if debug_print:
				print("Failed to get spacy ner tags", file=sys.stderr)
			if not safety:
				raise
		return

	# Bubble up NER data, when fed stuff that begins at the ground level
	def bubble_ner(self, tokens, corenlp=True):
		try:
			# GDI spacy why do you have to be different
			if not corenlp:
				tokens = self.align_ner_tokens(tokens)
				if tokens is None:
					if debug_print:
						print("Failed to collect set of aligned SpaCY tokens GDI", file=sys.stderr)
					if not safety:
						raise ValueError

			# Hooray we can do things
			node_stack = LifoQueue()
			frontier_1 = Queue()
			frontier_2 = Queue()

			# Do BFS, put onto a stack, so can pull stuff in reverse BFS order
			frontier_1.put_nowait((0,self.t))
			while not frontier_1.empty():
				while not frontier_1.empty():
					left_ind,curr = frontier_1.get_nowait()
					node_stack.put_nowait(curr)

					acclen = 0
					for i in range(len(curr)):
						my_ind = left_ind + acclen
						if curr[i].height() > 2:
							frontier_2.put_nowait((my_ind, curr[i]))
						else:
							if corenlp:
								curr[i].corenlp_tag = tokens[my_ind][1]
							else:
								curr[i].spacy_tag = tokens[my_ind]
						acclen += len(curr[i].leaves())

				temp = frontier_1
				frontier_1 = frontier_2
				frontier_2 = temp

			# Now pull off stack, and allow it to bubble up
			while not node_stack.empty():
				node = node_stack.get_nowait()
				if node.label() == "NP":
					# dostuff
					for i in range(len(node)):
						if node[i].label()[0] == "N":
							try:
								if corenlp:
									node.corenlp_tag = node[i].corenlp_tag
								else:
									node.spacy_tag = node[i].spacy_tag
							except AttributeError:
								pass
				elif node.label() == "PP":
					found_np = False
					found_tag = False
					for i in range(len(node)):
						if node[i].label() == "NP":
							try:
								if corenlp:
									node.corenlp_tag = node[1].corenlp_tag
								else:
									node.spacy_tag = node[1].spacy_tag
								found_np = True
							except AttributeError:
								pass
						elif not found_np and node[i].label() == "PP":
							try:
								if corenlp:
									node.corenlp_tag = node[1].corenlp_tag
								else:
									node.spacy_tag = node[1].spacy_tag
								found_tag = True
							except AttributeError:
								pass

					if not (found_np or found_tag) and debug_print:
						print("PP FAILED", file=sys.stderr)
						node.pretty_print()
		except:
			if debug_print:
				print("Somehow failed to bubble up ner data", file=sys.stderr)
			if not safety:
				raise
		return

	# align spacy tokens to corenlp tokens
	def align_ner_tokens(self, spacy_tokens):
		try:
			spacy_len = len(spacy_tokens)
			corenlp_len = len(self.text)
			if corenlp_len == spacy_len:
				return [token.ent_type_ for token in spacy_tokens]
			else:
				new_tokens = [''] * corenlp_len
				if spacy_len > corenlp_len:
					i = 0
					while i < corenlp_len:
						if spacy_tokens[i].text.lower() == self.text[i].lower():
							new_tokens[i] = spacy_tokens[i].ent_type_
						else:
							break
						i += 1
					
					start_offset = i
					spacy_ind = spacy_len - 1
					core_ind = corenlp_len - 1

					while spacy_ind > start_offset and core_ind > start_offset:
						if spacy_tokens[spacy_ind].text.lower() == self.text[core_ind].lower():
							new_tokens[core_ind] = spacy_tokens[spacy_ind].ent_type_
						else:
							break
						spacy_ind -= 1
						core_ind -= 1
					end_offset = spacy_ind
				elif debug_print:
					print("RED ALERT RED ALERT RED ALERT", file=sys.stderr)
				return new_tokens
		except:
			if not safety:
				raise
		return None

	# Assumes all NER data (both spacy and corenlp) has been accumulated and bubbled. have SPACY advanced tags take precedence
	def align_ner(self):
		
		return

# Given a list of sentence ending thresholds and mention, find which sentence it belongs to
def get_sent_num(thresholds, start):
	# print(thresholds, file=sys.stderr)
	if start >= thresholds[-1]:
		return len(thresholds) - 1

	loind = 0
	hiind = len(thresholds) - 2

	while loind < hiind:
		loval = thresholds[loind]
		hival = thresholds[hiind + 1]
		approx_size = (hival - loval) // (hiind + 1 - loind)
		hyp = min(hiind, ((start - loval) // approx_size) + loind)
		# print(str(loval) + ", " + str(hival) + ", " + str(approx_size) + ", " + str(thresholds[hyp]) + ", " + str(start), file=sys.stderr)

		if start >= thresholds[hyp]:
			if start < thresholds[hyp + 1]:
				hiind = hyp
				loind = hyp
			else:
				loind = hyp + 1
		else:
			if start >= thresholds[hyp - 1]:
				loind = hyp - 1
				hiind = hyp - 1
			else:
				hiind = hyp - 2
	# print("Found <" + str(start) + "> in bucket [" + str(thresholds[loind]) + ", " + str(thresholds[loind + 1]) + ")", file=sys.stderr)

	return loind

def reconstitute_sentence(raw, forCoreNLP=False):
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
	almost = pattern1.sub(r'\1', pattern2.sub('\'s ', reconstruct_quotes)).replace(' \' ', '\' ')
	if almost[-1] == "." and almost[-2] != " ":
		almost = almost[:-1] + " ."
	return almost

def valid_np(t):
	try:
		if t.label() != "NP":
			return False
		else:
			return has_valid_np(t)
	except:
		if debug_print:
			print("WTH? somehow failed valid_np", file=sys.stderr)
		if not safety:
			raise
	return False

def valid_vp(t):
	try:
		if t.label() != "VP":
			return False
		else:
			return has_valid_vp(t)
	except:
		if debug_print:
			print("WTH? somehow failed valid_vp", file=sys.stderr)
		if not safety:
			raise
		return False

def has_valid_np(t):
	try:
		pos_trunc = [p[1][:2] for p in t.pos()]
		return "NN" in pos_trunc
	except:
		if debug_print:
			print("WTH? somehow failed has_valid_np", file=sys.stderr)
		if not safety:
			raise
		return False

def has_valid_vp(t):
	try:
		pos_trunc = [p[1][:2] for p in t.pos()]
		return "VB" in pos_trunc
	except:
		if debug_print:
			print("WTH? somehow failed has_valid_vp", file=sys.stderr)
		if not safety:
			raise
		return False

def valid_s(t):
	try:
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
	except:
		if debug_print:
			print("WTH? somehow failed valid_np==s", file=sys.stderr)
		if not safety:
			raise
	return False

def validate_s(t):
	try:
		if t.leaves()[-1] != ".":
			return False
		elif t.label() == "ROOT":
			return valid_s(t[0])
		elif t.label() != "S":
			return False
		else:
			return has_valid_np(t) and has_valid_vp(t)
	except:
		if debug_print:
			print("WTH? somehow failed validate_s", file=sys.stderr)
		if not safety:
			raise
	return False

def use_second(np1, delim, np2):
	try:
		if not is_nnp(np1) and is_nnp(np2):
			return True
		elif not is_nnp(np1) and not is_nnp(np2) and delim.leaves()[0] == ":":
			return True
	except:
		if debug_print:
			print("WTH? somehow failed use_second", file=sys.stderr)
		if not safety:
			raise
	return False

def is_nnp(np):
	try:
		if np.label() != "NP":
			return False

		if "NNP" in [np[i].label() for i in range(len(np))]:
			return True

		for (word,pos) in np.pos():
			if pos in ["NN", "NNP", "NNS", "NNPS"]:
				if not word[0].isupper():
					return False
	except:
		if debug_print:
			print("WTH? somehow failed is_nnp", file=sys.stderr)
		if not safety:
			raise
	return True

def getSBARQuestion(SBAR, root):
	retlist = []
	try:
		if SBAR.height() > 3:
			lemmatizer = nltk.stem.WordNetLemmatizer()
			if SBAR[0].label() == "WHNP" and SBAR[0].height() > 2:
				# case who/what
				if SBAR[0][0].label() == "WDT":
					retlist.append(reconstitute_sentence(["SR: What"] + SBAR.leaves()[1:] + ["?"]))
				elif SBAR[0][0].label() == "WP":
					retlist.append(reconstitute_sentence(["SR: Who"] + SBAR.leaves()[1:] + ["?"]))
			elif len(SBAR) > 1 and SBAR[0].label() == "WHADVP" and SBAR[0].height() > 2:
				# case where/when
				if SBAR[0][0].label() == "WDT":
					retlist.append(reconstitute_sentence(["SR: What"] + SBAR.leaves()[1:] + ["?"]))
				elif SBAR[0][0].label() == "WP":
					retlist.append(reconstitute_sentence(["SR: Who"] + SBAR.leaves()[1:] + ["?"]))
				elif SBAR[0][0].label() == "WRB":
					S = SBAR[1]
					if S.label()[-1] == "S" and S.height() > 2:
						if len(S) >= 2 and len(S[0].label()) >= 2 and S[0].label()[-2:] == "NP" and len(S[1].label()) >= 2 and S[1].label()[-2:] == "VP":
							if len(S[1]) >= 2 and S[1][0].label()[:2] == "VB" and S[1][1].label()[:2] != "VB":
								vbn = S[1][0].leaves()[0]
								conj_verb = lemma(vbn)
								verblvs = []
								for i in range(1, len(S[1])):
									verblvs += S[1][i].leaves()
								retlist.append(reconstitute_sentence(["SR: "] + SBAR[0][0].leaves() + ["did"] + S[0].leaves() + [conj_verb] + verblvs + ["?"]))
			elif SBAR[0].label() == "IN" and SBAR[0].height() >= 2:
				# Case on that = what + invert, although = else
				if len(SBAR) > 1 and SBAR[1].label()[-1] == "S":
					if len(SBAR[1]) == 2:
						retlist.append(reconstitute_sentence(["SR: What"] + SBAR[1][1].leaves() + ["?"]))
						# if has_valid_np(SBAR[1][0]) and has_valid_vp(SBAR[1][1]):
						#	 retlist.append(reconstitute_sentence(["What", "was"] + SBAR[1].leaves() + ["?"]))
					else:
						retlist.append(reconstitute_sentence(["SR: What"] + SBAR[1][0].leaves() + ["?"]))
			elif SBAR[0].label() == "S":
				if len(SBAR[0]) == 2:
					retlist.append(reconstitute_sentence(["SR: What"] + SBAR[0][1].leaves() + ["?"]))
	except:
		if debug_print:
			print("WTH? somehow failed getSBARQuestion", file=sys.stderr)
		if not safety:
			raise
	return retlist


def acc_stage(stage):
	test = stage + 1
	if stage == 9:
		return True,3
	else:
		return False,test

def remove_q_dups(ql, t_order):
	t_order = list(reversed(t_order))
	out = {}
	res = []
	for q in ql:
		qtext = q[4:]
		ttext = q[:2]
		if qtext in out:
			ttext_2 = out[qtext]
			try: #Skip if tags are incomparable
				if t_order.index(ttext) > t_order.index(ttext_2):
					out[qtext] = ttext
			except:
				pass
		else:
			out[qtext] = ttext

	for qtext in out:
		res.append(out[qtext]+": "+qtext)
	return res

def preprocess(treelist, parser):
	preprocessed_questions = []
	preprocessed_trees = []
	try:
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
		for i in range(len(node_list)):
			FrontierQueue.put_nowait((i, 1))

		ner_stage = 8
		awaiting_ner = []
		while not FrontierQueue.empty():
			node_id,stage = FrontierQueue.get_nowait()
			curr_node = node_list[node_id]
			if debug_print:
				print("Stage " + str(stage) + ":", file=sys.stderr)
				print(curr_node.text, file=sys.stderr)
				print(curr_node.fulltext, file=sys.stderr)
				print("-----------------------\n", file=sys.stderr)
				# print(str(stage) + ": " + reconstitute_sentence(curr_node.t.leaves()), file=sys.stderr)
			full_replace = curr_node.handle_stage(stage, preprocessed_questions)
			if len(curr_node.children[stage]) > 0:
				# print("Ho: " + str(stage), file=sys.stderr)
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
			else:
				rollover, new_stage = acc_stage(stage)
				if not rollover:
					if new_stage != ner_stage:
						FrontierQueue.put_nowait((node_id, new_stage))
					else:
						awaiting_ner.append(node_id)

		finished_coref = updated_root.do_coref(awaiting_ner, node_list)
		updated_root = node_list[finished_coref[0]]

		preprocessed_trees = []
		curr = updated_root
		while curr is not None:
			preprocessed_trees.append(curr)
			curr = curr.nextST
	except:
		if debug_print:
			print("Failed at everything: likely CoreNLP not running since we have literally no treesß", file=sys.stderr)
		if not safety:
			raise
	return preprocessed_trees, preprocessed_questions


if __name__ == "__main__":
	if debug_print:
		print(type(str(english_classifiers_path)))
		st = StanfordNERTagger(str(english_classifiers_path), path_to_jar = str(ner_jar_path), encoding='utf-8')
		print("OK")