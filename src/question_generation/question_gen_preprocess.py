from collections import defaultdict

class SenTree:
    def __init__(self, t, parser, prevST=None, nextST=None):
        self.t = t
        self.fulltext = " ".join(t.leaves()) if t else ""
        self.text = t.leaves()
        self.type = None
        self.flags = defaultdict(lambda:False)
        self.parentheticals = None
        self.turns = None
        self.children = {}
        self.prevST = prevST
        self.nextST = nextST
        self.to_be_swapped = False

	#1
	def turn_of_phrase(self,text=self.text, pos = [p[1] for p in self.t.pos()],turns = []):
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
    				elif text[i] = "had":
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
    				turns.append(start,text[i+2])
    				restext = text[:start]+[replace]+text[end:]
    				respos = pos[:start]+["REPLACED"]+pos[end:]
    				break
    	if restext:
    		self.turn_of_phrase(restext, respos, turns)
    	elif turns:
	    	newtree = self.parser.parse_text(text, timeout=5000)
	    	newtree = next(newtree)
	    	child = SenTree(newtree, self.parser)
	    	child.type = 1
	    	child.flags = self.flags
	    	child.turns = self.turns
	    	child.flags["turns_removed"] = True
	    	self.children[1] = [child]

	#2
	def fix_sinv(self):
		result = []
		return

	#3
	def sbar_remove(self):
		for i in range(len(self.t)-1):
			if self.t[i].label == "NP" and self.t[i+1].label() == "VP" and self.t[i+1][0].label()[:2] == "VB" and self.t[i+1][1].label() == "NP" and self.t[i+1][2].label() == "SBAR":
				self.flags["3_applied"] = True
					new_sent = ""
					for k in range(i+1):
						new_sent += " ".join(self.t[k].leaves()) + " "
					new_sent += " ".join(self.t[i+1].leaves()[:2]+self.t[i+1].leaves()[3:]) + " "
					for m in range(i+2,len(self.t)):
						new_sent += " ".join(self.t[m].leaves()) + " "
					new_sent = new_sent[:-1]
					newtree = self.parser.parse_text(new_sent)
					newtree = next(newtree)
					child = SenTree(newtree, self.parser)
					child.type = 3
					child.flags = self.flags
					self.children[3].append(child)

	#4
	def s_cc_s_separation(self):
		for i in range(len(self.t)-2):
			if (valid_s(self.t[i]) and self.t[i+1].label() == "CC" and valid_s(self.t[i+2])):
		    	newtree1 = self.parser.parse_text(" ".join(self.t[i].leaves()), timeout=5000)
		    	newtree1 = next(newtree1)
		    	newtree2 = self.parser.parse_text(" ".join(self.t[i+2].leaves()), timeout=5000)
		    	newtree2 = next(newtree2)
		    	child1 = SenTree(newtree1, self.parser)
		    	child1.type = 4
		    	child1.flags = self.flags
		    	child2 = SenTree(newtree2, self.parser)
		    	child2.type = 4
		    	child2.flags = self.flags
		    	self.children[4] = [child1, child2]
				return
			elif i+3 < len(self.t) and valid_s(self.t[i]) and self.t[i+1].label() == "," and self.t[i+2].label() == "CC" and valid_s(self.t[i+3]):
		    	newtree1 = self.parser.parse_text(" ".join(self.t[i].leaves()), timeout=5000)
		    	newtree1 = next(newtree1)
		    	newtree2 = self.parser.parse_text(" ".join(self.t[i+3].leaves()), timeout=5000)
		    	newtree2 = next(newtree2)
		    	child1 = SenTree(newtree1, self.parser)
		    	child1.type = 4
		    	child1.flags = self.flags
		    	child2 = SenTree(newtree2, self.parser)
		    	child2.type = 4
		    	child2.flags = self.flags
		    	self.children[4] = [child1, child2]
				return

	#5
	def sbarpp_s_rearrange(self):
		if len(self.t) == 2 and self.t[0].label() in ["PP", "SBAR"] and valid_s(self.t[1]):
	    	newtree1 = self.parser.parse_text(" ".join(self.t[1].leaves())+" "+" ".join(self.t[0].leaves()), timeout=5000)
	    	newtree1 = next(newtree1)
	    	newtree2 = self.parser.parse_text(" ".join(self.t[1].leaves()), timeout=5000)
	    	newtree2 = next(newtree2)
	    	child1 = SenTree(newtree1, self.parser)
	    	child1.type = 5
	    	child1.flags = self.flags
	    	child2 = SenTree(newtree2, self.parser)
	    	child2.type = 5
	    	child2.flags = self.flags
	    	self.children[5] = [child1, child2]
		elif len(self.t) == 3 and self.t[0].label() in ["PP", "SBAR"] and self.t[1].label() == "," and valid_s(self.t[2]):
	    	newtree1 = self.parser.parse_text(" ".join(self.t[2].leaves())+" "+" ".join(self.t[0].leaves()), timeout=5000)
	    	newtree1 = next(newtree1)
	    	newtree2 = self.parser.parse_text(" ".join(self.t[2].leaves()), timeout=5000)
	    	newtree2 = next(newtree2)
	    	child1 = SenTree(newtree1, self.parser)
	    	child1.type = 5
	    	child1.flags = self.flags
	    	child2 = SenTree(newtree2, self.parser)
	    	child2.type = 5
	    	child2.flags = self.flags
	    	self.children[5] = [child1, child2]

	#6 - top-level only atm
	def to_be_equiv(self):
		if self.flags["to_be_swapped"]:
			return
		else:
			self.flags["to_be_swapped"] = True
			self.children[6] = []
		to_be_conj = ["be", "am", "is", "are", "was", "were", "been", "being"]
		for i in range(len(self.t)-1):
			if valid_np(self.t[i]) and valid_vp(self.t[i+1]) and len(self.t[i+1] > 1) and self.t[i+1][0].label()[:2] == "VB" and valid_np(self.t[i+1][1]):
				if " ".join(self.t[i+1][0].leaves()) in to_be_conj:
					new_sent = ""
					for k in range(i):
						new_sent += " ".join(self.t[k].leaves()) + " "
					new_sent += " ".join(self.t[i+1][1].leaves()) + " "
					new_sent += " ".join(self.t[i+1][1].leaves()) + " "
					new_sent += " ".join(self.t[i].leaves()) + " "
					for m in range(i+2,len(self.t)):
						new_sent += " ".join(self.t[m].leaves()) + " "
					new_sent = new_sent[:-1]
					newtree = self.parser.parse_text(new_sent)
					newtree = next(newtree)
					child = SenTree(newtree, self.parser)
					child.type = 6
					child.flags = self.flags
					self.children[6].append(child)
	#7
	def appositive_removal(self):
		return
	#8
    def parenthetical_removal(self):
    	result = []
    	count = 0
    	start = None
    	parentheticals = []
    	for word in self.text:
    		if word == "(":
    			count += 1
    			if count == 0:
    				parentheticals.append({"index" : len(res), "text":"("})
    			else:
    				parentheticals[-1]["text"].append(" (")
    		elif word == ")":
    			count -= 1
    			parentheticals[-1]["text"].append(" )")
    		elif count == 0:
    			result.append(word)
    		else:
    			parentheticals[-1]["text"].append(" "+word)
    	if parentheticals:
	    	newtree = self.parser.parse_text(result, timeout=5000)
	    	newtree = next(newtree)
	    	child = SenTree(newtree, self.parser)
	    	child.type = 8
	    	child.flags = self.flags
	    	child.parentheticals = parentheticals
	    	self.children[8] = [child]
	    	child.flags["parentheticals_removed"] = True

	#9
	def npvp_combo(self):
		subs = self.t.subtrees()
		self.flags["npvp_extracted"] = True
		self.children[9] = []
		for sub in subs:
			if len(sub) > 1:
				for i in range(len(sub)-1):
					if valid_np(sub[i]) and valid_vp(sub[i+1]):
						newtree = self.parser.parse_text(" ".join(sub[i].leaves())+" "+" ".join(sub[i+1].leaves()))
						newtree = next(newtree)
						child = SenTree(newtree, self.parser)
						child.type = 9
						child.flags = self.flags
						self.children[9].append(child)

def valid_np(t):
	if t.label() != "NP":
		return False
	else:
		pos_trunc = [p[1][:-1] for p in t.pos()]
		return "NN" in pos_trunc

def valid_vp(t):
	if t.label() != "VP":
		return False
	else:
		pos_trunc = [p[1][:-1] for p in t.pos()]
		return "VB" in pos_trunc

def valid_s(t):
	#Maybe remove this
	if t.label() == "ROOT":
		return valid_s(t[0])
	elif t.label != "S":
		return False
	else:
		for i in range(len(t)-1):
			if valid_np(t[i]) and valid_vp(t[i+1]):
				return True
	return False

def preprocess(treelist, parser):
	preprocessed_trees = []

	for tree in treelist:
		root = SenTree(tree, parser)

	return preprocessed_trees