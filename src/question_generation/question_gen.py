import nltk
from nltk.parse import corenlp
from queue import Queue
import pattern.en
from pattern.en import lemma
from src.question_generation.question_gen_preprocess import *
import copy


def getWhoWhat(t):
    out = []
    for candidate in t.subtrees():
        if candidate.label() == "S":
            if len(candidate) >= 2 and candidate[0].label() == "NP" and candidate[1].label() == "VP":
                vptext = candidate[1].leaves()
                testpos = candidate.pos()
                if testpos[0][1] != "NNP" and testpos[0][1] != "NNPS":
                    vptext[0] = vptext[0].lower()
                vptext = " ".join(vptext)
                if vptext and vptext[-1] in ".!?":
                    vptext = vptext[:-1]
                out.append("Who or what %s?" % vptext)
    return out

def getWhoWhatNP(t):
    out = []
    detected_NP = []
    FrontierQueue = Queue()
    FrontierQueue.put_nowait(t)

    while not FrontierQueue.empty():
        curr_node = FrontierQueue.get_nowait()
        # print(curr_node.height())
        # print(curr_node)
        contains_pp = False
        if curr_node.height() > 2:
            for i in range(len(curr_node)):
                if curr_node[i].label() == "PP" or curr_node[i].label() == "JJ":
                    contains_pp = True

        if contains_pp:
            contains_pp = False
            if curr_node.label() == "NP":
                detected_NP.append(curr_node)
            else:
                for i in range(len(curr_node)):
                    FrontierQueue.put_nowait(curr_node[i])
        elif curr_node.height() > 2:
            for i in range(len(curr_node)):
                FrontierQueue.put_nowait(curr_node[i])
    
    for NP in detected_NP:
        temp = " ".join(NP.leaves())
        was = "was"
        plural = False
        testpos = NP.pos()
        for pos in testpos:
            if pos[1] == "NNS" or pos[1] == "NNPS":
                plural = True
            elif pos[1] == "NN" or pos[1] == "NNP":
                break
        if plural:
            was = "were"
        out.append("Who or what " + was + " the %s?" % temp)

    return out

def getBinarySimple(t):
    out = []
    for candidate in t.subtrees():
        if candidate.label() == "S":
            lemmatizer = nltk.stem.WordNetLemmatizer()
            if len(candidate) >= 2 and candidate[0].label() == "NP" and candidate[1].label() == "VP":
                #Top level verified.
                nptext = " ".join(candidate[0].leaves())
                if nptext and nptext[-1] in ".!?":
                    nptext = nptext[:-1]
                verbforms = ["VBD", "VBG", "VBN", "VBP", "VBZ"]
                count = 0
                for  i in range(1,len(candidate[1])):
                    if candidate[1][i].label() in verbforms:
                        count += 1
                if len(candidate[1]) and candidate[1][0].label() in verbforms and not count:
                    #Second level verified
                    #vb = lemmatizer.lemmatize(candidate[1][0])
                    vbn = candidate[1][0].leaves()[0]
                    if candidate[1][0].label() != "VBN":
                        try:
                            vbn = pattern.en.conjugate(vbn, tense=pattern.en.PAST+pattern.en.PARTICIPLE)
                        except:
                            continue
                    vptext = " ".join(candidate[1].leaves())
                    if vptext and vptext[-1] in ".!?":
                        vptext = vptext[:-1]
                    vpptext = " ".join([vbn]+candidate[1].leaves()[1:])
                    if vpptext and vpptext[-1] in ".!?":
                        vpptext = vpptext[:-1]
                    prefdict = {"VBZ": "Has", "VBG": "Has", "VBP": "Has/Have", "VBD":"Had", "VBN":"Had"}
                    if lemmatizer.lemmatize(candidate[1][0].leaves()[0]) != "be":
                        if candidate[1][0].label() in prefdict:
                            out.append("%s %s %s?" % (prefdict[candidate[1][0].label()],nptext,vpptext))
                    elif candidate[1][0].label() in prefdict:
                        if candidate[1][0].label() == "VBD":
                            pref = "Did"
                        else:
                            pref = "Does"
                        out.append("%s %s have %s?" % (pref,nptext,vptext[1:]))
    return out

def getBinaryAuxiliary(t):
    out = []
    for candidate in t.subtrees():
        if candidate.label() == "S":
            lemmatizer = nltk.stem.WordNetLemmatizer()
            if len(candidate) >= 2 and candidate[0].label() == "NP" and candidate[1].label() == "VP":
                #Top level verified.
                verbforms1 = ["VBD", "VBP", "VBZ"]
                verbforms2 = ["VBG", "VBN"]
                targets = ["is", "are", "has", "have"]
                if len(candidate[1]) > 1 and candidate[1][0].label() in verbforms1 and candidate[1][0].leaves()[0] in targets and candidate[1][1].label() in verbforms2:
                    #Second level verified
                    nptext = " ".join(candidate[0].leaves())
                    if nptext and nptext[-1] in ".!?":
                        nptext = nptext[:-1]
                    vptext = " ".join(candidate[1].leaves()[1:])
                    if vptext and vptext[-1] in ".!?":
                        vptext = vptext[:-1]
                    verb = candidate[1].leaves()[0]
                    out.append("%s %s %s?" % (verb.capitalize(), nptext, vptext))
    return out



def handle_stage_1(parse_tree):
    # Have to manually recreate the logic since can't get Tregex working.
    # The following is extremely inelegant, inefficient and ugly, but it should be functional

    # Handy lists for storage. Guarantee all that once were of label are in there, do not guarantee have not been labelled "UNMV"
    VP_List = []
    NP_List = []
    PP_List = []
    S_List = []
    CC_List = []
    ADJP_List = []
    ADVP_List = []
    SBAR_List = []
    S_Tot_List = []

    # Common use frontier: make sure to sanitize after use
    frontier = Queue()

    # Populate lists for later access
    for sub in parse_tree.subtrees():
        if sub.label() == "VP":
            VP_List.append(sub)
        elif sub.label() == "NP":
            NP_List.append(sub)
        elif sub.label() == "PP":
            PP_List.append(sub)
        elif sub.label()[0] == 'ADJP':
            ADJP_List.append(sub)
        elif sub.label()[0] == 'ADVP':
            ADVP_List.append(sub)
        elif sub.label() == "S":
            S_List.append(sub)
            S_Tot_List.append(sub)
        elif sub.label() == "SBAR":
            SBAR_List.append(sub)
            S_Tot_List.append(sub)
        elif sub.label() == "CC":
            CC_List.append(sub)
        elif sub.label()[0] == 'S':
            S_Tot_List.append(sub)

    # Stage 1: Mark unmovable phrases
    for sub in VP_List:
        #1 VP < (S=UNMV $,, /,/)
        # Mark S as UNMV if is a child of a VP, and follows after a ,
        if sub.label() == "VP" and sub.height() > 2:
            init_comma = False
            detect_S = None
            for i in range(len(sub)):
                if sub[i].label() == ',':
                    if not init_comma:
                        init_comma = True
                elif init_comma and sub[i].label() == "S":
                    detect_S = i
                    break
            if detect_S is not None:
                sub[i].set_label("UNMV:S")
        
    #2 "S < PP|ADJP|ADVP|S|SBAR=UNMV > ROOT"
    # Mark nodes directly under the root S as UNMV
    undesirables_2 = ["PP", "ADJP", "ADVP", "S", "SBAR"]
    if parse_tree.label() == "ROOT" and len(parse_tree) > 0 and parse_tree.height() > 2:
        for i in range(len(parse_tree)):
            if parse_tree[i].label() == "S":
                target = parse_tree[i]
                for j in range(len(target)):
                    if target[j].label() in undesirables_2:
                        target[j].set_label("UNMV:" + target[j].label())

    #3 "/\\.*/ < CC << NP|ADJP|VP|ADVP|PP=UNMV"             <== highly questionable
    # Mark phrases under conjunctions as unmovable
    undesirables_3 = ["NP", "ADJP", "VP", "ADVP", "PP"]
    for sub in CC_List:
        if sub.label() == "CC":
            for descendant in sub.subtrees():
                if descendant.label() in undesirables_3 and descendant != sub:
                    descendant.set_label("UNMV:" + descendant.label())

    #4 "SBAR < (IN|DT < /[^that]/) << NP|PP=UNMV"
    # If there's a SBAR that begins with something other than "that", then we do not make questions from them
    for sub in SBAR_List:
        if sub.label() == "SBAR" and sub.height() > 2:
            failed = False
            for i in range(len(sub)):
                if sub[i].label() == "IN" or sub[i].label() == "DT":
                    words = sub[i].leaves()
                    if "that" not in words and "That" not in words:
                        failed = True
                        break
            if failed:
                for descendant in sub.subtrees():
                    if (descendant.label() == "NP" or descendant.label() == "PP") and descendant != sub:
                        descendant.set_label("UNMV:" + descendant.label())
    
    #5 SBAR < /^WH.*P$/ << NP|ADJP|VP|ADVP|PP=UNMV
    # If there is a WH_P under the SBAR, mark targets unmovable
    undesirables_5 = undesirables_3
    for sub in SBAR_List:
        if sub.label() == "SBAR" and sub.height() > 2:
            found_WHP = False
            for i in range(len(sub)):
                if sub[i].label()[:2] == "WH" and sub[i].label()[-1] == "P":
                    found_WHP = True
                    break
            if found_WHP:
                for i in range(len(sub)):
                    if sub[i].label() in undesirables_5:
                        sub[i].set_label("UNMV:" + sub[i].label())

    #6 "SBAR <, IN|DT < (S < (NP=UNMV !?,, VP))"
    # Avoid generating bad questions due to complement phrases missing complimentizer
    for sub in SBAR_List:
        if sub.label() == "SBAR" and sub.height() > 2:
            if len(sub) > 0 and (sub[0].label() == "IN" or sub[0].label() == "DT"):
                for i in range(len(sub)):
                    if sub[i].label() == "S":
                        vp_detected = False
                        for j in range(len(sub[i])):
                            if sub[i][j].label() == "VP":
                                vp_detected = True
                            elif sub[i][j].label() == "NP":
                                if not vp_detected:
                                    sub[i][j].set_label("UNMV:NP")

    #7 "S < (VP <+(VP) (VB|VBD|VBN|VBZ < be|being|been|is|are|was|were|am) <+(VP) (S << NP|ADJP|VP|VP|ADVP|PP=UNMV))"
    # Mark a node if it's under a clause that serves as a predicate of a larger clause with a copula verb.
    # If there exists a "be" verb of type "VB_" under a direct chain of "VP", then all "S" under a direct chain of "VP" shalt be cleansed
    undesirables_7 = undesirables_3
    target_7_1 = ["VB", "VBD", "VBN", "VBZ"]
    target_7_2 = ["be", "being", "been", "is", "are", "was", "were", "am"]
    for sub in S_List:
        if sub.label() == "S":
            found_helper = False
            curr_height = sub.height()
            frontier.put_nowait(sub)
            while not frontier.empty():
                curr = frontier.get_nowait()

                # If is VP on lowest possible level, check if children have VB, etc.
                if curr.height() == 3:
                    contained = curr.pos()
                    for elem in contained:
                        if contained[0] in target_7_2 and contained[1] in target_7_1:
                            found_helper = True
                            break

                if found_helper:
                    while not frontier.empty():
                        frontier.get_nowait()
                    break
                else:
                    if curr.height() >= 4:
                        for i in range(len(curr)):
                            if curr[i].label() == "VP":
                                frontier.put_nowait(curr[i])
            
            if found_helper:
                frontier.put_nowait(sub)
                while not frontier.empty():
                    curr = frontier.get_nowait()

                    if curr.height() > 3:
                        for i in range(len(curr)):
                            if curr[i].label() == "VP":
                                frontier.put_nowait(curr[i])
                            elif curr[i].label() == "S":
                                for descendant in curr[i].subtrees():
                                    if descendant.label() in undesirables_7:
                                        descendant.set_label("UNMV:" + descendant.label())


    #8 "NP << (PP=UNMV !< (IN < of|about))"
    # PP with preposition that is not of/about are theorized to be adjuncts, which should not become answer phrases
    for sub in NP_List:
        if sub.label() == "NP":
            for descendant in sub.subtrees():
                if descendant.label() == "PP" and sub.height() > 2:
                    found_invalid = False
                    for i in range(len(descendant)):
                        if descendant[i].label() == "IN":
                            if not ("of" in descendant[i].leaves() or "about" in descendant[i].leaves()):
                                found_invalid = True
                                break
                    descendant.set_label("UNMV:PP")

    #9 "PP << PP=UNMV"
    # Flag nested prepositions as troublesome
    for sub in PP_List:
        if sub.label() == "PP":
            for descendant in sub.subtrees():
                if descendant.label() == "PP" and descendant != sub:
                    descendant.set_label("UNMV:PP")

    #10 "NP $ VP << PP=UNMV"
    # Disallow prepositional phrases within subjects from being moved
    for sub in parse_tree.subtrees():
        found_NP = []
        found_VP = []
        if sub.height() > 2:
            for i in range(len(sub)):
                if sub[i].label() == "NP":
                    found_NP.append(sub[i])
                elif sub[i].label() == "VP":
                    found_VP.append(sub[i])
        if len(found_VP) > 0:
            for NP in found_NP:
                for descendant in NP.subtrees():
                    if descendant.label() == "PP":
                        descendant.set_label("UNMV:PP")
        found_NP = None
        found_VP = None

    #11 "SBAR=UNMV [ !> VP | $-- /,/ | < RB ]"
    # if SBAR contains an adverb, or is not directly under a VP, or has a comma in front of it, we do not like it
    frontier.put_nowait((parse_tree, False))
    while not frontier.empty():
        curr,parentSuccess = frontier.get_nowait()

        if curr.label() == "SBAR" and curr.height() > 2:
            found_adverb = False
            for i in range(len(curr)):
                if curr[i].label() == "RB":
                    found_adverb = True
                    break
            if (not parentSuccess) or found_adverb:
                curr.set_label("UNMV:SBAR")
            else:
                parentSuccess = (curr.label() == "VP")
                found_comma = False
                for i in range(len(curr)):
                    if curr[i].label() == ",":
                        found_comma = True
                    elif curr.height() > 2:
                        frontier.put_nowait((curr[i], (parentSuccess and (not found_comma))))
        # minimum layer is 4, likely higher
        elif curr.height() > 2:
            parentSuccess = (curr.label() == "VP")
            found_comma = False
            for i in range(len(curr)):
                if curr[i].label() == ",":
                    found_comma = True
                else:
                    frontier.put_nowait((curr[i], parentSuccess and (not found_comma)))

    #12 "SBAR=UNMV !< WHNP < (/^[^S].*/ !<< that|whether|how)"
    # SBAR that are children of verbs (already tagged), but not complements, should be marked
    desirables_12 = ["that", "whether", "how"]
    for sub in SBAR_List:
        if sub.label() == "SBAR" and sub.height() > 2:
            failed_whnp = True
            failed_spec = False
            for i in range(len(sub)):
                if sub[i].label() == "WHNP":
                    failed_whnp = False
                else:
                    if not failed_spec and sub[i].label()[0] != 'S':
                        text = sub[i].leaves()
                        found_spec = False
                        for word in text:
                            if word in desirables_12:
                                found_spec = True
                                break
                        if not found_spec:
                            failed_spec = True
            if failed_whnp and failed_spec:
                sub.set_label("UNMV:SBAR")

    #13 "NP=UNMV < EX"
    # set NP parents of EX to unavailable
    for sub in NP_List:
        if sub.label() == "NP" and sub.height() > 2:
            for i in range(len(sub)):
                if sub[i].label() == "EX":
                    sub.set_label("UNMV:NP")
                    break

    #14 "/^S/ < `` << NP|ADJP|VP|ADVP|PP=UNMV"
    # Mark phrases that occur with direct quotations
    undesirables_14 = undesirables_3
    for sub in S_Tot_List:
        if sub.label()[0] == 'S' and sub.height() > 2:
            # sub.pretty_print()
            for i in range(len(sub)):
                if sub[i].label() == "``":
                    for descendant in sub.subtrees():
                        if descendant.label() in undesirables_14 and descendant != sub:
                            descendant.set_label("UNMV:" + descendant.label())

    #15 "PP=UNMV !< NP"
    # Mark PP that don't contain a NP
    for sub in PP_List:
        if sub.label() == "PP" and sub.height() > 2:
            found_NP = False
            for i in range(len(sub)):
                if sub[i].label() == "NP":
                    found_NP = True
                    break
            if not found_NP:
                sub.set_label("UNMV:PP")

    #16 "NP=UNMV $ @NP"
    # Mark sibling NP sets as all unmv
    for sub in parse_tree.subtrees():
        first_target = None
        if sub.height() > 2:
            for i in range(len(sub)):
                if sub[i].label() == "NP" or sub[i].label() == "UNMV:NP":
                    if first_target is None:
                        first_target = sub[i]
                    else:
                        first_target.set_label("UNMV:NP")
                        sub[i].set_label("UNMV:NP")

    #17 "NP|PP|ADJP|ADVP << NP|ADJP|VP|ADVP|PP=UNMV"
    # Mark as unmovable all descendants of otherwise movable nodes
    undesirables_17 = undesirables_3
    for sub in (NP_List + PP_List + ADJP_List + ADVP_List):
        if sub.label()[:4] != "UNMV":
            for descendant in sub.subtrees():
                if descendant.label() in undesirables_17 and descendant != sub:
                    descendant.set_label("UNMV:" + descendant.label())

    #18 "@UNMV << NP|ADJP|VP|ADVP|PP=UNMV"
    # Mark as unmovable all descendants of an unmovable node
    undesirables_18 = undesirables_3
    frontier.put_nowait((parse_tree, False))
    while not frontier.empty():
        curr,marked = frontier.get_nowait()
        already_marked = curr.label()[:4] == "UNMV"
        marked = marked or already_marked

        if marked and not already_marked and (curr.label() in undesirables_18):
            curr.set_label("UNMV:" + curr.label())

        if curr.height() > 2:
            for i in range(len(curr)):
                frontier.put_nowait((curr[i], marked))

    # unmv_tregex = ["VP < (S=UNMV $,, /,/)", "S < PP|ADJP|ADVP|S|SBAR=UNMV > ROOT", "/\\.*/ < CC << NP|ADJP|VP|ADVP|PP=UNMV", "SBAR < (IN|DT < /[^that]/) << NP|PP=UNMV", "SBAR < /^WH.*P$/ << NP|ADJP|VP|ADVP|PP=UNMV", "SBAR <, IN|DT < (S < (NP=UNMV !?,, VP))", "S < (VP <+(VP) (VB|VBD|VBN|VBZ < be|being|been|is|are|was|were|am) <+(VP) (S << NP|ADJP|VP|VP|ADVP|PP=UNMV))", "NP << (PP=UNMV !< (IN < of|about))", "PP << PP=UNMV", "NP $ VP << PP=UNMV", "SBAR=UNMV [ !> VP | $-- /,/ | < RB ]", "SBAR=UNMV !< WHNP < (/^[^S].*/ !<< that|whether|how)", "NP=UNMV < EX", "/^S/ < `` << NP|ADJP|VP|ADVP|PP=UNMV", "PP=UNMV !< NP", "NP=UNMV $ @NP", "NP|PP|ADJP|ADVP << NP|ADJP|VP|ADVP|PP=UNMV", "@UNMV << NP|ADJP|VP|ADVP|PP=UNMV"]
    return [VP_List, NP_List, PP_List, S_List, CC_List, ADJP_List, ADVP_List, SBAR_List, S_Tot_List]

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

who_list = ["PERSON", "NORP", "ORG", "ORGANIZATION"]
where_list = ["LOC", "LOCATION", "GPE"]
when_list = ["DATE", "TIME"]

def gen_PP(phrases, parse):
    retlist = []
    frontier = LifoQueue()
    for phrase in phrases:
        print("Potential phrase: " + reconstitute_sentence(phrase.leaves()))
        tag = None
        try:
            tag = phrase.spacy_tag
            print("SPACY TAG: " + tag)
            if tag == "":
                tag = None
        except AttributeError:
            pass

        try:
            tag = phrase.corenlp_tag
            print("CORENLP TAG: " + tag)
        except AttributeError:
            pass

        if tag is not None:
            initial = ["How (NP)", "did"]
            if tag in where_list:
                initial = ["Where (PP)"]
            elif tag in when_list:
                initial = ["When (PP)"]
            
            frontier.put_nowait(parse.t)
            while not frontier.empty():
                curr = frontier.get_nowait()

                # minimum layer is 4, likely higher
                if curr.height() > 2:
                    for i in range(len(curr)-1, -1, -1):
                        if curr[i] != phrase:
                            frontier.put_nowait(curr[i])
                else:
                    test = curr.leaves()
                    if test[-1] == ".":
                        test = test[:-1]
                    initial += test
            initial += ["?"]
            retlist.append(reconstitute_sentence(initial))
        else:
            print("Phrase failed")
    return retlist

def gen_NP(phrases, parse):
    retlist = []
    frontier = LifoQueue()
    for phrase in phrases:
        print("Potential phrase: " + reconstitute_sentence(phrase.leaves()))
        tag = None
        try:
            tag = phrase.spacy_tag
            print("SPACY TAG: " + tag)
            if tag == "":
                tag = None
        except AttributeError:
            pass

        try:
            tag = phrase.corenlp_tag
            print("CORENLP TAG: " + tag)
        except AttributeError:
            pass

        # If we know the tag, use it
        if tag is not None:
            initial = ["What (NP)"]
            if tag in who_list:
                initial = ["Who (NP)"]

            frontier.put_nowait(parse.t)
            while not frontier.empty():
                curr = frontier.get_nowait()

                # minimum layer is 4, likely higher
                if curr.height() > 2:
                    for i in range(len(curr)-1, -1, -1):
                        if curr[i] != phrase:
                            frontier.put_nowait(curr[i])
                else:
                    test = curr.leaves()
                    if test[-1] == ".":
                        test = test[:-1]
                    initial += test
            initial += ["?"]
            retlist.append(reconstitute_sentence(initial))
        else:
            print("Phrase failed")
    return retlist

def gen_SBAR(phrases, parse):
    retlist = []
    for phrase in phrases:
        test = getSBARQuestion(phrase, parse)
        if test is not None:
            retlist.append(test)
    return retlist

def generate_questions(parse):
    if '.' not in parse.text:
        return []
    
    retlist = []
    
    parse_tree = parse.t

    # print("Initial Tree:")
    # parse_tree.pretty_print()

    # Stage 1: mark as unmovable dangerous nodes
    handy_lists = handle_stage_1(parse_tree)
    VP_List = handy_lists[0]
    NP_List = handy_lists[1]
    PP_List = handy_lists[2]
    S_List = handy_lists[3]
    CC_List = handy_lists[4]
    ADJP_List = handy_lists[5]
    ADVP_List = handy_lists[6]
    SBAR_List = handy_lists[7]
    S_Tot_List = handy_lists[8]

    # print("Processed Tree:")
    # print(parse_tree.pretty_print())

    # Stage 2: Select answer phrases and generate a set of question phrases for it
    possible_answer_phrases = []
    PP_phrases = []
    NP_phrases = []
    SBAR_phrases = []
    for node in (NP_List + PP_List + SBAR_List):
        if node.label()[:4] != "UNMV":
            if node.label() == "PP":
                PP_phrases.append(node)
            elif node.label() == "NP":
                NP_phrases.append(node)
            else:
                test = getSBARQuestion(node, parse)
                if len(test) > 0:
                    retlist += test
                else:
                    print("FAILED SBAR:")
                    node.pretty_print()
                # SBAR_phrases.append(node)
            possible_answer_phrases.append(node)
            # node.pretty_print()
        elif node.label()[-4:] == "SBAR":
            test = getSBARQuestion(node, parse)
            if len(test) > 0:
                retlist += test
            else:
                print("FAILED SBAR:")
                node.pretty_print()
    print("===============================================================================================================\n")
    # If current answer phrase is the subject: do the inversion stuff
        # Stage 3: Decompose the main verb
            #1 ROOT < (S=clause < (VP=mainvp [ < (/VB.?/=tensed !< is|was|were|am|are|has|have|had|do|does|did) | < /VB.?/=tensed !< VP]))

        # Stage 4: Invert subject/auxiliary verb
            #2 ROOT=root < (S=clause <+(/VP.*/) (VP < /(MD|VB.?)/=aux < (VP < /VB.?/=verb)))
            #3 ROOT=root < (S=clause <+(/VP.*/) (VP < (/VB.?/=copula < is|are|was|were|am !< VP)))
            # invert_subaux_tregex = ["ROOT=root < (S=clause <+(/VP.*/) (VP < /(MD|VB.?)/=aux < (VP < /VB.?/=verb)))", "ROOT=root < (S=clause <+(/VP.*/) (VP < (/VB.?/=copula < is|are|was|were|am !< VP)))"]
    # else: we're home free

    # Stage 5: Remove the answer phrase and insert one of the question phrases at the beginning of the main clause
    # Stage 6: Post-Process
    return retlist + gen_PP(PP_phrases, parse) + gen_NP(NP_phrases, parse)