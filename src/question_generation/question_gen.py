import nltk
from nltk.parse import corenlp
from queue import Queue
import pattern.en

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

        if curr_node.label() == "NP":
            contains_pp = False
            for i in range(len(curr_node)):
                if curr_node[i].label() == "PP" or curr_node[i].label() == "JJ":
                    contains_pp = True
            if contains_pp:
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
