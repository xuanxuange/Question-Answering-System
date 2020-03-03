import nltk
from nltk.parse import corenlp
import pattern.en

def getWhoWhat(t):
    out = []
    if t.label() != "ROOT":
        return out
    for i in range(len(t)):
        candidate = t[i]
        if candidate.label() == "S":
            if len(candidate) == 3 and candidate[0].label() == "NP" and candidate[1].label() == "VP" and candidate[2].label() == ".":
                vptext = " ".join(candidate[1].leaves())
                if vptext and vptext[-1] in ".!?":
                    vptext = vptext[:-1]
                out.append("Who or what %s?" % vptext)
    return out

def getBinarySimple(t):
    out = []
    if t.label() != "ROOT":
        return out
    for i in range(len(t)):
        candidate = t[i]
        if candidate.label() == "S":
            lemmatizer = nltk.stem.WordNetLemmatizer()
            if len(candidate) == 3 and candidate[0].label() == "NP" and candidate[1].label() == "VP" and candidate[2].label() == ".":
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
                            vbn = pattern.en.conjugate(vbn, tense=pattern.en.PAST)
                        except:
                            continue
                    vptext = " ".join(candidate[1].leaves())
                    if vptext and vptext[-1] in ".!?":
                        vptext = vptext[:-1]
                    vpptext = " ".join([vbn]+candidate[1].leaves()[1:])
                    if vpptext and vpptext[-1] in ".!?":
                        vpptext = vpptext[:-1]
                    prefdict = {"VBG": "Has", "VBN": "Has/Have", "VBD":"Had"}
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
    if t.label() != "ROOT":
        return out
    for i in range(len(t)):
        candidate = t[i]
        if candidate.label() == "S":
            lemmatizer = nltk.stem.WordNetLemmatizer()
            if len(candidate) == 3 and candidate[0].label() == "NP" and candidate[1].label() == "VP" and candidate[2].label() == ".":
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
