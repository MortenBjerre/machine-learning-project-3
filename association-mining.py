# association mining 

from dataextraction import X, attributeNames
from similarity import binarize2
from apyori import apriori
import numpy as np

#binarized data
Xbin, attributeNamesBin = binarize2(X, attributeNames)

# from course scripts
def mat2transactions(X, labels=[]):
    T = []
    for i in range(X.shape[0]):
        l = np.nonzero(X[i, :])[0].tolist()
        if labels:
            l = [labels[i] for i in l]
        T.append(l)
    return T


# This function print the found rules and also returns a list of rules in the format:
# [(x,y), ...]
# where x -> y

def print_apriori_rules(rules):
    frules = []
    for r in rules:
        for o in r.ordered_statistics:        
            conf = o.confidence
            supp = r.support
            x = ", ".join( list( o.items_base ))
            y = ", ".join( list( o.items_add ))
            print("{%s} -> {%s}  (supp: %.3f, conf: %.3f)"%(x,y, supp, conf))
            frules.append( (x,y) )
    return frules


T = mat2transactions(Xbin,labels=attributeNamesBin)
rules = apriori(T, min_support=0.3, min_confidence=.75)
print_apriori_rules(rules)
print("")
print("Frequent item sets: \n")
rules = apriori(T, min_support=0.4, min_confidence=.0)
print_apriori_rules(rules)