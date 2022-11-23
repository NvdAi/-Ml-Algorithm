import numpy as np
from collections import Counter
from collections import OrderedDict
import itertools
import os


def frequency(data):
    VF = Counter(data) # VF means value and frequency
    # print(VF)
    value = list(VF.keys())
    freq = list(VF.values())
    return value, freq


def make_heap(value, freq):
    hafman_tree = []
    itr = 1
    while(len(freq)!=1):
        left = (value.pop(freq.index(min(freq))), freq.pop(freq.index(min(freq))),0,"R"+str(itr))
        right = (value.pop(freq.index(min(freq))), freq.pop(freq.index(min(freq))),1,"R"+str(itr))
        root = ("R"+str(itr), left[1] + right[1])

        hafman_tree.append(left)
        hafman_tree.append(right)
        hafman_tree.append(root)

        freq.append(root[1])
        value.append(root[0])
        itr+=1

    internal_node = []
    for i in hafman_tree:
        if len(i)<4:
            internal_node.append(i)
            hafman_tree.remove(i)

    return hafman_tree


def get_code(HF, value):
    codelist = []
    TEMP = []
    for v in value:
        start = v
        while(start != "R" + str(len(value)-1)):
            for node in HF:
                if node[0] == start:
                    TEMP.append(node[2])
                    start = node[-1]
        
        
        TEMP = list(reversed(TEMP))
        code = ''.join(map(str, TEMP))
        print(v,code)
        codelist.append(code)
        TEMP = []

    print("================================================")
    final_hufman_code = ' '.join(map(str, codelist))
    return final_hufman_code


data = ["c", "c", "c", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "a", "a", "a", "a", "c", "c", "b", "b", "b", "b", "a", "a", "a", "b", "c", "e", "e", "d", "d", "d", "d" ]

org_value, freq = frequency(data)

# org_value =  ["a","b","c","d","e","f"]             
value = org_value.copy()
# freq = [5,12,9,16,45,13]

print("org_value =", org_value, "\n", "freq = ", freq)
print("================================================")
HFT = make_heap(value, freq)
print(HFT)
print("================================================")

hufman_code = get_code(HFT,org_value)
print("the hufman_code is :", hufman_code)
