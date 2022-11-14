import numpy as np
from collections import Counter
from collections import OrderedDict




def frequency(data):
    VF = Counter(data) # VF means value and frequency
    print(VF)
    value = list(VF.keys())
    freq = list(VF.values())
    return value, freq



def make_heap(value, freq):
    print(value, freq)
    # freq = sorted(freq)
    # print(freq)
    hafman_tree = []
    itr = 1
    while(len(freq)!=1):
        # freq.sort()
        left = (value.pop(freq.index(min(freq))), freq.pop(freq.index(min(freq))),0,itr)
        right = (value.pop(freq.index(min(freq))), freq.pop(freq.index(min(freq))),1,itr)
        root = (itr, left[1] + right[1])

        hafman_tree.append(left)
        hafman_tree.append(right)
        hafman_tree.append(root)

        freq.append(root[1])
        value.append(root[0])

        itr+=1

        print("left",left,"right",right,"rot",root,"freq",freq,"value",value)
        print("================================================")
        print(hafman_tree)
        print("================================================")

    # print(list(OrderedDict.fromkeys(hafman_tree)))
    # for i in hafman_tree:
    #     if len(i)==3:
    #         hafman_tree.remove(i)
    print(hafman_tree)
       



data = [4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 7, 7, 7, 7, 4, 4, 2, 2, 2, 2, 7, 7, 7, 2, 4, 8, 8, 5, 5, 5, 5 ]

value, freq = frequency(data)

make_heap(value, freq)





