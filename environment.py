import itertools
import numpy as np
from copy import deepcopy
import time
import sys
import math
from random import randint



def CreateStates_negative(LT_f, LT_s, Inv_Max, Inv_Min, OrderFastMin, OrderFastMax,OrderSlowMin,OrderSlowMax):
    Temp = []
    total_pipe = []
    total_pipe.append(range(Inv_Min, Inv_Max + 1))
    for i in range(1, LT_f + 1):
        #total_pipe.append(range(OrderFastMin,OrderFastMax + OrderSlowMax + 1))
        total_pipe.append(range(OrderSlowMin,OrderSlowMax + 1))
    for i in range(LT_f + 1, LT_s):
        total_pipe.append(range(OrderSlowMin,OrderSlowMax + 1))
    for index, i in enumerate(itertools.product(*total_pipe)):
        Temp.append(list(i))
        Temp[index].append(0)
    return np.array(Temp)


def CreateActions_negative(OrderFastMin,OrderFastMax, OrderSlowMin,OrderSlowMax):
    Temp = []
    for index, i in enumerate(itertools.product(list(range(OrderFastMin, OrderFastMax + 1)), list(range(OrderSlowMin, OrderSlowMax + 1)))):
        Temp.append(i)
    return np.array(Temp)

def to_string(state):
    s = ""
    for element in state:
        s += str(element) + "/"
    return s


def CreateDictStates(NewStates):
    dict_states = {}
    for index, state in enumerate(NewStates):
        s = to_string(state)
        dict_states[s] = index
    return dict_states


