import environment
import numpy as np
from copy import deepcopy

def MC(states, P, policy):
    MC = np.zeros([len(states),len(states)])
    MC_R = np.zeros([len(states), len(states)])
    for index,action in enumerate(policy):

        for prob, next_state, reward,_ in P[index][np.argmax(action)]:
            MC[index][next_state]+= prob
            MC_R[index][next_state] += reward
    return MC, MC_R



def steady_state(states, policy, MC):
    initial = np.zeros(len(states))
    initialtemp = deepcopy(initial)
    initial[0] = 1

    while np.sum(np.abs(np.subtract(initial, initialtemp))) > 0.000001:
        #print(np.sum(np.abs(np.subtract(initial, initialtemp))))
        initialtemp = deepcopy(initial)
        initial = np.dot(initial, MC)
        print(initial)
    return initial



def cost_steady_state(steady_state, Actions,policy, MC, MC_R):
    tempcostarray=[]
    expedited = 0
    regular = 0
    for index, prob in enumerate(steady_state):
        tempcost = 0
        action = np.argmax(policy[index])
        if (action == 0):
            expedited += 0
            regular+=0
        else:
            expedited += prob * Actions[action][0]
            regular +=prob * Actions[action][1]
        for index2, prob_next in enumerate(MC[index]):
            tempcost += prob * prob_next * MC_R[index][index2]
        tempcostarray.append(tempcost)
    share_expedited = expedited/(regular+expedited)
    print(share_expedited)
    return tempcostarray,share_expedited



