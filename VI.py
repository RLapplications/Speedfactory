import numpy as np
import environment
import MarkovChain
import time
from random import randint

class VI_env():
    def __init__(self, states, actions, P):
        self.nS = len(states)
        self.nA = len(actions)
        self.P = P


def value_iteration(env, theta=0.000001, discount_factor=1.0):
    """
    Value Iteration Algorithm.

    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """

    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.nS

        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A

    V = np.zeros(env.nS)
    deduct = int(env.nS/2)
    old_J = 0

    while True:
        # Stopping condition
        delta = 0
        # Update each state...
        old_total = np.sum(V)
        for s in range(env.nS):

            # Do a one-step lookahead to find the best action
            A = one_step_lookahead(s, V)
            #####################################################################################
            #IMPORTANT                                                                          #
            #Modified towards relative value iteration as we are working with average costs     #
            #We always deduct all value functions by the value of the same state                #
            #Specify below which one to deduct.                                                 #
            #####################################################################################
            best_action_value = np.min(A)# - np.min(one_step_lookahead(deduct,V))
            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(best_action_value - V[s]))
            # Update the value function. Ref: Sutton book eq. 4.10.
            V[s] = best_action_value #- np.min(one_step_lookahead(0,V))
            # Check if we can stop
        new_J = np.sum(V)- old_total
        diff = np.abs(new_J - old_J)
        print('DIFF',diff,new_J,old_J)
        old_J = new_J

        print(delta)
        if delta < theta:
            break

        #if diff < theta:
        #    break
    # Create a deterministic policy using the optimal value function
    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        # One step lookahead to find the best action for this state
        A = one_step_lookahead(s, V)
        best_action = np.argmin(A)
        # Always take the best action
        policy[s, best_action] = 1.0
    return policy, V

