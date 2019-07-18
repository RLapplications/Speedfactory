import environment
import time
import argparse
import MarkovChain
import numpy as np
import VI
import threading
import multiprocessing
from joblib import Parallel, delayed
from copy import deepcopy

def main(args,LT_tested,  k_tested,distribution,identifier,demand_values):
    optimal_costs = []
    optimal_expedited = []
    with open('Results_DP.csv', 'a') as f:
        f.write('optimal cost;expedited;regular;l_e;l_r;k;u;m;h;b;lower bound local;upper bound local;lower bound offshore;upper bound offshore;lower bound inventory;'
                'upper bound inventory;')
        for index in demand_values:
            f.write('P(d=%i);'%index)
        f.write('\n')

    for LT_s in LT_tested:
        for k in k_tested:
            print('STARTED',identifier,LT_s,k)
            args.LT_s = LT_s
            start = time.time()
            States = environment.CreateStates_negative(args.LT_f,args.LT_s,args.Inv_Max,args.Inv_Min,args.OrderFastMin,args.OrderFastMax,args.OrderSlowMin,args.OrderSlowMax)
            print('states created', time.time() - start)
            Actions = environment.CreateActions_negative(args.OrderFastMin,args.OrderFastMax,args.OrderSlowMin,args.OrderSlowMax)
            dict_states = environment.CreateDictStates(States)
            optimal_cost,expedited_share = cost_per_period(States,Actions,dict_states, args,k,args.u,args.m,distribution,identifier,demand_values)
            print(LT_s,k,optimal_cost)
            optimal_costs.append(optimal_cost)
            optimal_expedited.append(expedited_share)
            with open('Results_DP.csv', 'a') as f:
                f.write(str(optimal_cost) + ';' +str(expedited_share) + ';'+str(1-expedited_share) + ';'+ str(args.LT_f)+ ';' + str(args.LT_s) + ';' + str(k) + ';' + str(args.u) +';' +
                        str(args.m)+';' + str(args.h)+';'+str(args.b)+';'+str(args.OrderFastMin)+';'+str(args.OrderFastMax)+';'
                        + str(args.OrderSlowMin) + ';'+ str(args.OrderSlowMax) + ';'+str(args.Inv_Min)+';'+str(args.Inv_Max)+';')
                for item in distribution:
                    f.write(str(item)+';')
                f.write('\n')
    return


def TransitionProbs(States, Actions, Demand_Max,LT_s,LT_f,h,b,C_s,C_f,Inv_Max,Inv_Min, cap_fast, cap_slow,dict_states,k,u,m,distribution,demand_values):
    T = []
    for index, state in enumerate(States):
        Temp1 = []
        for index2, action in enumerate(Actions):
            Temp2 = []
            for index3,demand in enumerate(demand_values):
                reward, s1, done = SF_transition(state, action, demand, LT_s, LT_f, h, b, C_s, C_f,Inv_Max,
                                                          Inv_Min, cap_fast, cap_slow,k,u,m)
                Tuple = (distribution[index3], dict_states[environment.to_string(s1)], reward, done)
                Temp2.append(Tuple)
            Temp1.append(Temp2)
        T.append(Temp1)
    return T


def cost_per_period(States,Actions,dict_states, args,k,u,m,distribution,identifier,demand_values):
    start = time.time()
    P = TransitionProbs(States, Actions, args.Demand_Max, args.LT_s, args.LT_f, args.h, args.b, args.C_s, args.C_f,
                           args.Inv_Max, args.Inv_Min, args.cap_fast, args.cap_slow, dict_states,k,u,m,distribution,
     demand_values)

    env = VI.VI_env(States, Actions, P)

    print('environment created', time.time() - start)
    #policy, v = PI.policy_improvement(env, args.discount_factor)
    policy, v = VI.value_iteration(env,theta=0.000001, discount_factor=args.discount_factor)

    np.save('v_%s_%s_%s.npy'%(identifier,args.LT_s,k),v)
    optimal_policy = []


    MC, MC_R = MarkovChain.MC(States, P, policy)
    steady_state = MarkovChain.steady_state(States, policy, MC)
    optimal_cost_array,share_expedited = MarkovChain.cost_steady_state(steady_state, Actions,policy, MC, MC_R)
    optimal_cost = np.sum(optimal_cost_array)



    with open('optimal_policy-l_e%i-l_r%i-k%i-Distribution %s.csv'%(args.LT_f,args.LT_s,k,identifier),'w') as f:
        f.write('OPTIMAL COST;'+str(optimal_cost)+'\n')
        f.write('Share expedited;' + str(share_expedited) + '\n')
        f.write('Share regular;' + str(1-share_expedited) + '\n\n')


        f.write('PARAMETERS USED:\n')
        f.write('Demand;Prob\n')
        for index,item in enumerate(distribution):
            f.write(str(demand_values[index])+';'+str(item)+'\n')
        f.write('\n')

        f.write('l_r;')
        f.write(str(args.LT_s)+'\n')

        f.write('h;')
        f.write(str(args.h)+'\n')

        f.write('b;')
        f.write(str(args.b)+'\n')

        f.write('k;')
        f.write(str(k)+'\n')

        f.write('u;')
        f.write(str(u)+'\n')

        f.write('m;')
        f.write(str(m)+'\n')

        f.write('c_r;')
        f.write(str(args.C_s)+'\n')

        f.write('\n')
        for index,__ in enumerate(States[0]):
            f.write('State' + ';')
        f.write('optimal local'+';'+'optimal offshore')
        f.write(';prob state;')
        f.write('cost state;')
        f.write('weighted cost state;')
        f.write('\n')

        for index,state in enumerate(policy):
            for index2,action in enumerate(state):
                if(action ==1):
                    #print(States[index],Actions[index2])
                    optimal_policy.append([States[index],Actions[index2]])
                    for item in States[index]:
                        f.write(str(item) + ';')
                    for item in Actions[index2]:
                        f.write(str(item)+';')
                    #print(steady_state,steady_state[0])
                    f.write(str(steady_state[index])+';')
                    if(steady_state[index]>0):
                        f.write(str(optimal_cost_array[index]/steady_state[index]) + ';')
                    else:
                        f.write(str(0)+';')
                    f.write(str(optimal_cost_array[index]))
                    f.write('\n')
    return optimal_cost,share_expedited


def SF_transition(s, a, demand, LT_s, LT_f, h, b, C_s, C_f, Inv_Max, Inv_Min, cap_fast, cap_slow, k, u, m):
    done = False
    s1 = deepcopy(s)
    reward = 0
    s1[LT_f] += a[0]
    s1[LT_s] += a[1]
    s1[0] += - demand
    reward += k * u + u * m * max(a[0] - k, 0)  + a[1] * C_s
    if s1[0] >= 0:
        reward += s1[0] * h
    else:
        reward += -s1[0] * b
    s1[0] += s1[1]
    for i in range(1, LT_s):
        s1[i] = s1[i + 1]
    s1[LT_s] = 0
    if (s1[0] > Inv_Max):
        s1[0] = Inv_Max
        done = True
    if s1[0] < Inv_Min:
        s1[0] = Inv_Min
        done = True
    return reward, s1, done


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-Demand_Max', '--Demand_Max', default=10, type=float,
                        help="Demand_Max. Default = 4",
                        dest="Demand_Max")
    parser.add_argument('--OrderFastMax', default=10, type=int,
                        help="OrderFast. Default = 5",
                        dest="OrderFastMax")
    parser.add_argument('--OrderFastMin', default=-10, type=int,
                        help="OrderFast. Default = 5",
                        dest="OrderFastMin")
    parser.add_argument('--OrderSlowMax', default=10, type=int,
                        help="OrderSlowMax. Default = 5",
                        dest="OrderSlowMax")
    parser.add_argument('--OrderSlowMin', default=-10, type=int,
                        help="OrderSlowMin. Default = 5",
                        dest="OrderSlowMin")
    parser.add_argument('--OrderSlow', default=10, type=int, help="OrderSlow. Default = 5", dest="OrderSlow")
    parser.add_argument('--LT_s', default=1, type=float, help="LT_s. Default = 1", dest="LT_s")
    parser.add_argument('--LT_f', default=1, type=float, help="LT_f. Default = 0",
                        dest="LT_f")
    parser.add_argument('--Inv_Max', default=30, type=float,
                        help="Inv_Max. Default = 10",
                        dest="Inv_Max")
    parser.add_argument('--Inv_Min', default=-30, type=float,
                        help="Inv_Min. Default = -10",
                        dest="Inv_Min")
    parser.add_argument('--cap_slow', default=1, type=float,
                        help="cap_slow. Default = 1",
                        dest="cap_slow")
    parser.add_argument('--cap_fast', default=1, type=float,
                        help="cap_fast. Default = 1",
                        dest="cap_fast")
    parser.add_argument('--C_s', default=3.8, type=float,
                        help="C_s. Default = 3.8",
                        dest="C_s")
    parser.add_argument('--C_f', default=150, type=float,
                        help="C_f. Default = 150",
                        dest="C_f")
    parser.add_argument('--h', default=1 , type=float,
                        help="h. Default = 1",
                        dest="h")
    parser.add_argument('--b', default= 9, type=str,
                        help="b. Default = 9",
                        dest="b")
    parser.add_argument('--discount_factor', default= 1.0, type=float,
                        help="discount_factor. Default = 0.99",
                        dest="discount_factor")

    parser.add_argument('--u', default= 4, type=float,
                        help="u. Default = 4",
                        dest="u")

    parser.add_argument('--m', default= 1.5, type=float,
                        help="m. Default = 1.5",
                        dest="m")

    parser.add_argument('--uniform', default= False, type=bool,
                        help="uniform. Default = True",
                        dest="uniform")

    args = parser.parse_args()

    #HOW To USE CODE:
    # 1. Set bounds on Maximum Demand: args.Demand_Max
    # 2. Set bounds on Inventory: args.Inv_Max and Inv_Min
    # 3. Set bounds on fast (expedite) orders: args.OrderFastMin can be negative
    # 4. Set bounds on slow (regular) orders: args.OrderSlow (this is an upper bound, lower bound is always 0)
    # 5. Specify distribution (needs to be of size args.Demand_Max + 1)
    # 6. Specify list of lead times you want to test
    # 7. Specify list of k's you want to test
    # 8. Specify name of the experiment for output files (identifier)
    # 9. Run main(args, LT_tested, k_tested, args.u, args.m,distribution)

    if args.uniform == True:
        #EXPERIMENT 1 - UNIFORM[10:20]
        ###########################
        args.Demand_Max = 20
        args.Inv_Max = 100
        args.Inv_Min = -20
        args.OrderFastMin = -5
        args.OrderFastMax = 20
        args.OrderSlowMin = 0
        args.OrderSlowMax = 20
        distribution = [1/11, 1/11, 1/11, 1/11, 1/11,1/11, 1/11, 1/11, 1/11, 1/11,1/11]
        demand_values = [10,11,12,13,14,15,16,17,18,19,20]
        LT_tested = [1,2,3,4,5]
        k_tested = [1,2]
    #
        identifier = "Uniform[10-20]"
        main(args, LT_tested,  k_tested, distribution,identifier,demand_values)

    else:
        # EXPERIMENT 3 - Binomial
        #############################
        args.Demand_Max = 20
        args.Inv_Max = 100
        args.Inv_Min = -20
        args.OrderFastMin = -5
        args.OrderFastMax = 20

        args.OrderSlowMin = 0
        args.OrderSlowMax = 20

        #distribution = [0.000104858,0.00157286,0.0106168,0.0424673,0.111477,0.200658,0.250823,0.214991,0.120932,0.0403108,0.00604638199999996]
        distribution = [9.09495 * 10 ** -13,  5.45697 * 10 ** -11,  1.55524 * 10 ** -9,
                        2.79942 * 10 ** -8,3.56927 * 10 ** -7, 3.4265 * 10 ** -6,
                        0.0000256987, 0.000154192, 0.000751688,0.00300675,0.00992228, 0.0270608,0.0608867, 0.112406,0.168609
                        , 0.202331, 0.189685, 0.133896,0.0669478, 0.0211414, 0.00317121]
        #print(np.sum(distribution))
        demand_values = [0,1,2,3,4,5,6,7,8,9,10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        LT_tested = [1,2,3,4,5]
        k_tested = [1,2]#,11,12,13,14,15,16,17,18,19,20]

        identifier = "Binomial[n=20, p=0.75]"

        main(args, LT_tested,  k_tested, distribution,identifier,demand_values)
