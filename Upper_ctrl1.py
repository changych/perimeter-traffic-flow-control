# -*- coding: UTF-8 -*-
import os, sys
import xml.etree.ElementTree as etree
import xml.dom.minidom as doc
import math
import numpy as np
import random
import pickle
import cPickle
from collections import defaultdict
from scipy.stats import norm
from scipy.stats import poisson
import pdb
from time import time

zone = ['R1', 'R2', 'R3']
BetRgs = ['R1-R2', 'R1-R3', 'R2-R1', 'R2-R3', 'R3-R1', 'R3-R2']
NJam0 = [5683,3839,6242]
NJam = [v*1 for v in NJam0]

state_num = 3
action_num = 2
reward_ratio0 = 0.2
reward_ratio1 = 1

class Update_policy():
    def __init__(self, T, Tu, ActionRange, N_current, D_current, LoopsOutputNum,Occupy,randvalue):

        self.UpdateCycle = Tu
        self.CycleLen = T
        self.LoopsOutputNum = LoopsOutputNum

        # the output flow rate for each sub-region
        # predict the mean value and variance
        self.Min_P = np.zeros(3)
        self.Max_P = np.zeros(3)
        self.DemandIn = defaultdict(dict)
        self.Min_G = np.zeros(3)
        self.Max_G = np.zeros(3)
        self.OutFlow = defaultdict(dict)

        self.StateRange = np.zeros((3, 2))
        self.region_state_num = state_num
        self.network_state_num = int(math.pow(self.region_state_num, 3))
        # state=[max_R1, min_R1]
        self.state = defaultdict(lambda: np.zeros(self.region_state_num + 1, dtype='float'))

        self.ActionRange = ActionRange
        self.region_action_num = action_num### change the number of action
        self.network_action_num = int(math.pow(self.region_action_num, 6))
        # action = [max_r1,min_r1]
        self.action = defaultdict(lambda: np.zeros(self.region_action_num + 1, dtype='float'))
        self.RegionOccupy = defaultdict(lambda:0)

        for i in xrange(3):

            #### get the region occupy
            zone_id = 'R'+str(i+1)
            self.RegionOccupy[zone_id]=Occupy[zone_id]

            ### Poisson Distribution-----the external demand of each sub-region
            ArrivalRate = D_current[i]  # *self.UpdateCycle
            self.Min_P[i] = int(poisson.ppf(0.01, ArrivalRate))
            self.Max_P[i] = int(poisson.ppf(0.99, ArrivalRate))

            Qin = defaultdict(lambda: 0)
            # print self.Min_P[i],self.Max_P[i]
            for j in range(int(self.Min_P[i]), int(self.Max_P[i])):
                Qin[j] = poisson.pmf(j, ArrivalRate)
            maxkey = max(Qin, key=Qin.get)
            Qin[maxkey] = Qin[maxkey] + 1.0 - np.sum(Qin.values())
            self.DemandIn[i] = Qin
            # print np.sum(Qin.values())

            ## Gaussian distribution----the out flow of each -region

            # MFD = open('./MFD_' + zone[i] + '.sav', 'r')
            # model = pickle.load(MFD)
            # x = np.array([N_current[i]])
            # q_pre, q_var = model.predict(x.reshape((len(x), 1)))
            # q_sigma = np.sqrt(q_var)
            q_pre = randvalue[i,0]
            q_sigma = randvalue[i,1]
            ### linear relationship
            # LR_model = pickle.load(open('LR_'+zone[i]+'.sav','rb'))
            # qout_pre = LR_model.coef_[0]*q_pre+LR_model.intercept_[0]
            # qout_sigma = LR_model.coef_[0]*q_sigma
            ###
            qout_pre = q_pre * self.LoopsOutputNum[zone[i]]
            qout_sigma = q_sigma * self.LoopsOutputNum[zone[i]]

            Qout_mean = qout_pre * self.CycleLen / 3600.0  # *self.UpdateCycle
            Qout_std = qout_sigma * self.CycleLen / 3600.0  # *self.UpdateCycle

            # print 'Qout_mean:',Qout_mean,'Qout_std:',Qout_std

            self.Min_G[i] = int(norm.ppf(0.05, Qout_mean, Qout_std))
            self.Max_G[i] = int(norm.ppf(0.95, Qout_mean, Qout_std))

            # MaxOutputBetRegion = np.sum(self.ActionRange[2 * i:2 * (i + 1), 1])
            # if MaxOutputBetRegion > self.Max_G[i]:
            #     diffV = MaxOutputBetRegion - self.Max_G[i]
            #     self.ActionRange[2 * i:2 * (i + 1), 1] = self.ActionRange[2 * i:2 * (i + 1), 1] - np.ones(
            #         2) * diffV / 2.0

            Qout = defaultdict(lambda: 0)
            # print self.Min_G[i],self.Max_G[i]
            for j in range(int(self.Min_G[i]), int(self.Max_G[i])+1):
                Qout[j] = norm.cdf(j + 1, Qout_mean, Qout_std) - norm.cdf(j, Qout_mean, Qout_std) # 1/(self.Max_G[i]-self.Min_G[i]+1)
            Qout[int(self.Max_G[i])] = Qout[int(self.Max_G[i])] + 1.0 - np.sum(Qout.values())
            # if Qout == {}:
            #     Qout[0] = 1.0
            # else:
            #     maxkey = max(Qout, key=Qout.get)
            #     Qout[maxkey] = Qout[maxkey] + 1.0 - np.sum(Qout.values())
            self.OutFlow[i] = Qout
        # print np.sum(Qout.values())

        ### state_range
        print(N_current)
        index = np.array(
            [[2, 4], [0, 5], [1, 3]])  ## idex for BetRgs = ['R1-R2','R1-R3','R2-R1','R2-R3','R3-R1','R3-R2']
        Tx = 3
        for i in xrange(3):
            self.StateRange[i, 0] = max(0,N_current[i] - self.ActionRange[2 * i, 1]*Tx - self.ActionRange[2 * i + 1, 1]*Tx + Tx*(self.Min_P[i] - self.Max_G[i])) # *self.UpdateCycle
            self.StateRange[i, 1] = min(NJam0[i],N_current[i] + (self.ActionRange[index[i, 0], 1] + self.ActionRange[index[i, 1], 1] + self.Max_P[i] -self.Min_G[i])*Tx) # *self.UpdateCycle
            if self.StateRange[i,0]>self.StateRange[i,1]:
                self.StateRange[i,0]=self.StateRange[i,1] - self.ActionRange[2 * i, 1]*Tx - self.ActionRange[2 * i + 1, 1]*Tx + Tx*(self.Min_P[i] - self.Max_G[i])


    def Split_state(self):
        for i in xrange(3):
            min_state = self.StateRange[i, 0]
            max_state = self.StateRange[i, 1]
            interval = 1.0 * (max_state - min_state) / self.region_state_num
            self.state[zone[i]][0] = min_state
            for j in xrange(1, self.region_state_num):
                self.state[zone[i]][j] = min_state + j * interval
            self.state[zone[i]][-1] = max_state
        # print(self.state)

    def Split_action(self):
        action_interval = np.zeros(6)
        for i in xrange(6):
            min_action = self.ActionRange[i, 0]
            max_action = self.ActionRange[i, 1]
            interval = round(1.0 * (max_action - min_action) / self.region_action_num)
            action_interval[i] = interval
            self.action[BetRgs[i]][0] = min_action
            for j in xrange(1, self.region_action_num):
                self.action[BetRgs[i]][j] = min_action + j * interval
            self.action[BetRgs[i]][-1] = max_action

        a = defaultdict(list)
        for i in xrange(self.region_action_num):
            a1 = np.sum(self.action['R1-R2'][i:i + 2]) / 2.0
            for j in xrange(self.region_action_num):
                a2 = np.sum(self.action['R1-R3'][j:j + 2]) / 2.0
                for k in xrange(self.region_action_num):
                    a3 = np.sum(self.action['R2-R1'][k:k + 2]) / 2.0
                    for l in xrange(self.region_action_num):
                        a4 = np.sum(self.action['R2-R3'][l:l + 2]) / 2.0
                        for m in xrange(self.region_action_num):
                            a5 = np.sum(self.action['R3-R1'][m:m + 2]) / 2.0
                            for n in xrange(self.region_action_num):
                                a6 = np.sum(self.action['R3-R2'][n:n + 2]) / 2.0
                                index = i * math.pow(self.region_action_num, 5) + j * math.pow(
                                    self.region_action_num, 4) + k * math.pow(self.region_action_num,
                                                                              3) + l * math.pow(
                                    self.region_action_num, 2) + m * self.region_action_num + n
                                a[index] = np.array([a1, a2, a3, a4, a5, a6])
        self.ActionSpace = a
        return action_interval

    def Split_action1(self):
        action_interval = np.zeros(6)
        for i in xrange(6):
            min_action = self.ActionRange[i, 0]
            max_action = self.ActionRange[i, 1]
            interval = round(1.0 * (max_action - min_action) / self.region_action_num)
            action_interval[i] = interval
            self.action[BetRgs[i]][0] = min_action
            for j in xrange(1, self.region_action_num):
                self.action[BetRgs[i]][j] = min_action + j * interval
            self.action[BetRgs[i]][-1] = max_action

        a = defaultdict(list)
        for i in xrange(self.region_action_num):
            a1 = self.action['R1-R2'][i+1]
            for j in xrange(self.region_action_num):
                a2 = self.action['R1-R3'][j + 1]
                for k in xrange(self.region_action_num):
                    a3 = self.action['R2-R1'][k + 1]
                    for l in xrange(self.region_action_num):
                        a4 = self.action['R2-R3'][l + 1]
                        for m in xrange(self.region_action_num):
                            a5 = self.action['R3-R1'][m + 1]
                            for n in xrange(self.region_action_num):
                                a6 = self.action['R3-R2'][n + 1]
                                index = i * math.pow(self.region_action_num, 5) + j * math.pow(
                                    self.region_action_num, 4) + k * math.pow(self.region_action_num,
                                                                              3) + l * math.pow(
                                    self.region_action_num, 2) + m * self.region_action_num + n
                                a[index] = np.array([a1, a2, a3, a4, a5, a6])
        self.ActionSpace = a
        return action_interval

    # def STPM_region(self, RegionID, InsideConnection):
    #
    #     zone = ['R1', 'R2', 'R3']
    #     # Smin = int(self.Min_P[RegionID] - self.Max_G[RegionID])
    #     # Smax = int(self.Max_P[RegionID] - self.Min_G[RegionID])
    #     # Slist = range(Smin, Smax + 1)
    #     prob = defaultdict(lambda: 0)
    #     for j in range(int(self.Min_P[RegionID]), int(self.Max_P[RegionID]) + 1):
    #         for k in range(int(self.Min_G[RegionID]), int(self.Max_G[RegionID]) + 1):
    #             i = j - k
    #             prob[i] += self.DemandIn[RegionID][j] * self.OutFlow[RegionID][k]
    #     error = (1 - np.sum(prob.values())) / len(prob)
    #     Slist = prob.keys()
    #     for i in Slist:
    #         prob[i] += error
    #     prob[i] += 1 - np.sum(prob.values())
    #     # print np.sum(prob.values())
    #
    #     ## calculate STPM under different actions
    #     STPModel = defaultdict(lambda: np.zeros((self.region_state_num, self.region_state_num), dtype=float))
    #     Reward = defaultdict(lambda: np.zeros(self.region_state_num, dtype=float))
    #     for a in self.ActionSpace.keys():
    #         action = self.ActionSpace[a]
    #         RegionIn = 0
    #         RegionOut = 0
    #         for l in xrange(len(InsideConnection)):
    #             if InsideConnection[l] == 1:
    #                 RegionIn += action[l]
    #             elif InsideConnection[l] == -1:
    #                 RegionOut += action[l]
    #         DifferInside = RegionIn - RegionOut
    #         # self.Min_G[RegionID] = self.Min_G[RegionID]-RegionOut
    #         # self.Max_G[RegionID] = self.Max_G[RegionID]-RegionOut
    #         state = self.state[zone[RegionID]]
    #         for i in xrange(len(state) - 1):
    #             s = [state[i], state[i + 1]]
    #             Dtemp = range(int(s[0] + DifferInside), int(s[1] + DifferInside))
    #             # print len(Dtemp)
    #             UniformProb = 1.0 / (len(Dtemp) + 0.001)
    #             P_s_s_next = np.zeros(3)
    #             Ptemp = np.zeros(3)
    #             for k in Dtemp:
    #                 for m in Slist:
    #                     if m + k < state[1]:
    #                         Ptemp[0] += prob[m]
    #                     elif m + k >= state[1] and m + k < state[2]:
    #                         Ptemp[1] += prob[m]
    #                     elif m + k >= state[2]:
    #                         Ptemp[2] += prob[m]
    #                 for j in range(3):
    #                     P_s_s_next[j] += Ptemp[j] * UniformProb
    #                 Ptemp = np.zeros(3)
    #             STPModel[a][i, :] = P_s_s_next
    #             for j in range(3):
    #                 Reward[a][j] += P_s_s_next[j] * (state[j] + state[j + 1]) / (2.0*self.RegionOccupy[zone[RegionID]])
    #     return STPModel, Reward

    def STPM_region_revised(self, RegionID, InsideConnection):

        zone = ['R1', 'R2', 'R3']
        # Smin = int(self.Min_P[RegionID] - self.Max_G[RegionID])
        # Smax = int(self.Max_P[RegionID] - self.Min_G[RegionID])
        # Slist = range(Smin, Smax + 1)
        prob = defaultdict(lambda: 0)
        for j in range(int(self.Min_P[RegionID]), int(self.Max_P[RegionID]) + 1):
            for k in range(int(self.Min_G[RegionID]), int(self.Max_G[RegionID]) + 1):
                i = j - k
                prob[i] += self.DemandIn[RegionID][j] * self.OutFlow[RegionID][k]
        error = (1 - np.sum(prob.values())) / len(prob)

        for i in  prob.keys():
            prob[i] += error
        prob[i] += 1 - np.sum(prob.values())
        # print np.sum(prob.values())

        ## calculate STPM under different actions
        STPModel = defaultdict(lambda: np.zeros((self.region_state_num, self.region_state_num), dtype=float))
        Reward = defaultdict(lambda: np.zeros(self.region_state_num, dtype=float))
        for a in self.ActionSpace.keys():
            action = self.ActionSpace[a]
            RegionIn = 0
            RegionOut = 0
            for l in xrange(len(InsideConnection)):
                if InsideConnection[l] == 1:
                    RegionIn += action[l]
                elif InsideConnection[l] == -1:
                    RegionOut += action[l]
            DifferInside = RegionIn - RegionOut
            # self.Min_G[RegionID] = self.Min_G[RegionID]-RegionOut
            # self.Max_G[RegionID] = self.Max_G[RegionID]-RegionOut
            state = self.state[zone[RegionID]]

            for i in xrange(len(state) - 1):
                s = [state[i], state[i + 1]]
                Dtemp = range(int(s[0] + DifferInside), int(s[1] + DifferInside))
                # print len(Dtemp)
                UniformProb = 1.0 / (len(Dtemp) + 0.001)
                P_s_s_next = np.zeros(self.region_state_num)
                Ptemp = np.zeros(self.region_state_num)

                for m in prob.keys():
                    temp_range = [ v+m for v in Dtemp]
                    index_temp = np.zeros(self.region_state_num+1)
                    # temp_rangel0 = [x for x in temp_range if x > 0]
                    # lzero = len(temp_rangel0)

                    if int(state[0]) in temp_range:
                        index_temp[0]=0
                    else:
                        if int(state[0])<temp_range[0]:
                            index_temp[0]=0
                        elif int(state[0])>temp_range[-1]:
                            index_temp[0]=len(temp_range)-1

                    for h in range(self.region_state_num-1):
                        if int(state[h+1]) in temp_range:
                            index_temp[h + 1] = temp_range.index(int(state[h + 1]))
                        else:
                            if int(state[h + 1]) < temp_range[0] and index_temp[h] == 0:
                                index_temp[h + 1] = 0
                            elif int(state[h+1])>temp_range[-1] and index_temp[h]!=len(temp_range)-1:
                                index_temp[h+1]=len(temp_range)-1
                            elif int(state[h+1])>temp_range[-1] and index_temp[h]==len(temp_range)-1:
                                index_temp[h + 1] = len(temp_range)-1

                    index_temp[-1]=len(temp_range)-1

                    for h in range(self.region_state_num):
                        if index_temp[h+1]>index_temp[h]:
                            Ptemp[h] += prob[m]*len(Dtemp[int(index_temp[h]):int(index_temp[h+1])+1])
                        elif index_temp[h+1]==index_temp[h]:
                            Ptemp[h] += 0
                    if index_temp[0]==len(temp_range)-1:
                        Ptemp[0]+=prob[m]*len(Dtemp)

                for j in range(self.region_state_num):
                    P_s_s_next[j] += Ptemp[j] * UniformProb

                # if np.sum(P_s_s_next[0:self.region_state_num-1])==0:
                #     P_s_s_next[-1]=1

                 # Ptemp[0] += sum(v for k, v in prob.items() if k <label)
                STPModel[a][i, :] = P_s_s_next
                for j in range(self.region_state_num):
                    Reward[a][j] += P_s_s_next[j] * (state[j] + state[j + 1]) / (
                                2.0 * self.RegionOccupy[zone[RegionID]])
        return STPModel, Reward

    def CombineStates(self, STPM_R1, STPM_R2, STPM_R3, Reward_R1, Reward_R2, Reward_R3):
        STPM = np.zeros((self.network_state_num, self.network_state_num))
        Reward = np.zeros(self.network_state_num)
        state_list = []
        for i in xrange(self.region_state_num):
            for j in xrange(self.region_state_num):
                for k in xrange(self.region_state_num):
                    state_list.append([i, j, k])

        for i in xrange(len(state_list)):
            r = state_list[i]
            Reward[i] = reward_ratio0*(Reward_R1[r[0]]+Reward_R2[r[1]]+Reward_R3[r[2]])+reward_ratio1*(np.abs(Reward_R1[r[0]]-Reward_R2[r[1]]) +np.abs(Reward_R1[r[0]]-Reward_R3[r[2]])+ np.abs(Reward_R2[r[1]]-Reward_R3[r[2]]))
            Reward[i] = -Reward[i]
            # print(Reward_R1[r[0]]+Reward_R2[r[1]]+Reward_R3[r[2]])
            # print(np.abs(Reward_R1[r[0]]-Reward_R2[r[1]]) +np.abs(Reward_R1[r[0]]-Reward_R3[r[2]])+ np.abs(Reward_R2[r[1]]-Reward_R3[r[2]]))
            for j in xrange(len(state_list)):
                c = state_list[j]
                STPM[i, j] = STPM_R1[r[0], c[0]] * STPM_R2[r[1], c[1]] * STPM_R3[r[2], c[2]]

        return STPM, Reward

    def get_STPMatrix_Reward(self, policy):
        STPMatrix = np.zeros((self.network_state_num, self.network_state_num))
        RewardVector = np.zeros(self.network_state_num)
        for i in xrange(self.network_state_num):
            action = policy[i]
            STPMatrix[i, :] = self.STPModel_Network[action][i, :]
            RewardVector[i] = self.Reward_Network[action][i]

        return STPMatrix, RewardVector

    def Policy_Optimize(self):
        ##### algorithm
        bate = 1  ##discount
        Initial_policy = [random.randint(0, self.network_action_num - 1) for _ in range(self.network_state_num)]
        Opt_policy = Initial_policy

        policy = [1] * self.network_state_num

        l = 0

        while (cmp(Opt_policy, policy) != 0):
            if l > 200:
                break
            l += 1

            policy = Opt_policy[:]
            policy_temp = Opt_policy[:]

            STPMatrix, RewardVector = self.get_STPMatrix_Reward(Opt_policy)

            ## get potential by solving the passion equation
            temp = np.linalg.matrix_power(STPMatrix, 50)
            Pai0 = temp[0, :]
            eta0 = np.dot(Pai0, RewardVector)
            I = np.diag(np.ones(self.network_state_num))
            e = bate * np.ones(self.network_state_num)
            temp1 = np.linalg.inv(I - bate * STPMatrix + np.outer(e, Pai0))
            g0 = temp1.dot(RewardVector)
            # g0=inv(I-STPMatrix+e*pai0)*reward;
            for i in xrange(self.network_state_num):
                min_value = 0
                for j in xrange(self.network_action_num):
                    policy_temp[i] = j
                    P_temp, F_temp = self.get_STPMatrix_Reward(policy_temp)
                    current_value = bate * np.dot(P_temp[i, :] - STPMatrix[i, :], g0) + (F_temp[i] - RewardVector[i])
                    # current_value = (P_temp(i,:)-P0(i,:))*g0+(f_temp(i)-f0(i));
                    if current_value < min_value:
                        min_value = current_value
                        Opt_policy[i] = j
                # break
                policy_temp[i] = Opt_policy[i]

        return Opt_policy

    def STPM_network(self):


        self.Split_state()
        # pdb.set_trace()
        action_interval = self.Split_action()
        # pdb.set_trace()


        ## get the state transition probability model for subregion under different actions
        STPModel_Region = defaultdict()
        Reward_Region = defaultdict()
        ConnectionMatrix = np.zeros((3, 6))
        ConnectionMatrix[0, :] = np.array([-1, -1, 1, 0, 1, 0])
        ConnectionMatrix[1, :] = np.array([1, 0, -1, -1, 0, 1])
        ConnectionMatrix[2, :] = np.array([0, 1, 0, 1, -1, -1])

        #### calculate the state transition probability matrix of each subregion
        for i in xrange(3):  ##zone
            STPModel_Region[zone[i]], Reward_Region[zone[i]] = self.STPM_region_revised(i, ConnectionMatrix[i, :])



        ### combine the subregion STPM to get the network STPM
        STPModel_Network = defaultdict(lambda: np.zeros((self.network_state_num, self.network_state_num), dtype=float))
        Reward_Network = defaultdict(lambda: np.zeros(self.network_state_num, dtype=float))
        for i in xrange(len(self.ActionSpace)):
            STPModel_Network[i], Reward_Network[i] = self.CombineStates(STPModel_Region['R1'][i],
                                                                        STPModel_Region['R2'][i],
                                                                        STPModel_Region['R3'][i],
                                                                        Reward_Region['R1'][i], Reward_Region['R2'][i],
                                                                        Reward_Region['R3'][i])
        self.STPModel_Network = STPModel_Network
        self.Reward_Network = Reward_Network


        begin_time = time()                   
        Optimal_policy = self.Policy_Optimize()
        end_time = time()
        run_time = round(end_time - begin_time)
        print('####################################'+str(run_time))

        return self.state, self.ActionSpace, Optimal_policy, action_interval













