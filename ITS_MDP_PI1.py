# -*- coding: UTF-8 -*-
import os, sys
import xml.etree.ElementTree as etree
import xml.dom.minidom as doc
import math
import numpy as np
import random
# from tqdm import tqdm
from time import time
from scipy.optimize import linprog
# from cvxopt import matrix,solvers
from scipy.sparse import identity
from collections import defaultdict
from Upper_ctrl1 import Update_policy
import pickle
import cplex
import pdb

# SUMO_HOME = "/usr/local/bin/sumo"
# tools = "/home/Arain/sumo-git/tools/"
# sys.path.append(tools)

# sumoBinary = "/usr/local/bin/sumo"

# sumoCmd = [sumoBinary, "-c", "chj.sumocfg","--seed", str(random.randint(1,100))]


if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
sumoBinary = "E:/software/sumo-win64-0.32.0/sumo-0.32.0/bin/sumo"#-gui

sumoCmd = [sumoBinary, "-c", "chj.sumocfg", "--seed", str(random.randint(1, 100))]

PORT = 8813
import traci

##############################################################################################################
if __name__ == '__main__':

    name = sys.argv[1]
    level = sys.argv[2]
    print '======================================'
    print name,level

    # level = '12000'
    # name = 'PI52'
    ##############################################################################################
    #############################################################################################
    ### read the edge id in each region
    AllEdgesList = []
    f1 = open('R1_EdgeID.txt', 'r')
    R1_EdgeID = list(f1)
    R1EdgeNum = len(R1_EdgeID)
    for i in range(R1EdgeNum):
        R1_EdgeID[i] = R1_EdgeID[i].strip('\n')[5:]
        AllEdgesList.append('R1_' + R1_EdgeID[i])

    f2 = open('R2_EdgeID.txt', 'r')
    R2_EdgeID = list(f2)
    R2EdgeNum = len(R2_EdgeID)
    for i in range(R2EdgeNum):
        R2_EdgeID[i] = R2_EdgeID[i].strip('\n')[5:]
        AllEdgesList.append('R2_' + R2_EdgeID[i])

    f3 = open('R3_EdgeID.txt', 'r')
    R3_EdgeID = list(f3)
    R3EdgeNum = len(R3_EdgeID)
    for i in range(R3EdgeNum):
        R3_EdgeID[i] = R3_EdgeID[i].strip('\n')[5:]
        AllEdgesList.append('R3_' + R3_EdgeID[i])

    ##### network file
    doc2 = etree.parse('./ITS_demand' + str(level) + '.rou.xml')
    RouteRoot = doc2.getroot()
    doc3 = etree.parse('./loops_ctrl.xml')
    LoopsRoot = doc3.getroot()
    doc4 = etree.parse('./ITS_Chj_final.net.xml')
    NetRoot = doc4.getroot()

    ValidLoop = defaultdict(list)  ###{'R1':[]}
    LaneTypes = defaultdict(list)
    for loop in LoopsRoot.iter('inductionLoop'):
        region_id = loop.get('Region')
        loop_id = loop.get('id')
        for region in ['R1', 'R2', 'R3']:
            if region_id == region:
                ValidLoop[region].append(loop_id)
        position = loop.get('Position')
        for type in ['R1-R2', 'R1-R3', 'R2-R1', 'R2-R3', 'R3-R1', 'R3-R2', 'R1I', 'R1O', 'R2I', 'R2O', 'R3I', 'R3O']:
            if position == type:
                LaneTypes[type].append(loop_id)

    ### get the controlled edges between edges ['R1-R2','R1-R3','R2-R1','R2-R3','R3-R1','R3-R2']
    CtrlBetEdges = defaultdict(list)
    CtrlEdgeLaneNum = defaultdict(lambda: 0)
    TransflowMove = defaultdict(dict)
    for connection in ['R1-R2', 'R1-R3', 'R2-R1', 'R2-R3', 'R3-R1', 'R3-R2']:
        TransflowMove[connection] = {}
        for lane in LaneTypes[connection]:
            edge_id = lane[:-2]
            CtrlEdgeLaneNum[edge_id] += 1
            if edge_id not in CtrlBetEdges[connection]:
                CtrlBetEdges[connection].append(edge_id)
                TransflowMove[connection][edge_id] = 1000

    CtrlEdgeAction = defaultdict(dict)
    for tl in NetRoot.iter('tlLogic'):
        tl_id = tl.get('id')
        if tl_id[0:5] in ['R1-R2', 'R1-R3', 'R2-R1', 'R2-R3', 'R3-R1', 'R3-R2']:
            lane_num = 0
            for connection in NetRoot.iter('connection'):
                if tl_id == connection.get('tl'):
                    lane_num += 1
                    ctrl_edge = connection.get('to')
            green = ''
            red = ''
            for i in range(lane_num):
                green = green + 'G'
                red = red + 'r'
            CtrlEdgeAction[ctrl_edge] = [tl_id, green, red]

    RegionOut = {'R1Out': ['R1-R2', 'R1-R3', 'R1O'], 'R2Out': ['R2-R1', 'R2-R3', 'R2O'],
                 'R3Out': ['R3-R1', 'R3-R2', 'R3O']}
    RegionIn = {'R1In': ['R2-R1', 'R2-R1', 'R1I'], 'R2In': ['R1-R2', 'R3-R2', 'R2I'], 'R3In': ['R1-R3', 'R2-R3', 'R3I']}

    #####
    VehNum_files = doc.Document()
    VehNum = VehNum_files.createElement('VehicleNumber')
    VehNum_files.appendChild(VehNum)

    Action_files = doc.Document()
    actions = Action_files.createElement('Actions')
    Action_files.appendChild(actions)

    #### parameters
    Tsim = 3600
    Tc = 60
    Tu = 1  ## upper level control cycle
    period = 10  ##seconds
    Veh_L = 12

    # Nmax = [2035.7174010590718, 1379.620420632074, 2225.3686825451805]
    Nmax = [1498,1012,1554]



    ####loops installed in the enterance of the network
    LoopsInput = defaultdict(list)  # {"R1":[loop_input_id]}
    LoopsOutput = defaultdict(list)  # {"R1":[loop_output_id]}
    LoopsOutputNum = defaultdict(lambda: 0)  # {"R1":the number of output loops for region R1}
    LoopsAmongRegions = defaultdict(list)

    for zone in ["R1", "R2", "R3"]:
        for out_type in RegionOut[zone + 'Out']:
            for lane in LaneTypes[out_type]:
                LoopsOutput[zone].append(lane)
        LoopsOutputNum[zone] = len(LoopsOutput[zone])
        for in_type in RegionIn[zone + 'In']:
            for lane in LaneTypes[in_type]:
                LoopsInput[zone].append(lane)

    for connection in ['R1-R2', 'R1-R3', 'R2-R1', 'R2-R3', 'R3-R1', 'R3-R2']:
        for loop in LaneTypes[connection]:
            LoopsAmongRegions[connection].append(loop)

    ########## calculate the link occupancy for links ##########################
    LinksOccupy = {"R1": 0, "R2": 0, "R3": 0}

    for edge in NetRoot.iter('edge'):
        l = 0
        edge_id = edge.get('id')
        if edge_id in R1_EdgeID:
            zone = 'R1'
        elif edge_id in R2_EdgeID:
            zone = 'R2'
        elif edge_id in R3_EdgeID:
            zone = 'R3'
        else:
            continue
        for lane in edge.findall('lane'):
            lanelen = float(lane.get('length'))
            l += 1
        LinksOccupy[zone] += lanelen * l / Veh_L

    ### initialize the signal settings for all controlled intersections
    VehNumEdge = defaultdict(lambda: 0)  ### record
    VehNumRegion = defaultdict(lambda: np.zeros(Tc))
    CtrlEdgeVehNum = defaultdict(dict)
    for bet in ['R1-R2', 'R1-R3', 'R2-R1', 'R2-R3', 'R3-R1', 'R3-R2']:
        CtrlEdgeVehNum[bet] = defaultdict(lambda: np.zeros(Tc))
    ### policy update cycle Tu
    VehNumLoop = defaultdict(lambda: np.zeros(Tu))
    VehNumLoopsAmongRegions = defaultdict(lambda: np.zeros(Tu))

    #### load the model
    with open('./randomodel.pkl', "rb") as tf:
        randmodel = pickle.load(tf)

    ############################################################################################################
    ############################ simulation start ########################################################
    traci.start(sumoCmd)
    Counter = 0
    zone = ['R1', 'R2', 'R3']
    warm_up = Tc * Tu
    flag = 0
    flag0 = 0
    firstOpt = 10000
    OptimizeTimes = 0
    Srange =defaultdict(lambda :np.zeros((3,2)))

    for i in range(Tsim):
        traci.simulationStep()
        ###record the number of vehicles on each edge in one cycle
        for edge in AllEdgesList:
            edge_id = edge[3:]
            VehNumEdge[edge] += traci.edge.getLastStepVehicleNumber(edge_id)
        # OccupyEdge[edge]+= traci.edge.getLastStepOccupancy(edge_id)

        if i > warm_up:  ### warm-up the network

            ########################every Tc steps##################################
            # get the number of vehicles in each region at each time step within one cycle
            j1 = i % Tc - 1
            if j1 < 0:
                j1 = 59
            for zone in ['R1', 'R2', 'R3']:
                VehNumRegion[zone][j1] = 0
            for edge in AllEdgesList:
                region_id = edge[0:2]
                VehNumRegion[region_id][j1] += traci.edge.getLastStepVehicleNumber(edge[3:])

            # get the number of vehicles on the controlled lanes at each time step within one cycle
            for bet in ['R1-R2', 'R1-R3', 'R2-R1', 'R2-R3', 'R3-R1', 'R3-R2']:
                for edge_id in CtrlBetEdges[bet]:
                    CtrlEdgeVehNum[bet][edge_id][j1]=traci.edge.getLastStepVehicleNumber(edge_id)

            ########################every Tu  steps#######################################
            ### get the total number of vehicles entering one region within one cycle, record for the control cycle Tu ——VehNumLoop
            ### get the total number of vehicles driving from one region to another region within one cycle, record for the control cycle Tu—VehNumLoopsAmongRegions
            j2 = i % (Tu * Tc) - 1
            if j2 < 0:
                j2 = Tu * Tc - 1

            # if i > 3000:
                # print('Time to debug!')

            for k in range(Tu):
                if j2 == k * Tc:
                    for zone in ['R1', 'R2', 'R3']:
                        for loop in LoopsInput[zone]:
                            VehNumLoop[loop][k] = 0 #{loop_id:[Tu]}
                if j2 >= k * Tc and j2 < (k + 1) * Tc:
                    for zone in ['R1', 'R2', 'R3']:
                        for loop in LoopsInput[zone]:
                            if traci.inductionloop.getLastStepMeanSpeed(loop) < 0.1:
                                vehnum = 0
                            else:
                                vehnum = traci.inductionloop.getLastStepVehicleNumber(loop)
                            VehNumLoop[loop][k] += vehnum

            if i % Tc==0 and flag0==0:
                zone = ['R1', 'R2', 'R3']
                N_current = np.zeros(3)
                for j in range(3):
                    N_current[j] = np.mean(VehNumRegion[zone[j]])
                    if N_current[j]>0.9*Nmax[j]:
                        flag0=1
                        print(i)
            #################################################################################
            ################# update the upper level policy every Tu cycles #################
            # if i == Tc * Tu+warm_up:
            if flag0 ==1 and i%Tc==0:
                zone = ['R1', 'R2', 'R3']
                firstOpt = i
                flag0 = 2

                ### calculate the number of movements between regions
                ### set the maximum and minimum numbre of vehicles among regions
                SaturationFlow = 0.5#veh/s
                ActionRange = np.zeros((6, 2))#[,the average traffic flow drving R1-R2]
                linktype = ['R1-R2', 'R1-R3', 'R2-R1', 'R2-R3', 'R3-R1', 'R3-R2']
                loopsnum = defaultdict(lambda: 0)#[the number of loops among R1-R2]

                ### get the maximum traffic flow between two adjacent regions
                for bet in linktype:
                    j = linktype.index(bet)
                    ActionRange[j,1]=len(LoopsAmongRegions[bet])
                    ActionRange[j,0] = len(LoopsAmongRegions[bet])
                ####  using the saturation traffic flow
                ActionRange[:,1] = ActionRange[:,1]*SaturationFlow*Tc
                ActionRange[:,0] = ActionRange[:,0] * SaturationFlow * Tc*0.4
                ## use the average output of previous signal cycles
                # for j in xrange(len(linktype)):
                    # ActionRange[j, 1] = ActionRange[j, 1] * np.mean(VehNumLoopsAmongRegions[linktype[j]]) / loopsnum[j]

                ### the average number of vehicles in regions
                N_current = np.zeros(3)
                for j in range(3):
                    N_current[j] = np.mean(VehNumRegion[zone[j]])

                ### the average traffic demand for regions
                D_current = np.zeros(3)
                for j in xrange(3):
                    for loop in LoopsInput[zone[j]]:
                        D_current[j] += np.mean(VehNumLoop[loop])

                begin_time = time()
                ranvalue = np.zeros((3,2))
                for j in range(3):
                    label = int(N_current[j])
                    ranvalue[j,:]=randmodel[zone[j]][label,:]
                upper_controller = Update_policy(Tc, Tu, ActionRange, N_current, D_current, LoopsOutputNum,LinksOccupy,ranvalue)

                label =int(i/Tc)
                Srange[str(label)] = upper_controller.StateRange

                State, ActionSpace, Opt_policy, action_interval = upper_controller.STPM_network()
                end_time = time()
                run_time = round(end_time - begin_time)
                OptimizeTimes += 1
                # print(i,OptimizeTimes,run_time)

            #################################################################################
            ################# the lower level control every one cycles #################
            # print(i,Tc,firstOpt)
            
            if i % Tc == 0 and i>firstOpt:

                ## determine current network state and action to take
                VehNumZone = defaultdict(lambda: 0)
                for zone in ['R1', 'R2', 'R3']:
                    VehNumZone[zone] = np.mean(VehNumRegion[zone])

                state_num = len(State['R1'])-1
                s = -1 * np.ones(3)  ##region
                zone = ['R1', 'R2', 'R3']

                # if VehNumZone['R1']-ActionRange[0,1]-ActionRange[1,1]<State['R1'][0]:
                #     over = 1
                # elif VehNumZone['R1']+ActionRange[2,1]+ActionRange[4,1]>State['R1'][-1]:
                #     over = 1
                # elif VehNumZone['R2']+ActionRange[0,1]+ActionRange[5,1]>State['R2'][-1]:
                #     over = 1

                for k in xrange(3):  # zone 
                    for j in xrange(state_num):  # judge state
                        if VehNumZone[zone[k]] >= State[zone[k]][j] and VehNumZone[zone[k]] < State[zone[k]][j + 1]:
                            s[k] = j
                            break
                    # if s[k] == -1 and VehNumZone[zone[k]] < State[zone[k]][0]:
                    #     s[k]=0
                    # elif s[k] == -1 and VehNumZone[zone[k]] >= State[zone[k]][-1]:
                    #     s[k]=state_num-1
                    if s[k]==-1:
                        flag = 1

                ######################update the policy #####################
                if flag==1:

                    ### the average number of vehicles in regions
                    N_current = np.zeros(3)
                    for j in range(3):
                        N_current[j] = np.mean(VehNumRegion[zone[j]])

                    ### the average traffic demand for regions
                    D_current = np.zeros(3)
                    for j in xrange(3):
                        for loop in LoopsInput[zone[j]]:
                            D_current[j] += np.mean(VehNumLoop[loop])

                    begin_time = time()
                    ranvalue = np.zeros((3, 2))
                    for j in range(3):
                        label = int(N_current[j])
                        ranvalue[j, :] = randmodel[zone[j]][label, :]
                    upper_controller = Update_policy(Tc, Tu, ActionRange, N_current, D_current, LoopsOutputNum,
                                                     LinksOccupy, ranvalue)

                    label = int(i / Tc)
                    Srange[str(label)] = upper_controller.StateRange

                    # print(Srange[str(label)])

                    State, ActionSpace, Opt_policy, action_interval = upper_controller.STPM_network()
                    end_time = time()
                    run_time = round(end_time - begin_time)
                    OptimizeTimes+=1
                    print(i,OptimizeTimes,run_time)

                    flag=0

                    for k in xrange(3):  # zone
                        for j in xrange(state_num):  # judge state
                            if VehNumZone[zone[k]] >= State[zone[k]][j] and VehNumZone[zone[k]] < State[zone[k]][j + 1]:
                                s[k] = j
                                break

                
                ####################################################################
                # print('start take actions')
                State_current=0   
                for j in xrange(3):
                    State_current+=s[j]*(math.pow(state_num, 3-j-1))

                # State_current = np.dot(s, np.array([9, 3, 1]))
                ActionLabel = Opt_policy[int(State_current)]
                Action_current = ActionSpace[int(ActionLabel)]

                #############################################################################################################################################
                ############### Control the boundary intersections  #########################################################
                EdgeVehNum = defaultdict(dict)

                for k in range(6): ##['R1-R2', 'R1-R3', 'R2-R1', 'R2-R3', 'R3-R1', 'R3-R2']
                    OptimalFlow = Action_current[k]
                    bet=linktype[k]
                    EdgeVehNum[bet] = defaultdict(lambda :0)
                    #TransflowMove[RegionMove] = defaultdict(lambda: 0)
                    for edge in CtrlEdgeVehNum[bet].keys():
                        EdgeVehNum[bet][edge] = np.mean(CtrlEdgeVehNum[bet][edge])
                    #max(LaneVehNum[RegionMove], key=LaneVehNum[RegionMove].get)
                    ### calculate the average optimal flow of each lane
                    ContTotalVehNum = np.sum(EdgeVehNum[bet].values())
                    for edge in EdgeVehNum[bet].keys():
                        TransflowMove[bet][edge]=OptimalFlow*EdgeVehNum[bet][edge]/(ContTotalVehNum+0.001) ### the transfer flow according to the controller
                ############################get the actions
                print('start take actions!')
                action = Action_files.createElement('action')
                action.setAttribute('time',str(i))
                ka = 0
                for connection in ['R1-R2','R1-R3','R2-R1','R2-R3','R3-R1','R3-R2']:
                    BetRegion = Action_files.createElement('BetRegion')
                    BetRegion.setAttribute('name',connection)
                    BetRegion.setAttribute('value',str(Action_current[ka]))
                    ka+=1
                    # for edge in TransflowMove[connection].keys():
                    #     action_value = TransflowMove[connection][edge]
                    #     Betedge = Action_files.createElement('Betedge')
                    #     Betedge.setAttribute('value',str(action_value))
                    #     Betedge.setAttribute('edge_id',edge)
                    #     BetRegion.appendChild(Betedge)
                    action.appendChild(BetRegion)
                actions.appendChild(action)

            ################## implement the signal setting of controlled edges  #################
            if i >= Tu * Tc + warm_up:
                j = i%Tc
                if j==0:
                    PassVehNum = defaultdict(lambda:0)
                for bet in TransflowMove.keys():
                    for edge_id in CtrlBetEdges[bet]:
                        lane_num = CtrlEdgeLaneNum[edge_id]
                        for k in range(lane_num):
                            loop_id = edge_id+'_'+str(k)
                            PassVehNum[edge_id]+=traci.inductionloop.getLastStepVehicleNumber(loop_id)
                        if PassVehNum[edge_id]<=TransflowMove[bet][edge_id]:
                            traci.trafficlights.setRedYellowGreenState(CtrlEdgeAction[edge_id][0],CtrlEdgeAction[edge_id][1])
                        else:
                            traci.trafficlights.setRedYellowGreenState(CtrlEdgeAction[edge_id][0],CtrlEdgeAction[edge_id][2])
                            # print(CtrlEdgeAction[edge_id][0],CtrlEdgeAction[edge_id][2])

        ############################################################################################################################
        ####################################-------------Implement the signal setting--------------###################################
        if Counter == Tc:
            Counter = 0
            #### record the vehnum on each pahse every 60 seconds
            period0 = i / Tc
            for edge in AllEdgesList:
                interval = VehNum_files.createElement('interval')
                interval.setAttribute('begin', str(period0 * Tc))
                interval.setAttribute('end', str((period0 + 1) * Tc))
                interval.setAttribute('id', edge)
                interval.setAttribute('vehnum', str(VehNumEdge[edge] / Tc))
                # interval.setAttribute('Occupy',str(OccupyEdge[edge]/Tc))
                VehNum.appendChild(interval)
            VehNumEdge = defaultdict(lambda: 0)
        # OccupyEdge = defaultdict(lambda:0)

        Counter = Counter + 1

    # fp = open('./ITS_results/PI52/state_range'+str(level)+'.xml', 'w')


    fp = open('./ITS_results/' + name + '/ITS2_' + name + '_VehNum' + level + '.xml', 'w')
    fp1 = open('./ITS_results/' + name + '/ITS2_' + name + '_Actions' + level + '.xml', 'w')

    
    try:
        VehNum_files.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding="utf-8")        
        Action_files.writexml(fp1,indent='\t', addindent='\t', newl='\n', encoding="utf-8")
    except:
        trackback.print_exc()
    finally:
        fp.close()
        fp1.close()

    # #save the model to the disk
    # StateRange_files = doc.Document()
    # state_range = StateRange_files.createElement('StateRange')
    # StateRange_files.appendChild(state_range)
    # for key in  Srange.keys():
    #         statezone = StateRange_files.createElement('state')
    #         statezone.setAttribute('time', str(key))
    #         statezone.setAttribute('R1_start', str(Srange[key][0,0]))
    #         statezone.setAttribute('R1_end', str(Srange[key][0, 1]))
    #         statezone.setAttribute('R2_start', str(Srange[key][1, 0]))
    #         statezone.setAttribute('R2_end', str(Srange[key][1, 1]))
    #         statezone.setAttribute('R3_start', str(Srange[key][2, 0]))
    #         statezone.setAttribute('R3_end', str(Srange[key][2,1]))
    #         state_range.appendChild(statezone)

    # fp = open('./ITS_results/Parameter/state_range_new'+str(level)+'.xml', 'w')


    # try:
    #     StateRange_files.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding="utf-8")
    # except:
    #     trackback.print_exc()
    # finally:
    #     fp.close()


    traci.close()