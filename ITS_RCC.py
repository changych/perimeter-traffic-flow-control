# -*- coding: UTF-8 -*-
import os, sys
import xml.etree.ElementTree as etree
import xml.dom.minidom as doc
import math
import numpy as np
import random
# from tqdm import tqdm
# from time import time
from scipy.optimize import linprog
# from cvxopt import matrix,solvers
from scipy.sparse import identity
from collections import defaultdict
import pickle
import cplex
import pdb
import matplotlib.pyplot as plt
import scipy.io as scio

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

sumoCmd = [sumoBinary, "-c", "chj.sumocfg", "--seed",'101']#str(random.randint(1, 100))

PORT = 8813
import traci

### read the edge id in each region
AllEdgesList = []
f1 = open('R1_EdgeID.txt','r')
R1_EdgeID = list(f1)
R1EdgeNum = len(R1_EdgeID)
for i in range(R1EdgeNum):
	R1_EdgeID[i]=R1_EdgeID[i].strip('\n')[5:]
	AllEdgesList.append('R1_'+R1_EdgeID[i])

f2 = open('R2_EdgeID.txt','r')
R2_EdgeID = list(f2)
R2EdgeNum = len(R2_EdgeID)
for i in range(R2EdgeNum):
	R2_EdgeID[i]=R2_EdgeID[i].strip('\n')[5:]
	AllEdgesList.append('R2_' + R2_EdgeID[i])

f3 = open('R3_EdgeID.txt','r')
R3_EdgeID = list(f3)
R3EdgeNum = len(R3_EdgeID)
for i in range(R3EdgeNum):
	R3_EdgeID[i]=R3_EdgeID[i].strip('\n')[5:]
	AllEdgesList.append('R3_' + R3_EdgeID[i])


def checkConstraints(states,As,bs):
	x = states
	# print(x,As,bs)
	c = np.zeros(len(bs))
	for i in xrange(len(bs)):
		# temp = 0
		c[i]=np.dot(As[i],x)-bs[i]
		# for j in range(3):
		# 	temp += As[i,j]*x[j]
		# c[i] = temp-bs[i]
	# c = As*x-bs
	if max(c)>0:
		flag = 1
	else:
		flag = 0
	return flag



def RCC_ctrl(Mode,Setpoints,VehNumRegion,all_As,all_bs,k_set):
	## reference points
	ReferStates = np.zeros((3,1))
	ReferStates[:,0]=Setpoints[0:3]

	ReferAction = np.zeros((6,1))
	ReferAction[:,0] = Setpoints[3:]

	## current delta states
	states = np.zeros((3,1))
	states[0]=VehNumRegion['R1']
	states[1]=VehNumRegion['R2']
	states[2]=VehNumRegion['R3']
	deltstates = states-ReferStates

	# all_As = BatchA[mode]
	# all_bs = BatchB[mode]
	# k_set = KSet[mode]

	### determine the polytopic that the state is in 
	whichPoly = 1
	for batch in range(1,4):
		As = all_As[batch-1]
		bs = all_bs[batch-1]
		checkbool = checkConstraints(deltstates,As,bs)
		if checkbool==0:
			whichPoly = batch

	## calculate the feedback law
	online_a = 0
	online_b = 0
	if whichPoly==3:
		F = k_set[2]
	else:
		online_a = 1/(max(np.dot(all_As[whichPoly-1],deltstates)))
		online_b = 1/(max(np.dot(all_As[whichPoly],deltstates)))
		online_lambda = (1-online_b)/(online_a-online_b)
		F = online_lambda*k_set[whichPoly-1]*online_a+(1-online_lambda)*k_set[whichPoly]*online_b

	DeltCtrlValue = np.dot(F,deltstates)
	CtrlValue = DeltCtrlValue+ReferAction
	return CtrlValue

##############################################################################################################
if __name__ == '__main__':
	name = sys.argv[1]
	level = sys.argv[2]
	print '======================================'
	print name, level
	# level = 4000
	#############################################################################
	### load control laws
	Modes = ['000', '001', '010', '011', '100', '101', '110', '111']
	BatchA = dict()
	BatchB = dict()
	KSet = dict()
	for times in range(1, 9):
		A = scio.loadmat('./Controller_ITS2/AsMode' + str(times) + '.mat')
		B = scio.loadmat('./Controller_ITS2/bsMode' + str(times) + '.mat')
		K = scio.loadmat('./Controller_ITS2/KMode' + str(times) + '.mat')
		ktemp = np.zeros((3, 6, 3))
		for i in range(3):
			for j in range(6):
				for n in range(3):
					ktemp[i][j, n] = K['K_set_2th'][j][n, i]

		BatchA[Modes[times - 1]] = A['AsMode'][0]
		BatchB[Modes[times - 1]] = B['bsMode'][0]
		KSet[Modes[times - 1]] = ktemp

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

	### CtrlBetEdges: get the controlled edges between edges ['R1-R2','R1-R3','R2-R1','R2-R3','R3-R1','R3-R2']
	### TransflowMove: set the traffic flow among regions
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
	Tu = 3  ## upper level control cycle
	Veh_L = 12

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

	#############################################################################
	#################set variables######################
	NJam = {'R1': 5683.355833333334, 'R2': 3839.5775, 'R3': 6242.914166666667}
	# Nmax = {'R1': 2035.7174010590718, 'R2': 137?9.620420632074, 'R3': 2225.3686825451805}
	Nmax = {'R1': 1498, 'R2': 1012, 'R3': 1554}
	ucg = 0.9
	cg = 1.3
	ReferenceStates = {'000':[Nmax['R1']*ucg,Nmax['R2']*ucg,Nmax['R3']*ucg],
					'001':[Nmax['R1']*ucg,Nmax['R2']*ucg,Nmax['R3']*cg],
					'010':[Nmax['R1']*ucg,Nmax['R2']*cg,Nmax['R3']*ucg],
					'011':[Nmax['R1']*ucg,Nmax['R2']*cg,Nmax['R3']*cg],
					'100':[Nmax['R1']*cg,Nmax['R2']*ucg,Nmax['R3']*ucg],
					'101':[Nmax['R1']*cg,Nmax['R2']*ucg,Nmax['R3']*cg],
					'110':[Nmax['R1']*cg,Nmax['R2']*cg,Nmax['R3']*ucg],
					'111':[Nmax['R1']*cg,Nmax['R2']*cg,Nmax['R3']*cg]}
	ReferActions = {'000':[1,1,1,1,1,1],
					'001':[1,1,0.5,1,0.5,1],
					'010':[0.5,1,1,1,1,0.5],
					'011':[0.5,1,0.5,1,0.5,0.5],
					'100':[1,0.5,1,0.5,1,1],
					'101':[1,0.5,0.5,0.5,0.5,1],
					'110':[0.5,0.5,1,0.5,1,0.5],
					'111':[0.5,0.5,0.5,0.5,0.5,0.5]}
    #############################################################################

	traci.start(sumoCmd)

	### initialize the signal settings for all controlled intersections
	VehNumEdge = defaultdict(lambda: 0)  ### record
	VehNumRegion = defaultdict(lambda: 0)
	PassVehNum = defaultdict(lambda: 0)
	Counter = 0
	zone = ['R1', 'R2', 'R3']
	warm_up = Tc * Tu
	TransflowValue = np.ones(6)
	linktype = ['R1-R2', 'R2-R1', 'R1-R3', 'R3-R1', 'R2-R3', 'R3-R2']
	flag=0

	############################################################################################################
	############################ simulation start ###########################################
	for i in range(Tsim):

		traci.simulationStep()

		###record the number of vehicles on each edge in one cycle
		for edge in AllEdgesList:
			edge_id = edge[3:]
			VehNumEdge[edge] += traci.edge.getLastStepVehicleNumber(edge_id)
			# OccupyEdge[edge]+= traci.edge.getLastStepOccupancy(edge_id)

		if i > warm_up:

			#################################################################################
			################# update the upper level policy every Tu cycles #################
			if i % Tc  == 0:
				zone = ['R1', 'R2', 'R3']
				for z in zone:
					VehNumRegion[z]= 0
				for edge in AllEdgesList:
					region_id = edge[0:2]
					edge_id = edge[3:]
					VehNumRegion[region_id]+=traci.edge.getLastStepVehicleNumber(edge_id)

				if i % Tc == 0 and flag == 0:
					for j in range(3):
						if VehNumRegion[zone[j]] > 0.9*Nmax[zone[j]]:
							flag = 1
							print(i)

				##determine the mode of current states
				if i % Tc == 0 and flag==1:

					if VehNumRegion['R1']<Nmax['R1'] and VehNumRegion['R2']<Nmax['R2'] and VehNumRegion['R3']<Nmax['R3']:
						Mode = '000'
						#[n1,n2,n3,'R1-R2', 'R1-R3', 'R2-R1', 'R2-R3', 'R3-R1', 'R3-R2']:
						Setpoints=ReferenceStates[Mode]+ReferActions[Mode]
						TransflowValue = RCC_ctrl(Mode,Setpoints,VehNumRegion,BatchA[Mode],BatchB[Mode],KSet[Mode])

					elif VehNumRegion['R1']<Nmax['R1'] and VehNumRegion['R2']<Nmax['R2'] and VehNumRegion['R3']>Nmax['R3']:
						Mode = '001'
						#[n1,n2,n3,'R1-R2', 'R1-R3', 'R2-R1', 'R2-R3', 'R3-R1', 'R3-R2']:
						Setpoints=ReferenceStates[Mode]+ReferActions[Mode]
						TransflowValue = RCC_ctrl(Mode,Setpoints,VehNumRegion,BatchA[Mode],BatchB[Mode],KSet[Mode])

					elif VehNumRegion['R1']<Nmax['R1'] and VehNumRegion['R2']>Nmax['R2'] and VehNumRegion['R3']<Nmax['R3']:
						Mode = '010'
						#[n1,n2,n3,'R1-R2', 'R1-R3', 'R2-R1', 'R2-R3', 'R3-R1', 'R3-R2']:
						Setpoints=ReferenceStates[Mode]+ReferActions[Mode]
						TransflowValue = RCC_ctrl(Mode,Setpoints,VehNumRegion,BatchA[Mode],BatchB[Mode],KSet[Mode])

					elif VehNumRegion['R1']<Nmax['R1'] and VehNumRegion['R2']>Nmax['R2'] and VehNumRegion['R3']>Nmax['R3']:
						Mode = '011'
						#[n1,n2,n3,'R1-R2', 'R1-R3', 'R2-R1', 'R2-R3', 'R3-R1', 'R3-R2']:
						Setpoints=ReferenceStates[Mode]+ReferActions[Mode]
						TransflowValue = RCC_ctrl(Mode,Setpoints,VehNumRegion,BatchA[Mode],BatchB[Mode],KSet[Mode])

					elif VehNumRegion['R1']>Nmax['R1'] and VehNumRegion['R2']<Nmax['R2'] and VehNumRegion['R3']<Nmax['R3']:
						Mode = '100'
						#[n1,n2,n3,'R1-R2', 'R1-R3', 'R2-R1', 'R2-R3', 'R3-R1', 'R3-R2']:
						Setpoints=ReferenceStates[Mode]+ReferActions[Mode]
						TransflowValue = RCC_ctrl(Mode,Setpoints,VehNumRegion,BatchA[Mode],BatchB[Mode],KSet[Mode])

					elif VehNumRegion['R1']>Nmax['R1'] and VehNumRegion['R2']<Nmax['R2'] and VehNumRegion['R3']>Nmax['R3']:
						Mode = '101'
						#[n1,n2,n3,'R1-R2', 'R1-R3', 'R2-R1', 'R2-R3', 'R3-R1', 'R3-R2']:
						Setpoints=ReferenceStates[Mode]+ReferActions[Mode]
						TransflowValue = RCC_ctrl(Mode,Setpoints,VehNumRegion,BatchA[Mode],BatchB[Mode],KSet[Mode])

					elif VehNumRegion['R1']>Nmax['R1'] and VehNumRegion['R2']>Nmax['R2'] and VehNumRegion['R3']<Nmax['R3']:
						Mode = '110'
						#[n1,n2,n3,'R1-R2', 'R1-R3', 'R2-R1', 'R2-R3', 'R3-R1', 'R3-R2']:
						Setpoints=ReferenceStates[Mode]+ReferActions[Mode]
						TransflowValue = RCC_ctrl(Mode,Setpoints,VehNumRegion,BatchA[Mode],BatchB[Mode],KSet[Mode])

					elif VehNumRegion['R1']>Nmax['R1'] and VehNumRegion['R2']>Nmax['R2'] and VehNumRegion['R3']>Nmax['R3']:
						Mode = '111'
						#[n1,n2,n3,'R1-R2', 'R1-R3', 'R2-R1', 'R2-R3', 'R3-R1', 'R3-R2']:
						Setpoints=ReferenceStates[Mode]+ReferActions[Mode]
						TransflowValue = RCC_ctrl(Mode,Setpoints,VehNumRegion,BatchA[Mode],BatchB[Mode],KSet[Mode])
					else:
						print('no solution!!!')

					print(str(i)+'  '+Mode)

					################## implement the signal setting of controlled edges  #################
					for k in range(6):
						bet = linktype[k]
						for edge_id in TransflowMove[bet].keys():
							TransflowMove[bet][edge_id]=TransflowValue[k]*CtrlEdgeLaneNum[edge_id]*18#veh/min

					action = Action_files.createElement('action')
					action.setAttribute('time',str(i))
					ka = 0
					for connection in linktype: 
						BetRegion = Action_files.createElement('BetRegion')
						BetRegion.setAttribute('name',connection)
						BetRegion.setAttribute('value',str(TransflowValue[ka][0]))
						ka+=1
						# for edge in TransflowMove[connection].keys():
						# 	action_value = TransflowMove[connection][edge][0]
						# 	# print(TransflowMove[connection][edge],action_value)
						# 	Betedge = Action_files.createElement('Betedge')
						# 	Betedge.setAttribute('value',str(action_value))
						# 	Betedge.setAttribute('edge_id',edge)
						# 	BetRegion.appendChild(Betedge)
						action.appendChild(BetRegion)
					actions.appendChild(action)




		############################################################################################################################
		####################################-------------Implement the signal setting--------------###################################
		if i >  warm_up:
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
		####################################-------------record--------------###################################
		if Counter == Tc:
			Counter = 0
			#### record the vehnum on each pahse every 60 seconds
			period = i / Tc
			for edge in AllEdgesList:
				interval = VehNum_files.createElement('interval')
				interval.setAttribute('begin', str(period * Tc))
				interval.setAttribute('end', str((period + 1) * Tc))
				interval.setAttribute('id', edge)
				interval.setAttribute('vehnum', str(VehNumEdge[edge] / Tc))
				# interval.setAttribute('Occupy',str(OccupyEdge[edge]/Tc))
				VehNum.appendChild(interval)
			VehNumEdge = defaultdict(lambda: 0)
		# OccupyEdge = defaultdict(lambda:0)

		Counter = Counter + 1

	# fp = open('./ITS_results/RCC/ITS_RCC_VehNum'+str(level)+'.xml', 'w')
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

	traci.close()