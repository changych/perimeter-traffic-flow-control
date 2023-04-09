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
from scipy.optimize import minimize

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


##################################constraints#############################################
def GValue(f, x):
	fv = f[0]*x**4+f[1]*x**3+f[2]*x**2+f[3]*x
	# fv = f[0] * x * x * x + f[1] * x * x + f[2] * x + f[3]
	return fv

def cons1_FirstStep(n_initial,M_initial,demand,sv):
	#[n11,n12,n13,n21,n22,n23,n31,n32,n33,M11,M12,M13,M21,M22,M23,M31,M32,M33,u12,u13,u21,u23,u31,u32]
	cons1 = []
	cons1.append({'type': 'eq', 'fun': lambda x: n_initial[0]+demand[0]+x[sv+3]*M_initial[3]+x[sv+5]*M_initial[6]-M_initial[0]-x[0]})
	cons1.append({'type': 'eq', 'fun': lambda x: n_initial[1]+demand[1]-x[sv+1]*M_initial[1]-x[1]})
	cons1.append({'type': 'eq', 'fun': lambda x: n_initial[2]+demand[2]-x[sv+2]*M_initial[2]-x[2]})

	cons1.append({'type': 'eq', 'fun': lambda x: n_initial[4]+demand[4]+x[sv+1]*M_initial[1]+x[sv+6]*M_initial[7]-M_initial[4]-x[4]})
	cons1.append({'type': 'eq', 'fun': lambda x: n_initial[3]+demand[3]-x[sv+3]*M_initial[3]-x[3]})
	cons1.append({'type': 'eq', 'fun': lambda x: n_initial[5]+demand[5]-x[sv+4]*M_initial[5]-x[5]})

	cons1.append({'type': 'eq', 'fun': lambda x: n_initial[8]+demand[8]+x[sv+2]*M_initial[2]+x[sv+4]*M_initial[5]-M_initial[8]-x[8]})
	cons1.append({'type': 'eq', 'fun': lambda x: n_initial[6]+demand[6]-x[sv+5]*M_initial[6]-x[6]})
	cons1.append({'type': 'eq', 'fun': lambda x: n_initial[7]+demand[7]-x[sv+6]*M_initial[7]-x[7]})

	return cons1


def cons4_vehsum(Np,sm):
	cons4 = []
	## n11+n12+n13 = n1
	for i in range(Np):

		cons4.append({'type': 'eq','fun': lambda x: x[i*sm+0]+x[i*sm+1]+x[i*sm+2]-x[i*sm+sm-3]})
		cons4.append({'type': 'eq', 'fun': lambda x: x[i*sm+3] + x[i*sm+4] + x[i*sm+5] - x[i*sm+sm-2]})
		cons4.append({'type': 'eq', 'fun': lambda x: x[i*sm+6] + x[i*sm+7] + x[i*sm+8] - x[i*sm+sm-1]})

	return cons4



def cons6_lmax(Np,NJam,sm):
	cons6 = []
	for i in range(Np):
		###n1<nmax
		cons6.append({'type': 'ineq', 'fun': lambda x: NJam['R1']-x[i*sm+sm-3]})

		cons6.append({'type': 'ineq', 'fun': lambda x: NJam['R2']-x[i*sm+sm-2]})

		cons6.append({'type': 'ineq', 'fun': lambda x: NJam['R3']-x[i*sm+sm-1]})
	return cons6


def cons7_lzero(nlist):  ##>0
	cons7 = []
	for i in nlist:
		cons7.append({'type':'ineq','fun':lambda x: x[i]-0})
	# return {'type': 'ineq', 'fun': lambda x: x[nlist]-0}
	return cons7

def cons8_ul(sv,Nc):
	cons8 = []
	for i in range(6*Nc):
		cons8.append({'type':'ineq','fun':lambda x: 1-x[sv+i+1]})
	return cons8


def set_bounds(NJam,Number):
	bounds = []
	for i in Number:
		if i<=8:
			bounds.append([0,None])
		elif i==9:
			bounds.append([0,NJam['R1']])
		elif i == 10:
			bounds.append([0, NJam['R2']])
		elif i == 11:
			bounds.append([0, NJam['R3']])
		elif i>=12:
			bounds.append([0.2,1])
	return bounds

def obj(x):
	f=x[9]+x[10]+x[11]
	return f

def set_initial(xlist):
	x_0 = np.zeros(len(xlist))
	for i in range(6):
		x_0[12+i]=1
	return x_0


##############################################################################################################
if __name__ == '__main__':
	name = sys.argv[1]
	level = sys.argv[2]
	print '======================================'
	print name, level
	# level = '4000'
	#############################################################################


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

	RegionOut = {'R1Out': ['R1-R2', 'R1-R3', 'R1O'], 'R2Out': ['R2-R1', 'R2-R3', 'R2O'],'R3Out': ['R3-R1', 'R3-R2', 'R3O']}
	RegionIn = {'R1In': ['R2-R1', 'R2-R1', 'R1I'], 'R2In': ['R1-R2', 'R3-R2', 'R2I'], 'R3In': ['R1-R3', 'R2-R3', 'R3I']}

	####loops installed in the enterance of the network
	#  {"R1":[loop_input_id]}
	LoopsInput = defaultdict(list)
	for zone in ["R1", "R2", "R3"]:
		for in_type in RegionIn[zone + 'In']:
			for lane in LaneTypes[in_type]:
				LoopsInput[zone].append(lane)

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

	# Nmax = {'R1': 2035.7174010590718, 'R2': 1379.620420632074, 'R3': 2225.3686825451805}
	Nmax = {'R1': 1498, 'R2': 1012, 'R3': 1554}

 

	# G = {'R1': [1.056e-08, -0.00013, 0.398, 18.73],
	# 	 'R2': [2.96e-08, -0.0002454, 0.5081, 14.11],
	# 	 'R3': [7.644e-09, -0.0001022, 0.3413, 16.6]}
	G = {'R1': [-5.62e-12,7.22e-08,-3.44e-04,0.61],
		 'R2': [-2.31e-11,2.06e-07,-6.60e-04,0.80],
		 'R3': [-4.12e-12,5.84e-08,-2.92e-04,0.55]}

	RegionOut = {'R1Out': ['R1-R2', 'R1-R3', 'R1O'], 'R2Out': ['R2-R1', 'R2-R3', 'R2O'],
				 'R3Out': ['R3-R1', 'R3-R2', 'R3O']}

	sm = 12 ## state variables 9*2+3
	Np = 1
	Nc = 1
	sv = sm * Np - 1
	datalen = 3
	DemandToRegion = np.zeros((datalen,9))
	Demand = np.zeros(9)
	flag=0
	PassVehNum = defaultdict(lambda: 0)

	#############################################################################

	traci.start(sumoCmd)

	### initialize the signal settings for all controlled intersections
	VehNumEdge = defaultdict(lambda: 0)  ### record
	VehNumRegion = defaultdict(lambda: 0)
	Counter = 0
	zone = ['R1', 'R2', 'R3']
	warm_up = Tc * Tu
	TransflowValue = np.ones(6)
	linktype = ['R1-R2', 'R1-R3', 'R2-R1', 'R2-R3', 'R3-R1', 'R3-R2']

	############################################################################################################
	############################ simulation start ###########################################
	for i in range(Tsim):

		traci.simulationStep()

		###record the number of vehicles on each edge in one cycle
		for edge in AllEdgesList:
			edge_id = edge[3:]
			VehNumEdge[edge] += traci.edge.getLastStepVehicleNumber(edge_id)
			# OccupyEdge[edge]+= traci.edge.getLastStepOccupancy(edge_id)

		## record the internal demand of each region every 600 seconds
		j2 = i % (datalen * Tc) - 1
		if j2 < 0:
			j2 = Tc * datalen - 1
		for k in range(datalen):
			if j2 == k * Tc:
				DemandToRegion[k,:] = np.zeros(9)  # {loop_id:[Tu]}
			if j2 >= k * Tc and j2 < (k + 1) * Tc:
				for zone in ['R1', 'R2', 'R3']:
					r0 = int(zone[1])-1
					for loop in LoopsInput[zone]:
						vehID = traci.inductionloop.getLastStepVehicleIDs(loop)
						for veh_id in vehID:
							routelist=traci.vehicle.getRoute(veh_id)
							toRegion = routelist[-1]
							toRegionNum = int(toRegion[1])-1
							DemandToRegion[k][r0*3+toRegionNum]+=1

		if i > warm_up:

			#################################################################################
			################# update the upper level policy every Tu cycles #################
			if i % Tc == 0:
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

				if i % Tc == 0 and flag ==1:
					### get the initial value for optimization problem
					vehID = []
					vehID = traci.vehicle.getIDList()

					n_initial = np.zeros(9)  #n11,n12,n13,n21,n22,n23,n31,n32,n33
					for veh_id in vehID:
						routelist = traci.vehicle.getRoute(veh_id)
						edge_id = traci.vehicle.getRoadID(veh_id)
						destination = routelist[-1]
						end_region_num = int(destination[1])
						if edge_id in R1_EdgeID:
							n_initial[end_region_num-1]+=1
						elif edge_id in R2_EdgeID:
							n_initial[3+end_region_num-1]+=1
						elif edge_id in R3_EdgeID:
							n_initial[6 + end_region_num - 1] += 1

					M_initial = np.zeros(9)
					for i0 in range(9):
						if i0 < 3:
							nsum = np.sum(n_initial[0:3])
							M_initial[i0] = GValue(G['R1'], nsum) * n_initial[i0] / nsum
						elif i >= 3 and i < 6:
							nsum = np.sum(n_initial[3:6])
							M_initial[i0] = GValue(G['R2'], nsum) * n_initial[i0] / nsum
						elif i >= 6:
							nsum = np.sum(n_initial[6:9])
							M_initial[i0] = GValue(G['R3'], nsum) * n_initial[i0] / nsum


					for i0 in range(9):
						Demand[i0]=np.mean(DemandToRegion[:,i0])



					##generate the optimization problem
					Number = []
					for i0 in range(Np):
						for j0 in range(sm):
							Number.append(sm * i0 + j0)
					for i0 in range(Nc):
						for j0 in range(6):
							Number.append(sm * Np + 6 * i0 + j0)
					fun = lambda x: obj(x)
					x0 = set_initial(Number)
					cons = []
					cons+=cons1_FirstStep(n_initial,M_initial,Demand,sv)

					cons+=cons4_vehsum(Np,sm)

					bounds = set_bounds(NJam,Number)
					# cons+=cons6_lmax(Np,NJam,sm)
					# cons+=cons7_lzero(Number)
					# cons+=cons8_ul(sv,Nc)
					print(str(i)+'   cons are all set!!!')
					res = minimize(fun, x0, bounds=bounds,constraints=cons) #method='SLSQP'
					solution = res.x
					print(solution[12:18])

					for i0 in range(6):
						TransflowValue[i0]=solution[sv+i0+1]

					################## implement the signal setting of controlled edges  #################
					for k in range(6):
						bet = linktype[k]
						for edge_id in TransflowMove[bet].keys():
							TransflowMove[bet][edge_id]=TransflowValue[k]*CtrlEdgeLaneNum[edge_id]*18#veh/min

					############################################### get the actions
					action = Action_files.createElement('action')
					action.setAttribute('time',str(i))
					ka = 0
					for connection in linktype:
						BetRegion = Action_files.createElement('BetRegion')
						BetRegion.setAttribute('name',connection)
						BetRegion.setAttribute('value',str(TransflowValue[ka]))
						ka+=1
						# for edge in TransflowMove[connection].keys():
						# 	action_value = TransflowMove[connection][edge]
						# 	Betedge = Action_files.createElement('Betedge')
						# 	Betedge.setAttribute('value',str(action_value))
						# 	Betedge.setAttribute('edge_id',edge)
						# 	BetRegion.appendChild(Betedge)
						action.appendChild(BetRegion)
					actions.appendChild(action)

		if i > warm_up:
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

	# fp = open('./ITS_results/MPC1/ITS_MPC1_VehNum'+str(level)+'.xml', 'w')
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