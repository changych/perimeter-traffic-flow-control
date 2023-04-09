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



##############################################################################################################
if __name__ == '__main__':
	name = sys.argv[1]
	level = sys.argv[2]
	print '======================================'
	print name, level
	# level = '12000'
	# name = 'BB'
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
				TransflowMove[connection][edge_id] = 1

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
	flag=0

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

	# LinksOccupy = {"R1":4600,"R2":3800,"R3":5800}
	### get the MFD curves
	Nmax = defaultdict(lambda: 0)
	# TotalData = np.load('./TotalData.npy').item()
	# for zone in ["R1", "R2", "R3"]:
	# 	fig, ax = plt.subplots()
	# 	ax.scatter(TotalData[zone][:, 1], TotalData[zone][:, 0])
	# 	ax.set_ylim([0, 600])
	# 	# ax.set_xlim([0, 2600])
	# 	plt.title('Region ' + zone)
	# 	plt.xlabel('Vehicle number')
	# 	plt.ylabel('Average traffic flow per hour')

	# 	#### estimate the MFD curve and the saturation traffic flow
	# 	z = np.polyfit(TotalData[zone][:, 1], TotalData[zone][:, 0], 3)
	# 	p = np.poly1d(z)
	# 	x = np.linspace(0, np.max(TotalData[zone][:, 1]), 1000)
	# 	y = p(x)

	# 	maxpos = np.where(y == np.max(y))
	# 	Nmax[zone] = x[maxpos]
	# 	plt.plot(x, y)

	# plt.show()

	Nmax = {'R1': 1498, 'R2': 1012, 'R3': 1554}

    #############################################################################
	traci.start(sumoCmd)
	

	### initialize the signal settings for all controlled intersections
	VehNumEdge = defaultdict(lambda: 0)  ### record
	VehNumRegion = defaultdict(lambda: np.zeros(Tc))
	Counter = 0

	zone = ['R1', 'R2', 'R3']
	warm_up = Tc * Tu


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
			j1 = i % Tc - 1
			if j1 < 0:
				j1 = 59
			for zone in ['R1', 'R2', 'R3']:
				VehNumRegion[zone][j1] = 0
			for edge in AllEdgesList:
				region_id = edge[0:2]
				VehNumRegion[region_id][j1] += traci.edge.getLastStepVehicleNumber(edge[3:])


			if i % Tc == 0 and flag == 0:
				zone = ['R1', 'R2', 'R3']
				N_current = np.zeros(3)
				for j in range(3):
					N_current[j] = np.mean(VehNumRegion[zone[j]])
					if N_current[j] > 0.9*Nmax[zone[j]]:
						flag = 1
						print(i)

			if i % Tc  == 0 and flag==1:

				zone = ['R1', 'R2', 'R3']
				Nveh = defaultdict(lambda :0)
				for j in zone:
					Nveh[j] = np.mean(VehNumRegion[j])


				linktype = ['R1-R2', 'R1-R3', 'R2-R3']
				for bet in linktype:
					r1 = bet[0:2]
					r2 = bet[3:5]
					if Nveh[r1]<=Nmax[r1] and Nveh[r2]<=Nmax[r2]:
						connection = r1+'-'+r2
						for edge in TransflowMove[connection].keys():
							TransflowMove[connection][edge]=0.75
						connection = r2+'-'+r1
						for edge in TransflowMove[connection].keys():
							TransflowMove[connection][edge]=0.75
					elif Nveh[r1]>Nmax[r1] and Nveh[r2]<=Nmax[r2]:
						connection = r1 + '-' + r2
						for edge in TransflowMove[connection].keys():
							TransflowMove[connection][edge] = 0.75
						connection = r2 + '-' + r1
						for edge in TransflowMove[connection].keys():
							TransflowMove[connection][edge] = 0.25
					elif Nveh[r1]<=Nmax[r1] and Nveh[r2]>Nmax[r2]:
						connection = r1 + '-' + r2
						for edge in TransflowMove[connection].keys():
							TransflowMove[connection][edge] = 0.25
						connection = r2 + '-' + r1
						for edge in TransflowMove[connection].keys():
							TransflowMove[connection][edge] = 0.75
					elif Nveh[r1]>Nmax[r1] and Nveh[r2]>Nmax[r2]:
						ratio_r1 = Nveh[r1]/LinksOccupy[r1]
						ratio_r2 = Nveh[r2]/LinksOccupy[r2]
						if ratio_r1>ratio_r2:
							connection = r1 + '-' + r2
							for edge in TransflowMove[connection].keys():
								TransflowMove[connection][edge] = 0.75
							connection = r2 + '-' + r1
							for edge in TransflowMove[connection].keys():
								TransflowMove[connection][edge] = 0.25
						else:
							connection = r1 + '-' + r2
							for edge in TransflowMove[connection].keys():
								TransflowMove[connection][edge] = 0.25
							connection = r2 + '-' + r1
							for edge in TransflowMove[connection].keys():
								TransflowMove[connection][edge] = 0.75

				action = Action_files.createElement('action')
				action.setAttribute('time',str(i))
				for connection in ['R1-R2','R1-R3','R2-R1','R2-R3','R3-R1','R3-R2']:
					BetRegion = Action_files.createElement('BetRegion')
					BetRegion.setAttribute('name',connection)
					for edge in TransflowMove[connection].keys():
						action_value = TransflowMove[connection][edge]
						# Betedge = Action_files.createElement('Betedge')
						# Betedge.setAttribute('value',str(action_value))
						# Betedge.setAttribute('edge_id',edge)
						# BetRegion.appendChild(Betedge)
					BetRegion.setAttribute('value',str(action_value))
					action.appendChild(BetRegion)
				actions.appendChild(action)


		################## implement the signal setting of controlled edges  #################
		for bet in TransflowMove.keys():
			for edge_id in CtrlBetEdges[bet]:
				if TransflowMove[bet][edge_id]==1:
					traci.trafficlights.setRedYellowGreenState(CtrlEdgeAction[edge_id][0],CtrlEdgeAction[edge_id][1])
				elif TransflowMove[bet][edge_id]==0:
					#print(CtrlEdgeAction[edge_id][0])
					traci.trafficlights.setRedYellowGreenState(CtrlEdgeAction[edge_id][0],CtrlEdgeAction[edge_id][2])



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

	# fp = open('./ITS_results/BB/ITS_BB_VehNum'+str(level)+'.xml', 'w')
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