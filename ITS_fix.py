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
	# level = 4000
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

	#### parameters
	Tsim = 3600
	Tc = 60
	Tu = 3  ## upper level control cycle
	Veh_L = 12




	# plt.show()
    #############################################################################
	traci.start(sumoCmd)

	### initialize the signal settings for all controlled intersections
	VehNumEdge = defaultdict(lambda: 0)  ### record
	VehNumRegion = defaultdict(lambda: 0)

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

	# fp = open('./ITS_results/FT/ITS_FT_VehNum'+str(level)+'.xml', 'w')
	fp = open('./ITS_results/' + name + '/ITS2_' + name + '_VehNum' + level + '.xml', 'w')

	try:
		VehNum_files.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding="utf-8")
	except:
		trackback.print_exc()
	finally:
		fp.close()

	traci.close()