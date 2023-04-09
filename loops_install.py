import os,sys
import xml.etree.ElementTree as etree
import xml.dom.minidom as doc
from collections import defaultdict

if __name__ == '__main__':

	name = sys.argv[1]
	level = sys.argv[2]

	print '======================================'
	print name,level
	# level = 4000

	Chj_net = etree.parse('./ITS_Chj_final.net.xml')
	NetRoot = Chj_net.getroot()

	### read the edge id in each region
	f1 = open('R1_EdgeID.txt','r')
	R1_EdgeID = list(f1)
	R1EdgeNum = len(R1_EdgeID)
	for i in range(R1EdgeNum):
		R1_EdgeID[i]=R1_EdgeID[i].strip('\n')[5:]

	f2 = open('R2_EdgeID.txt','r')
	R2_EdgeID = list(f2)
	R2EdgeNum = len(R2_EdgeID)
	for i in range(R2EdgeNum):
		R2_EdgeID[i]=R2_EdgeID[i].strip('\n')[5:]

	f3 = open('R3_EdgeID.txt','r')
	R3_EdgeID = list(f3)
	R3EdgeNum = len(R3_EdgeID)
	for i in range(R3EdgeNum):
		R3_EdgeID[i]=R3_EdgeID[i].strip('\n')[5:]

	## get the edges between regions
	BetLoopsType = ['R1-R2', 'R1-R3', 'R2-R1', 'R2-R3', 'R3-R1', 'R3-R2']
	BetRegionLoops = defaultdict(list)
	for connection in NetRoot.iter('connection'):
		tl = connection.get('tl')
		if tl is not None:
			if tl[0:5] in BetLoopsType:
				edge_id = connection.get('to')
				if edge_id not in BetRegionLoops[tl[0:5]]:
					BetRegionLoops[tl[0:5]].append(edge_id)

	##get the edges of the input and output of regions
	RegionLoopsType = ['R1I','R1O','R2I','R2O','R3I','R3O']
	RegionLoops = defaultdict(list)
	for edge in NetRoot.findall('edge'):
		edge_id = edge.get('id')
		if edge_id[0:3] in RegionLoopsType:
			if edge_id not in RegionLoops[edge_id[0:3]]:
				RegionLoops[edge_id[0:3]].append(edge_id)

	loops_files = doc.Document()
	addition = loops_files.createElement('additional')
	loops_files.appendChild(addition)


	for edge in NetRoot.findall('edge'):
		edge_id = edge.get('id')
		### determine the region that the edge belongs to
		edge_region=''
		if edge_id in R1_EdgeID:
			edge_region = 'R1'
		elif edge_id in R2_EdgeID:
			edge_region = 'R2'
		elif edge_id in R3_EdgeID:
			edge_region = 'R3'
		else:
			print edge_id
		### determine the position that the edge locates
		edge_position=''
		for temp in RegionLoops.keys():
			if edge_id in RegionLoops[temp]:
				edge_position = temp
		for temp in BetRegionLoops.keys():
			if edge_id in BetRegionLoops[temp]:
				edge_position = temp
		for lane in edge.findall('lane'):
			lane_id = lane.get('id')
			lane_len = float(lane.get('length'))
			loop = loops_files.createElement('inductionLoop')
			loop.setAttribute('id',lane_id)
			loop.setAttribute('Region',edge_region)
			if len(edge_position)>0:
				loop.setAttribute('Position',edge_position)
			loop.setAttribute('ForRoads',edge_id)
			loop.setAttribute('lane',lane_id)
			loop.setAttribute('pos',str(10))
			loop.setAttribute('freq','60')
			loop.setAttribute('file','./ITS_results/'+name+'/ITS_'+name+'_DetInfo'+level+'.xml')
			addition.appendChild(loop)

	fp = open('./loops_ctrl.xml','w')

	try:
		loops_files.writexml(fp,indent='\t', addindent='\t',newl='\n',encoding="utf-8")
	except:
		trackback.print_exc()
	finally:
		fp.close()