from __future__ import print_function

import random
import os
from time import sleep

def generate_nodes(n, m, filename):
	with open(filename, 'w') as f_nodes:
		print('<?xml version="1.0" encoding="UTF-8"?>', file = f_nodes)
		print('<nodes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/nodes_file.xsd">',
			file = f_nodes)
		for i in range(n):
			for j in range(m):
				if i in [0, n-1] or j in [0, m-1]:
					if i in [0, n-1] and j in [0, m-1]:
						continue
					print('  <node id="({},{})" x="{}" y="{}"  type="priority"/>'.format(
						i, j, (2*i-n+1)*250.0, (2*j-m+1)*250.0), file = f_nodes)
				else:
					print('  <node id="({},{})" x="{}" y="{}"  type="traffic_light"/>'.format(
						i, j, (2*i-n+1)*250.0, (2*j-m+1)*250.0), file = f_nodes)
		print('</nodes>', file = f_nodes)


def generate_edges(n, m, filename):
	direction = [(1,0), (0,1), (-1,0), (0,-1)]
	with open(filename, 'w') as f_edges:
		print('<?xml version="1.0" encoding="UTF-8"?>', file = f_edges)
		print('<edges xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/edges_file.xsd">', file = f_edges)
		for i in range(n):
			for j in range(m):
				if i in [0, n-1] and j in [0, m-1]:
					continue
				for x_t, y_t in direction:
					x = i + x_t
					y = j + y_t
					if (x in range(n)) and (y in range(m)):
						if x in [0, n-1] and y in [0, m-1]:
							continue
						fromID = '({},{})'.format(i,j)
						toID = '({},{})'.format(x,y)
						print('  <edge id="{}_{}" from="{}" to="{}" priority="0" numLanes="4" speed="19.444" />'.format(
							fromID, toID, fromID, toID), file = f_edges)
		print('</edges>', file = f_edges)

def generate_route(seed, n, m, filename, end_time):
	random.seed(seed)
	# probability for straight route, left turn, right turn
	prob = [1./5, 1./20, 1./10]
	with open(filename, 'w') as f_route:
		print('<routes>', file = f_route)
		print('  <vType id="typeCar" accel="0.8" decel="4.5" sigma="0.5" length="6" minGap="3" maxSpeed="16.67" guiShape="passenger"/>', file = f_route)
		all_route_prob = []
		# N to S, S to N
		for i in range(1, n-1):
			edge_list_NS = '({},{})_({},{})'.format(i, 0, i, 1)
			edge_list_SN = '({},{})_({},{})'.format(i, m-1, i, m-2)
			for j in range(2, m):
				edge_list_NS += ' ({},{})_({},{})'.format(i, j-1, i, j)
				edge_list_SN += ' ({},{})_({},{})'.format(i, m-j, i, m-j-1)
			print('  <route id="NS_{}" edges="{}" />'.format(i, edge_list_NS), file = f_route)
			print('  <route id="SN_{}" edges="{}" />'.format(i, edge_list_SN), file = f_route)
			all_route_prob.append((prob[0], "NS_{}".format(i)))
			all_route_prob.append((prob[0], "SN_{}".format(i)))

		# E to W, W to E
		for j in range(1, m-1):
			edge_list_EW = '({},{})_({},{})'.format(0, j, 1, j)
			edge_list_WE = '({},{})_({},{})'.format(n-1, j, n-2, j)
			for i in range(2, n):
				edge_list_EW += ' ({},{})_({},{})'.format(i-1, j, i, j)
				edge_list_WE += ' ({},{})_({},{})'.format(n-i, j, n-i-1, j)
			print('  <route id="EW_{}" edges="{}" />'.format(j, edge_list_EW), file = f_route)
			print('  <route id="WE_{}" edges="{}" />'.format(j, edge_list_WE), file = f_route)
			all_route_prob.append((prob[2], "EW_{}".format(j)))
			all_route_prob.append((prob[2], "WE_{}".format(j)))

		# left turn: EN, SE, WS, NW
		# right turn: ES, SW, WN, NE 
		for i in range(1, n-1):
			for j in range(1, m-1):
				# from E
				edge_list_EN = '({},{})_({},{})'.format(0,j,1,j)
				edge_list_ES = '({},{})_({},{})'.format(0,j,1,j)
				for x in range(2, i+1):
					edge_list_EN += ' ({},{})_({},{})'.format(x-1,j,x,j)
					edge_list_ES += ' ({},{})_({},{})'.format(x-1,j,x,j)
				# from N
				edge_list_NE = '({},{})_({},{})'.format(i,0,i,1)
				edge_list_NW = '({},{})_({},{})'.format(i,0,i,1)
				for y in range(2, j+1):
					edge_list_NE += ' ({},{})_({},{})'.format(i,y-1,i,y)
					edge_list_NW += ' ({},{})_({},{})'.format(i,y-1,i,y)
				# from W
				edge_list_WN = '({},{})_({},{})'.format(n-1,j,n-2,j)
				edge_list_WS = '({},{})_({},{})'.format(n-1,j,n-2,j)
				for x in range(2, n-i):
					edge_list_WN += ' ({},{})_({},{})'.format(n-x,j,n-x-1,j)
					edge_list_WS += ' ({},{})_({},{})'.format(n-x,j,n-x-1,j)
				# from S
				edge_list_SE = '({},{})_({},{})'.format(i,m-1,i,m-2)
				edge_list_SW = '({},{})_({},{})'.format(i,m-1,i,m-2)
				for y in range(2, m-j):
					edge_list_SE += ' ({},{})_({},{})'.format(i,m-y,i,m-y-1)
					edge_list_SW += ' ({},{})_({},{})'.format(i,m-y,i,m-y-1)
				# to W
				for x in range(i, n-1):
					edge_list_NW += ' ({},{})_({},{})'.format(x,j,x+1,j)
					edge_list_SW += ' ({},{})_({},{})'.format(x,j,x+1,j)
				# to S
				for y in range(j, m-1):
					edge_list_ES += ' ({},{})_({},{})'.format(i,y,i,y+1)
					edge_list_WS += ' ({},{})_({},{})'.format(i,y,i,y+1)
				# to E
				for x in range(n-i, n):
					edge_list_NE += ' ({},{})_({},{})'.format(n-x,j,n-x-1,j)
					edge_list_SE += ' ({},{})_({},{})'.format(n-x,j,n-x-1,j)
				# to N
				for y in range(m-j, m):
					edge_list_EN += ' ({},{})_({},{})'.format(i,m-y,i,m-y-1)
					edge_list_WN += ' ({},{})_({},{})'.format(i,m-y,i,m-y-1)

				print('  <route id="EN_({},{})" edges="{}" />'.format(i, j, edge_list_EN), file = f_route)
				print('  <route id="ES_({},{})" edges="{}" />'.format(i, j, edge_list_ES), file = f_route)
				print('  <route id="NE_({},{})" edges="{}" />'.format(i, j, edge_list_NE), file = f_route)
				print('  <route id="NW_({},{})" edges="{}" />'.format(i, j, edge_list_NW), file = f_route)
				print('  <route id="WN_({},{})" edges="{}" />'.format(i, j, edge_list_WN), file = f_route)
				print('  <route id="WS_({},{})" edges="{}" />'.format(i, j, edge_list_WS), file = f_route)
				print('  <route id="SE_({},{})" edges="{}" />'.format(i, j, edge_list_SE), file = f_route)
				print('  <route id="SW_({},{})" edges="{}" />'.format(i, j, edge_list_SW), file = f_route)
				# right turn route
				all_route_prob.append((0*prob[2], "EN_({},{})".format(i,j)))
				all_route_prob.append((0*prob[1], "SE_({},{})".format(i,j)))
				all_route_prob.append((0*prob[2], "WS_({},{})".format(i,j)))
				all_route_prob.append((0*prob[1], "NW_({},{})".format(i,j)))
				# left turn route
				all_route_prob.append((prob[1], "ES_({},{})".format(i,j)))
				all_route_prob.append((prob[1], "SW_({},{})".format(i,j)))
				all_route_prob.append((prob[1], "WN_({},{})".format(i,j)))
				all_route_prob.append((prob[1], "NE_({},{})".format(i,j)))

		# generate vehicle
		car_ID = 0
		for i in range(end_time):
			for prob, routeID in all_route_prob:
				if random.uniform(0,1) < 0.5 * prob:
					print('  <vehicle id="{}_{}" type="typeCar" route="{}" depart="{}" />'.format(
						car_ID, routeID, routeID, i), file = f_route)
					car_ID += 1
		print("</routes>", file = f_route)

def generate_detectors(n, m, net_file_ID, num_lanes = 4):
	direction = [(1,0,0), (0,1,1), (-1,0,2), (0,-1,3)]
	with open("data/grids.det_{}.xml".format(net_file_ID), 'w') as detectors:
		print('<additional>', file = detectors)
		for i in range(1, n-1):
			for j in range(1, m-1):
				to_node = '({},{})'.format(i,j)
				for x_t, y_t, directionID in direction:
					x = i + x_t
					y = j + y_t
					from_node = '({},{})'.format(x,y)
					edge = from_node+'_'+to_node
					for lane in range(num_lanes):
						lane_name = edge + '_{}'.format(lane)
						for block in range(3):
							print('		<laneAreaDetector id="{}" lane="{}" pos="-{}" length="90" freq="30" file="NUL"/>'.format(
							to_node+'_{}_{}_{}'.format(directionID, lane, block), lane_name, 90*(block+1)), file = detectors)

		print('</additional>', file = detectors)
def generate_tlLogic(n, m, filename):
	with open(filename, 'w') as f_tlLogic:
		print('<additional>', file = f_tlLogic)
		for i in range(1,n-1):
			for j in range(1,m-1):
				print('''	<tlLogic id="({},{})" type="static" programID="0" offset="0">
        <phase duration="29" state="GGGGGggrrrrrrrGGGGGggrrrrrrr"/>
        <phase duration="5" state="yyyyyyyrrrrrrryyyyyyyrrrrrrr"/>
        <phase duration="6" state="rrrrrGGrrrrrrrrrrrrGGrrrrrrr"/>
        <phase duration="5" state="rrrrryyrrrrrrrrrrrryyrrrrrrr"/>
        <phase duration="29" state="rrrrrrrGGGGGggrrrrrrrGGGGGgg"/>
        <phase duration="5" state="rrrrrrryyyyyyyrrrrrrryyyyyyy"/>
        <phase duration="6" state="rrrrrrrrrrrrGGrrrrrrrrrrrrGG"/>
        <phase duration="5" state="rrrrrrrrrrrryyrrrrrrrrrrrryy"/>
    </tlLogic>'''.format(i,j), file = f_tlLogic)
		print('</additional>', file = f_tlLogic)

def generate_cfg(filename, net_file_ID, rou_file_ID):
	with open(filename, 'w') as f_cfg:
		print('''<?xml version="1.0" encoding="UTF-8"?>

<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">

    <input>
        <net-file value="grids.net_{}.xml"/>
        <route-files value="grids.rou_{}.xml"/>
        <additional-files value="grids.det_{}.xml"/>
    </input>

    <time>
        <begin value="0"/>
    </time>

    <report>
        <verbose value="true"/>
        <no-step-log value="true"/>
    </report>

</configuration>'''.format(net_file_ID, rou_file_ID, net_file_ID), file = f_cfg)
def initial_netfile(n, m, net_file_ID):
	generate_nodes(n, m, 'data/grids.nod.xml')
	generate_edges(n, m, 'data/grids.edg.xml')
	generate_tlLogic(n,m, 'data/grids.tlLogic.xml')
	os.system('netconvert -n data/grids.nod.xml -e data/grids.edg.xml -i data/grids.tlLogic.xml -o data/grids.net_{}.xml'.format(net_file_ID))	


if __name__ == "__main__":
	file_ID = 0
	seed = 43
	end_time = 1000
	n = 3
	m = 3
	initial_netfile(n,m,0)
	#generate_detectors(n,m)
	#generate_cfg('data/grids.sumocfg_{}'.format(file_ID), file_ID)
	#generate_route(seed, n, m,'data/grids.rou_{}.xml'.format(file_ID), end_time)
