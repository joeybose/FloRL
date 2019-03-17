# Author : Ziyang Tang
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import argparse
import optparse
import subprocess
import random
import numpy as np
from time import sleep
from generate_network import generate_route, generate_detectors, generate_cfg, initial_netfile
from Easy_agent import Easy_agent, Simple_agent
try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
    from sumolib import checkBinary  # noqa
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

import traci

def get_state(detector_IDs, TL_IDs):
	state = np.zeros((len(TL_IDs), 8),dtype = np.float32)
	for i, tl in zip(range(len(TL_IDs)), TL_IDs):
		for detector in detector_IDs:
			if detector.startswith(tl):
				if detector[len(tl)+5] != '0':
					continue
				# East-West direction
				if detector[len(tl)+1] == '0' or detector[len(tl)+1] == '2':
					# left turn lane
					if detector[len(tl)+3] == '3':
						state[i, 3] += traci.lanearea.getLastStepVehicleNumber(detector)
					else:
						state[i, 2] += traci.lanearea.getLastStepVehicleNumber(detector)
				else:
					if detector[len(tl)+3] == '3':
						state[i, 1] += traci.lanearea.getLastStepVehicleNumber(detector)
					else:
						state[i, 0] += traci.lanearea.getLastStepVehicleNumber(detector)
		tl_phase = traci.trafficlight.getPhase(tl)
		if tl_phase % 2 == 1:
			tl_phase -= 1
		state[i, 4 + tl_phase/2] = 1
	return state

def get_simple_state(detector_IDs, TL_IDs):
	state = np.zeros((len(TL_IDs), 4), dtype = np.float32)
	for i, tl in zip(range(len(TL_IDs)), TL_IDs):
		for detector in detector_IDs:
			if detector.startswith(tl):
				if detector[len(tl)+5] != '0':
					continue
				# East-West direction
				if detector[len(tl)+1] == '0' or detector[len(tl)+1] == '2':
					# left turn lane
					if detector[len(tl)+3] == '3':
						state[i, 3] += traci.lanearea.getLastStepVehicleNumber(detector)
					else:
						state[i, 2] += traci.lanearea.getLastStepVehicleNumber(detector)
				else:
					if detector[len(tl)+3] == '3':
						state[i, 1] += traci.lanearea.getLastStepVehicleNumber(detector)
					else:
						state[i, 0] += traci.lanearea.getLastStepVehicleNumber(detector)
	return state

def get_reward(detector_IDs):
	reward = 0.
	for detector in detector_IDs:
		reward -= traci.lanearea.getLastStepVehicleNumber(detector)
	return reward

def smooth_change_light(TL_IDs, action):
	current_light_phase = np.zeros(len(TL_IDs), dtype = int)
	for i, tl in zip(range(len(TL_IDs)), TL_IDs):
		current_light_phase[i] = traci.trafficlight.getPhase(tl)
	for t in range(3):
		for i, tl in zip(range(len(TL_IDs)), TL_IDs):
			if current_light_phase[i] % 2 == 1:
				current_light_phase[i] -= 1
				print('Amazing!')
			if action[i] * 2 == current_light_phase[i]:
				traci.trafficlight.setPhase(tl, current_light_phase[i])
			else:
				traci.trafficlight.setPhase(tl, current_light_phase[i] + 1)
		traci.simulationStep()

	for t in range(3):
		for i, tl in zip(range(len(TL_IDs)), TL_IDs):
			if current_light_phase[i] % 2 == 1:
				current_light_phase[i] -= 1
			traci.trafficlight.setPhase(tl, action[i] * 2)
		traci.simulationStep()
def SASR_encoding(state, action, next_state, reward):
	return np.hstack((state.reshape(-1), action, next_state.reshape(-1), reward))

def rollout(truncation_size, filename, agent = None, simple = True):
	TL_IDs = traci.trafficlight.getIDList()
	#print(TL_IDs)
	detector_IDs = traci.lanearea.getIDList()
	#print(detector_IDs)
	total_reward = 0.0
	
	if simple:
		state_dim = len(TL_IDs) * 4
	else:
		state_dim = len(TL_IDs) * 8
	action_dim = len(TL_IDs)
	total_dim = 2 * state_dim + action_dim + 1
	sasr = np.zeros((truncation_size, total_dim), dtype = np.float32)

	# Let cars come
	for i in range(100):
		traci.simulationStep()
	if simple:
		state = get_simple_state(detector_IDs, TL_IDs)
	else:
		state = get_state(detector_IDs, TL_IDs)
	step = 0
	while traci.simulation.getMinExpectedNumber() > 0 and step < truncation_size:
		if agent == None:
			action = np.random.randint(4, size = len(TL_IDs))
		else:
			action = agent.get_action(state)
		
		smooth_change_light(TL_IDs, action)
		
		if simple:
			next_state = get_simple_state(detector_IDs, TL_IDs)
		else:
			next_state = get_state(detector_IDs, TL_IDs)
		reward = get_reward(detector_IDs)

		sasr[step, :] = SASR_encoding(state, action, next_state, reward)
		state = next_state
		total_reward += reward
		#if step % 10 == 0:
			#print('step = {} with state = {}, action = {}, reward = {}'.format(step, state, action, reward))
		step += 1
	print ("avr_reward = {}".format(total_reward/ step))
	if filename != None:
		np.save(filename, sasr)
	return total_reward/step
	#return sasr

def search_parameter(n_iter, batch_size, n_elite, n, m, command, simple = True):
	n_tl = (n-2)*(m-2)
	if simple:
		th_mean = np.zeros(1, dtype = np.float32)
	else:
		th_mean = np.zeros(12, dtype = np.float32)
		th_mean[:4] += 1.0
	th_std = np.ones_like(th_mean)
	if simple:
		th_mean_store = np.zeros((n_iter, 1))
	else:
		th_mean_store = np.zeros((n_iter, 12))
	fs_store = np.zeros(n_iter)
	for _ in range(n_iter):	
		print('-------New iteration {} --------'.format(_))
		ths = np.array([th_mean + dth for dth in  th_std[None,:]*np.random.randn(batch_size, th_mean.size)])
		fs = np.zeros((batch_size), dtype = np.float32)
		for i in range(batch_size):
			if simple:
				alpha = ths[i, 0]
				agent = Simple_agent(n_tl, alpha)
				print('alpha = {}'.format(alpha))
			else:
				alpha = ths[i,:4]
				beta = ths[i,4:8]
				gamma = ths[i,8:]
				agent = Easy_agent(n_tl, alpha, beta, gamma)
				print('alpha = {}, beta = {}, gamma = {}'.format(alpha, beta, gamma))
			sys.stdout.flush()
			traci.start(command)
			fs[i] = rollout(400, None, agent = agent)
			traci.close()
		elite_inds = fs.argsort()[::-1][:n_elite]
		elite_th = ths[elite_inds]
		th_mean = elite_th.mean(axis = 0)
		th_std = elite_th.std(axis = 0)
		th_mean_store[_, :] = th_mean
		fs_store[_] = np.max(fs)
	for _ in range(n_iter):
		print('iteration = {},\n theta = {},\n max_score = {}'.format(_, th_mean_store[_,:], fs_store[_]))
	print('th_mean = {}, th_std = {}'.format(th_mean, th_std))
	if simple:
		np.save('policy/simple_agent_parameter_{}*{}.npy'.format(n, m), th_mean_store)
	else:
		np.save('policy/agent_parameter_{}*{}.npy'.format(n, m), th_mean_store)


if __name__ == '__main__':
	n = 3
	m = 3
	net_file_ID = 0 	# we can change this ID with different number of n,m
	n_tl = (n-2)*(m-2)	# number of traffic light
	# initial_netfile(n, m, net_file_ID)
	generate_detectors(n, m, net_file_ID)
	
	parser = argparse.ArgumentParser(description='Create SARS file for SUMO environment')
	parser.add_argument('file_ID', type = int)
	parser.add_argument('repeat_time', type = int)
	args = parser.parse_args()

	file_ID = args.file_ID
	repeat_time = args.repeat_time
	seed_base = file_ID * repeat_time
	truncation_size = 1000
	end_time = 6 * truncation_size + 500
	simple = True

	generate_cfg('data/grids.sumocfg_{}'.format(file_ID), net_file_ID, file_ID)
	#generate_route(seed_base, n, m,'data/grids.rou_{}.xml'.format(file_ID), end_time)
	sumoBinary = checkBinary('sumo')
	command = [sumoBinary, "-c", "data/grids.sumocfg_{}".format(file_ID)]

	theta = np.array([[0.1], [0.2], [0.4], [0.6], [0.8], [1.0]])
	#print(theta)
	for i in range(repeat_time):
		print('----iteration {}-------'.format(i))
		print('---- seed = {} -----'.format(seed_base + i))
		sys.stdout.flush()
		generate_route(seed_base + i, n, m,'data/grids.rou_{}.xml'.format(file_ID), end_time)
		sleep(1)
		for j in range(6):
			if simple:
				agent = Simple_agent(n_tl, theta[j,:])
				filename = 'SASR_data/SASR{}_{}.npy'.format(j, seed_base + i)
			else:
				agent = Easy_agent(n_tl, theta[j,:4], theta[j,4:8], theta[j,8:])
				filename = 'SASR_data/SASR{}_{}.npy'.format(j, seed_base + i)
			sys.stdout.flush()
			traci.start(command)
			rollout(truncation_size, filename, agent = agent, simple = simple)
			traci.close()
			
	
	