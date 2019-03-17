# Author : Ziyang Tang
import sys
import os
import argparse
import numpy as np
from Density_ratio_continuous import Density_Ratio_GAN, Density_Ratio_kernel
from Easy_agent import Easy_agent, Simple_agent

def SASR_encoding(state, action, next_state, reward):
	return np.hstack((state.reshape(-1), action, next_state.reshape(-1), reward))

def SASR_decoding(sasr):
	n_tl = (sasr.shape[0]-1)/17
	state = sasr[0:n_tl*8]
	action = sasr[n_tl*8 : n_tl*9]
	next_state = sasr[n_tl*9: n_tl*17]
	reward = sasr[-1]
	return state, action, next_state, reward

def SASR_decoding2(sasr):
	n_tl = (sasr.shape[0]-1)/9
	state = sasr[0:n_tl*4]
	action = sasr[n_tl*4 : n_tl*5]
	next_state = sasr[n_tl*5: n_tl*9]
	reward = sasr[-1]
	return state, action, next_state, reward

def load_file(filename_prescript, trajectory_IDs, truncaton_size, simple = True):
	SASR = []
	for i in trajectory_IDs:
		sasr_encoded = np.load(filename_prescript+"_{}.npy".format(i))
		sasr = []
		for piece in sasr_encoded[:truncaton_size]:
			if simple:
				state, action, next_state, reward = SASR_decoding2(piece)
			else:
				state, action, next_state, reward = SASR_decoding(piece)
			sasr.append((state, action, next_state, reward))
		SASR.append(sasr)
	return SASR

def load_file_batch(filename_prescript, startPoint_IDs, nt, truncaton_size, simple = True):
	SASR = []
	for i in startPoint_IDs:
		for z in range(nt):
			sasr_encoded = np.load(filename_prescript+"_{}.npy".format(i+z))
			sasr = []
			for piece in sasr_encoded[:truncaton_size]:
				if simple:
					state, action, next_state, reward = SASR_decoding2(piece)
				else:
					state, action, next_state, reward = SASR_decoding(piece)
				sasr.append((state, action, next_state, reward))
			SASR.append(sasr)
	return SASR
def load_file_state(filename_prescript, trajectory_IDs, truncaton_size, simple = True):
	S = []
	for i in trajectory_IDs:
		sasr_encoded = np.load(filename_prescript+"_{}.npy".format(i))
		for piece in sasr_encoded[:truncaton_size]:
			if simple:
				state, action, next_state, reward = SASR_decoding2(piece)
			else:
				state, action, next_state, reward = SASR_decoding(piece)
			S.append(state)
	return np.array(S)

def on_policy_estimate(SASR):
	total_reward = 0.0
	num_SASR = 0
	for sasr in SASR:
		for state, action, next_state, reward in sasr:
			total_reward += reward
			num_SASR += 1
	return total_reward/num_SASR

def importance_sampling_estimator(SASR, policy0, policy1):
	mean_est_reward = 0.0
	for sasr in SASR:
		log_trajectory_ratio = 0.0
		total_reward = 0.0
		for state, action, next_state, reward in sasr:
			log_trajectory_ratio += policy1.log_pi(state, action) - policy0.log_pi(state, action)
			total_reward += reward
		avr_reward = total_reward / len(sasr)
		mean_est_reward += avr_reward * np.exp(log_trajectory_ratio)
	mean_est_reward /= len(SASR)
	return mean_est_reward

def importance_sampling_estimator_stepwise(SASR, policy0, policy1):
	mean_est_reward = 0.0
	for sasr in SASR:
		step_log_pr = 0.0
		est_reward = 0.0
		for state, action, next_state, reward in sasr:
			step_log_pr += policy1.log_pi(state, action) - policy0.log_pi(state, action)
			est_reward += np.exp(step_log_pr)*reward
		est_reward /= len(sasr)
		mean_est_reward += est_reward
	mean_est_reward /= len(SASR)
	return mean_est_reward

def weighted_importance_sampling_estimator(SASR, policy0, policy1):
	total_rho = 0.0
	est_reward = 0.0
	for sasr in SASR:
		total_reward = 0.0
		log_trajectory_ratio = 0.0
		for state, action, next_state, reward in sasr:
			log_trajectory_ratio += policy1.log_pi(state, action) - policy0.log_pi(state, action)
			total_reward += reward
		avr_reward = total_reward / len(sasr)
		#print ('------')
		#print (log_trajectory_ratio)
		#print (avr_reward)
		trajectory_ratio = np.exp(log_trajectory_ratio)
		total_rho += trajectory_ratio
		est_reward += trajectory_ratio * avr_reward

	return est_reward / (total_rho + 1e-300)

def weighted_importance_sampling_estimator_stepwise(SASR, policy0, policy1):
	Log_policy_ratio = []
	REW = []
	for sasr in SASR:
		log_policy_ratio = []
		rew = []
		for state, action, next_state, reward in sasr:
			log_pr = policy1.log_pi(state, action) - policy0.log_pi(state, action)
			if log_policy_ratio:
				log_policy_ratio.append(log_pr + log_policy_ratio[-1])
			else:
				log_policy_ratio.append(log_pr)
			rew.append(reward)
		Log_policy_ratio.append(log_policy_ratio)
		REW.append(rew)
	est_reward = 0.0
	rho = np.exp(Log_policy_ratio)
	#print 'rho shape = {}'.format(rho.shape)
	REW = np.array(REW)
	for i in range(REW.shape[0]):
		#est_reward += np.mean(rho[i]/np.maximum(np.mean(rho, axis = 0), 0 )* REW[i])
		est_reward += np.mean(rho[i]/(np.mean(rho, axis = 0) + 1e-300)* REW[i])
	return est_reward/REW.shape[0]

def run_evaluate(n_tl, SASR0, SASR1, pi0, pi1, w_hidden = 30, f_hidden = 60, gau = 0, epsilon = 0.1, Learning_rate = 1e-3, reg_weight = 3e-3, simple = True):

	est_on_policy = on_policy_estimate(SASR1)
	est_naive_average = on_policy_estimate(SASR0)
	print("est_on_policy = {}".format(est_on_policy))
	print("est_naive_average = {}".format(est_naive_average))

	est_IST = importance_sampling_estimator(SASR0, pi0, pi1)
	est_ISS = importance_sampling_estimator_stepwise(SASR0, pi0, pi1)
	print("est_IST = {}".format(est_IST))
	print("est_ISS = {}".format(est_ISS))

	est_WIST = weighted_importance_sampling_estimator(SASR0, pi0, pi1)
	est_WISS = weighted_importance_sampling_estimator_stepwise(SASR0, pi0, pi1)
	print("est_WIST = {}".format(est_WIST))
	print("est_WISS = {}".format(est_WISS))
	
	if simple:
		den_ratio = Density_Ratio_GAN(n_tl * 4, w_hidden, f_hidden, gau, Learning_rate, reg_weight)
	else:
		den_ratio = Density_Ratio_GAN(n_tl * 8, w_hidden, f_hidden, gau, Learning_rate, reg_weight)
	den_ratio = Density_Ratio_kernel(n_tl * 4, w_hidden, Learning_rate, reg_weight)
	est_DENR = den_ratio.train(SASR0, pi0, pi1, epsilon = epsilon)
	est_DENR = den_ratio.evaluate(SASR0, pi0, pi1)
	den_ratio.close_Session()
	print("est_DENR = {}".format(est_DENR))

	est_Model_based = 0.0
	return est_on_policy, est_DENR, est_naive_average, est_IST, est_ISS, est_WIST, est_WISS, est_Model_based

if __name__ == "__main__":
	n = 3
	m = 3
	n_tl = (n-2)*(m-2)

	num_trajectory = 250
	truncaton_size = 400
	BP = 2

	parser = argparse.ArgumentParser(description='evaluate SARS file for SUMO environment')
	parser.add_argument('--nt', type = int, required = False, default = num_trajectory)
	parser.add_argument('--ts', type = int, required = False, default = truncate_size)
	parser.add_argument('--bp', type = int, required = False, default = BP)
	args = parser.parse_args()

	theta = np.array([[0.1], [0.2], [0.4], [0.6], [0.8], [1.0]])
	agent_target = Simple_agent(n_tl, theta[5, :])
	
	filename_prescript = 'SASR_data/SASR'

	nt = args.nt 	#number of trajectory
	ts = args.ts 	#truncated size
	bp = args.bp 	#behavior policy ID
	
	agent0 = Simple_agent(n_tl, theta[bp, :])	#Behavior policy
	agent0_filename_prescript = filename_prescript+'{}'.format(bp)
	agent1_filename_prescript = filename_prescript+'5'

	startPoints = range(0, 3000, 300)

	print("num_trajectory = {}, truncaton_size = {}, BP = {}".format(nt, ts, bp))
	res = np.zeros((8, len(startPoints)))
	print("*****Start Evaluate*****")

	for i in range(len(startPoints)):
		print("=====Repeat time = {} =====".format(i))
		SASR0 = load_file(agent0_filename_prescript, range(startPoints[i], startPoints[i]+nt), ts)
		SASR1 = load_file(agent1_filename_prescript, range(startPoints[i], startPoints[i]+nt), ts)

		res[:, i] = run_evaluate(n_tl, SASR0, SASR1, agent0, agent_target)
		den_ratio2 = Density_Ratio_kernel(n_tl * 4, 30, 1e-3, 3e-3)
		den_ratio2.train(SASR0, agent0, agent_target, epsilon = 0.1)
		res[1, i] = den_ratio2.evaluate(SASR0, agent0, agent_target)
		den_ratio2.close_Session()
		print('res = {}'.format(res[:,i]))
		sys.stdout.flush()
	np.save('evaluate_result/NT={}_TS={}_BP={}.npy'.format(nt, ts, bp), res)
	
	
	
