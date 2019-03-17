import numpy as np

def softmax(x, alpha, beta):
	exp = np.exp(alpha * x + beta)
	return exp/np.sum(exp)

class Simple_agent(object):
	def __init__(self, action_size, alpha, epsilon = 0.04):
		self.action_size = action_size
		self.alpha = alpha
		self.epsilon = epsilon

	def get_action(self, state):
		state = state.reshape(-1, 4)
		'''
		for i in range(state.shape[0]):
			if np.sum(state[i,:4]) > 1e-3:
				state[i, :4] = state[i, :4]/np.sum(state[i,:4])
			else:
				state[i, :4] += 0.25
		'''
		action = np.zeros(self.action_size)
		for i in range(self.action_size):
			if np.random.rand() < self.epsilon:
				action[i] = np.random.randint(4)
			else:
				prob = softmax(state[i,:], self.alpha, 0)
				action[i] = np.random.choice(4, p = prob)
		return action

	def log_pi(self, state, action):
		state = state.reshape(-1, 4)
		'''
		for i in range(state.shape[0]):
			if np.sum(state[i,:4]) > 1e-3:
				state[i, :4] = state[i, :4]/np.sum(state[i,:4])
			else:
				state[i, :4] += 0.25
		'''
		log_pi_action = 0.0
		for i in range(self.action_size):
			prob = softmax(state[i, :], self.alpha, 0)
			prob = (1-self.epsilon) * prob + self.epsilon * 0.25
			log_pi_action += np.log(prob[int(action[i])])
		return log_pi_action

	def pi(self, state, action):
		return np.exp(self.log_pi(state, action))

class Easy_agent(object):
	def __init__(self, action_size, alpha, beta, gamma, epsilon = 1e-4):
		self.action_size = action_size
		self.alpha = alpha
		self.beta = beta
		self.gamma = gamma
		self.epsilon = epsilon

	def get_action(self, state):
		state = state.reshape(-1,8)
		for i in range(state.shape[0]):
			if np.sum(state[i,:4]) > 1e-3:
				state[i, :4] = state[i, :4]/np.sum(state[i,:4])
			else:
				state[i, :4] += 0.25
		action = np.zeros(self.action_size)
		for i in range(self.action_size):
			if np.random.rand() < self.epsilon:
				action[i] = np.random.randint(4)
			else:
				weight_keep = state[i,4:] * self.gamma
				prob = softmax(state[i,:4], self.alpha, weight_keep + self.beta)
				action[i] = np.random.choice(4, p = prob)
		return action

	def log_pi(self, state, action):
		state = state.reshape(-1,8)
		for i in range(state.shape[0]):
			if np.sum(state[i,:4]) > 1e-3:
				state[i, :4] = state[i, :4]/np.sum(state[i,:4])
			else:
				state[i, :4] += 0.25
		log_pi_action = 0.0
		for i in range(self.action_size):
			weight_keep = state[i,4:] * self.gamma
			prob = softmax(state[i, :4], self.alpha, weight_keep + self.beta)
			prob = (1-self.epsilon) * prob + self.epsilon * 0.25
			log_pi_action += np.log(prob[int(action[i])])
		return log_pi_action

	def pi(self, state, action):
		return np.exp(self.log_pi(state, action))
