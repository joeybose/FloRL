import numpy as np

class Q_learning(object):
	def __init__(self, n_state, n_action, alpha, gamma):
		self.n_state = n_state
		self.n_action = n_action
		self.alpha = alpha
		self.gamma = gamma
		self.Q = np.random.rand(n_state, n_action)

	def update(self, s, a, sNext, r):
		self.Q[s,a] = (1-self.alpha)*self.Q[s,a] + self.alpha * (r + self.gamma * np.max(self.Q[sNext]))

	def choose_action(self, s, temperature):
		p = np.exp(1.0/temperature * self.Q[s])
		p = p/ np.sum(p)
		action = np.random.choice(p.shape[0], 1, p = p)
		return action.reshape([])

	def get_pi(self, temperature):
		t = 1.0/temperature * self.Q
		p = np.exp(t - 0.5 * (np.amax(t)+np.amin(t)))
		p = p/np.sum(p, axis = 1)[:,None]
		return p