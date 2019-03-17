import numpy as np

class toy_Markov(object):
	def __init__(self, number_state, transition_matrix, initial_distribution):
		self.number_state = number_state
		self.transition_matrix = transition_matrix #np.zeros((number_state,number_state))
		self.initial_distribution = initial_distribution
		self.initial_state()

	def initial_state(self):
		self.cur_state = np.random.choice(self.number_state, p = self.initial_distribution)
	
	def next_state(self):
		self.cur_state = np.random.choice(self.number_state, p = self.transition_matrix[:,self.cur_state])

	def solve_stationary_distribution(self):
		p = self.initial_distribution
		for i in range(100):
			p = np.dot(self.transition_matrix,p)
		return p

class random_walk_2d(object):
	n_state = 0
	n_action = 4
	def __init__(self, length):
		self.length = length
		self.x = np.random.randint(length)
		self.y = np.random.randint(length)
		self.n_state = length*length

	def reset(self):
		self.x = np.random.randint(self.length)
		self.y = np.random.randint(self.length)
		return self.state_encoding()

	def state_encoding(self):
		return self.x * self.length + self.y

	def step(self, action):
		if action == 0:
			if self.x < self.length - 1:
				self.x += 1
		elif action == 1:
			if self.y < self.length - 1:
				self.y += 1
		elif action == 2:
			if self.x > 0:
				self.x -= 1
		elif action == 3:
			if self.y > 0:
				self.y -= 1
		return self.state_encoding(), 0

	def state_decoding(self, state):
		x = state / self.length
		y = state % self.length
		return x,y

class taxi(object):
	n_state = 0
	n_action = 6
	def __init__(self, length):
		self.length = length
		self.x = np.random.randint(length)
		self.y = np.random.randint(length)
		self.possible_passenger_loc = [(0,0), (0,length-1), (length-1,0), (length-1, length-1)]
		self.passenger_status = np.random.randint(16)
		self.taxi_status = 4
		self.n_state = (length**2)*16*5

	def reset(self):
		length = self.length
		self.x = np.random.randint(length)
		self.y = np.random.randint(length)
		self.possible_passenger_loc = [(0,0), (0,length-1), (length-1,0), (length-1, length-1)]
		self.passenger_status = np.random.randint(16)
		self.taxi_status = 4
		return self.state_encoding()

	def state_encoding(self):
		length = self.length
		return self.taxi_status + (self.passenger_status + (self.x * length + self.y) * 16) * 5

	def state_decoding(self, state):
		length = self.length
		taxi_status = state % 5
		state = state / 5
		passenger_status = state % 16
		state = state / 16
		y = state % length
		x = state / length
		return x,y,passenger_status,taxi_status

	def render(self):
		MAP = []
		length = self.length
		for i in range(length):
			if i == 0:
				MAP.append('-'*(3*length+1))
			MAP.append('|' + '  |' * length)
			MAP.append('-'*(3*length+1))
		MAP = np.asarray(MAP, dtype = 'c')
		if self.taxi_status == 4:
			MAP[2*self.x+1, 3*self.y+2] = 'O'
		else:
			MAP[2*self.x+1, 3*self.y+2] = '@'
		for i in range(4):
			if self.passenger_status & (1<<i):
				x,y = self.possible_passenger_loc[i]
				MAP[2*x+1, 3*y+1] = 'a'
		for line in MAP:
			print ''.join(line)
		if self.taxi_status == 4:
			print 'Empty Taxi'
		else:
			x,y = self.possible_passenger_loc[self.taxi_status]
			print 'Taxi destination:({},{})'.format(x,y)

	def step(self, action):
		reward = -1
		length = self.length
		if action == 0:
			if self.x < self.length - 1:
				self.x += 1
		elif action == 1:
			if self.y < self.length - 1:
				self.y += 1
		elif action == 2:
			if self.x > 0:
				self.x -= 1
		elif action == 3:
			if self.y > 0:
				self.y -= 1
		elif action == 4:	# Try to pick up
			for i in range(4):
				x,y = self.possible_passenger_loc[i]
				if x == self.x and y == self.y and(self.passenger_status & (1<<i)):
					# successfully pick up
					self.passenger_status -= 1<<i
					self.taxi_status = np.random.randint(4)
					while self.taxi_status == i:
						self.taxi_status = np.random.randint(4)
		elif action == 5:
			if self.taxi_status < 4:
				x,y = self.possible_passenger_loc[self.taxi_status]
				if self.x == x and self.y == y:
					reward = 20
				self.taxi_status = 4
		self.change_passenger_status()
		return self.state_encoding(), reward

	def change_passenger_status(self):
		p_generate = [0.3, 0.05, 0.1, 0.2]
		p_disappear = [0.05, 0.1, 0.1, 0.05]
		for i in range(4):
			if self.passenger_status & (1<<i):
				if np.random.rand() < p_disappear[i]:
					self.passenger_status -= 1<<i
			else:
				if np.random.rand() < p_generate[i]:
					self.passenger_status += 1<<i
	def debug(self):
		self.reset()
		while True:
			self.render()
			action = input('Action:')
			if action > 5 or action < 0:
				break
			else:
				_, reward = self.step(action)
				print reward





