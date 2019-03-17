import numpy as np
import tensorflow as tf
from time import sleep
import sys

# Hyper parameter
#Learning_rate = 1e-3
#initial_stddev = 0.5

#Training Parameter
training_batch_size = 512
training_maximum_iteration = 3001
TEST_NUM = 2000

class Density_Ratio_kernel(object):
	def __init__(self, obs_dim, w_hidden, Learning_rate, reg_weight):
		# place holder
		self.state = tf.placeholder(tf.float32, [None, obs_dim])
		self.med_dist = tf.placeholder(tf.float32, [])
		self.next_state = tf.placeholder(tf.float32, [None, obs_dim])

		self.state2 = tf.placeholder(tf.float32, [None, obs_dim])
		self.next_state2 = tf.placeholder(tf.float32, [None, obs_dim])
		self.policy_ratio = tf.placeholder(tf.float32, [None])
		self.policy_ratio2 = tf.placeholder(tf.float32, [None])
					
		# density ratio for state and next state
		w = self.state_to_w(self.state, obs_dim, w_hidden)
		w_next = self.state_to_w(self.next_state, obs_dim, w_hidden)
		w2 = self.state_to_w(self.state2, obs_dim, w_hidden)
		w_next2 = self.state_to_w(self.next_state2, obs_dim, w_hidden)
		norm_w = tf.reduce_mean(w)
		norm_w_next = tf.reduce_mean(w_next)
		norm_w_beta = tf.reduce_mean(w * self.policy_ratio)
		norm_w2 = tf.reduce_mean(w2)
		norm_w_next2 = tf.reduce_mean(w_next2)
		norm_w_beta2 = tf.reduce_mean(w2 * self.policy_ratio2)
		self.output = w

		# calculate loss function
		# x = w * self.policy_ratio - w_next
		# x2 = w2 * self.policy_ratio2 - w_next2
		# x = w * self.policy_ratio / norm_w_beta - w_next / norm_w
		# x2 = w2 * self.policy_ratio2 / norm_w_beta2 - w_next2 / norm_w2
		x = w * self.policy_ratio / norm_w - w_next / norm_w_next
		x2 = w2 * self.policy_ratio2 / norm_w2 - w_next2 / norm_w_next2

		diff_xx = tf.expand_dims(self.next_state, 1) - tf.expand_dims(self.next_state2, 0)
		K_xx = tf.exp(-tf.reduce_sum(tf.square(diff_xx), axis = -1)/(2.0*self.med_dist*self.med_dist))
		norm_K = tf.reduce_sum(K_xx)

		loss_xx = tf.matmul(tf.matmul(tf.expand_dims(x, 0),K_xx),tf.expand_dims(x2, 1))
		
		# self.loss = tf.squeeze(loss_xx)/(norm_w*norm_w2*norm_K)
		self.loss = tf.squeeze(loss_xx)/norm_K
		self.reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'w'))
		self.train_op = tf.train.AdamOptimizer(Learning_rate).minimize(self.loss + reg_weight * self.reg_loss)

		# Debug
		self.debug1 = tf.reduce_mean(w)
		self.debug2 = tf.reduce_mean(w_next)

		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

	def reset(self):
		self.sess.run(tf.global_variables_initializer())

	def close_Session(self):
		tf.reset_default_graph()
		self.sess.close()

	def state_to_w(self, state, obs_dim, hidden_dim_dr):
		with tf.variable_scope('w', reuse = tf.AUTO_REUSE):
			w = tf.ones([tf.shape(state)[0]])
			for i in range(obs_dim/4):
				w_part_i = self.state_to_w_tl(state[:, i:(i+4)], 4, hidden_dim_dr)
				w = w * w_part_i
			return w
	def state_to_w_tl(self, state, obs_dim, hidden_dim_dr):
		with tf.variable_scope('w', reuse = tf.AUTO_REUSE):
			
			# First layer
			W1 = tf.get_variable('W1', initializer = tf.random_normal(shape = [obs_dim, hidden_dim_dr]))#, regularizer = tf.contrib.layers.l2_regularizer(1.))
			b1 = tf.get_variable('b1', initializer = tf.zeros([hidden_dim_dr]))#, regularizer = tf.contrib.layers.l2_regularizer(1.))
			z1 = tf.matmul(state, W1) + b1
			mean_z1, var_z1 = tf.nn.moments(z1, [0])
			scale_z1 = tf.get_variable('scale_z1', initializer = tf.ones([hidden_dim_dr]))
			beta_z1 = tf.get_variable('beta_z1', initializer = tf.zeros([hidden_dim_dr]))
			l1 = tf.tanh(tf.nn.batch_normalization(z1, mean_z1, var_z1, beta_z1, scale_z1, 1e-10))

			# Second layer
			W2 = tf.get_variable('W2', initializer = 0.01 * tf.random_normal(shape = [hidden_dim_dr, 1]), regularizer = tf.contrib.layers.l2_regularizer(1.))
			b2 = tf.get_variable('b2', initializer = tf.zeros([1]), regularizer = tf.contrib.layers.l2_regularizer(1.))
			z2 = tf.matmul(l1, W2) + b2
			#return tf.exp(tf.squeeze(z2))
			#mean_z2, var_z2 = tf.nn.moments(z2, [0])
			#scale_z2 = tf.get_variable('scale_z2', initializer = tf.ones([1]))
			#beta_z2 = tf.get_variable('beta_z2', initializer = tf.zeros([1]))
			#l2 = tf.nn.batch_normalization(z2, mean_z2, var_z2, beta_z2, scale_z2, 1e-10)
			return tf.log(1+tf.exp(tf.squeeze(z2)))
	
	def get_density_ratio(self, states):
		return self.sess.run(self.output, feed_dict = {
			self.state : states
			})

	def train(self, SASR, policy0, policy1, batch_size = training_batch_size, max_iteration = training_maximum_iteration, test_num = TEST_NUM, fPlot = False, epsilon = 1e-3):
		S = []
		SN = []
		# POLICY_RATIO = []
		# POLICY_RATIO2 = []
		PI1 = []
		PI0 = []
		REW = []
		for sasr in SASR:
			for state, action, next_state, reward in sasr:
				# POLICY_RATIO.append((epsilon + policy1.pi(state, action))/(epsilon + policy0.pi(state, action)))
				# POLICY_RATIO2.append(policy1.pi(state, action)/policy0.pi(state, action))
				#POLICY_RATIO.append(epsilon + (1-epsilon) * policy1.pi(state, action)/policy0.pi(state, action))
				PI1.append(policy1.pi(state, action))
				PI0.append(policy0.pi(state, action))
				S.append(state)
				SN.append(next_state)
				REW.append(reward)
		# normalized
		
		S = np.array(S)
		S_max = np.max(S, axis = 0)
		S_min = np.min(S, axis = 0)
		S = (S - S_min)/(S_max - S_min)
		SN = (np.array(SN) - S_min)/(S_max - S_min)
		

		if test_num > 0:
			S_test = np.array(S[:test_num])
			SN_test = np.array(SN[:test_num])
			# POLICY_RATIO_test = np.array(POLICY_RATIO[:test_num])
			PI1_test = np.array(PI1[:test_num])
			PI0_test = np.array(PI0[:test_num])

		S = np.array(S[test_num:])
		SN = np.array(SN[test_num:])
		# POLICY_RATIO = np.array(POLICY_RATIO[test_num:])
		# POLICY_RATIO2 = np.array(POLICY_RATIO2[test_num:])
		PI1 = np.array(PI1[test_num:])
		PI0 = np.array(PI0[test_num:])
		REW = np.array(REW[test_num:])
		N = S.shape[0]

		subsamples = np.random.choice(N, 1000)
		s = S[subsamples]
		med_dist = np.median(np.sqrt(np.sum(np.square(s[None, :, :] - s[:, None, :]), axis = -1)))

		for i in range(max_iteration):
			if test_num > 0 and i % 500 == 0:
				subsamples = np.random.choice(test_num, batch_size)
				s_test = S_test[subsamples]
				sn_test = SN_test[subsamples]
				# policy_ratio_test = POLICY_RATIO_test[subsamples]
				policy_ratio_test = (PI1_test[subsamples] + epsilon)/(PI0_test[subsamples] + epsilon)

				subsamples = np.random.choice(test_num, batch_size)
				s_test2 = S_test[subsamples]
				sn_test2 = SN_test[subsamples]
				# policy_ratio_test2 = POLICY_RATIO_test[subsamples]
				policy_ratio_test2 = (PI1_test[subsamples] + epsilon)/(PI0_test[subsamples] + epsilon)

				test_loss, reg_loss, norm_w, norm_w_next = self.sess.run([self.loss, self.reg_loss, self.debug1, self.debug2], feed_dict = {
					self.med_dist: med_dist,
					self.state: s_test,
					self.next_state: sn_test,
					self.policy_ratio: policy_ratio_test,
					self.state2: s_test2,
					self.next_state2: sn_test2,
					self.policy_ratio2: policy_ratio_test2
					})
				print('----Iteration = {}-----'.format(i))
				print("Testing error = {}".format(test_loss))
				print('Regularization loss = {}'.format(reg_loss))
				print('Norm_w = {}'.format(norm_w))
				print('Norm_w_next = {}'.format(norm_w_next))
				DENR = self.get_density_ratio(S)
				# T = DENR*POLICY_RATIO2
				T = DENR*PI1/PI0
				print('DENR = {}'.format(np.sum(T*REW)/np.sum(T)))
				sys.stdout.flush()
				# epsilon *= 0.9
			
			subsamples = np.random.choice(N, batch_size)
			s = S[subsamples]
			sn = SN[subsamples]
			# policy_ratio = POLICY_RATIO[subsamples]
			policy_ratio = (PI1[subsamples] + epsilon)/(PI0[subsamples] + epsilon)

			subsamples = np.random.choice(N, batch_size)
			s2 = S[subsamples]
			sn2 = SN[subsamples]
			# policy_ratio2 = POLICY_RATIO[subsamples]
			policy_ratio2 = (PI1[subsamples] + epsilon)/(PI0[subsamples] + epsilon)
			
			self.sess.run(self.train_op, feed_dict = {
				self.med_dist: med_dist,
				self.state: s,
				self.next_state: sn,
				self.policy_ratio: policy_ratio,
				self.state2 : s2,
				self.next_state2: sn2,
				self.policy_ratio2: policy_ratio2
				})
		DENR = self.get_density_ratio(S)
		# T = DENR*POLICY_RATIO2
		T = DENR*PI1/PI0
		return np.sum(T*REW)/np.sum(T)

	def evaluate(self, SASR0, policy0, policy1):
		S = []
		POLICY_RATIO = []
		REW = []
		for sasr in SASR0:
			for state, action, next_state, reward in sasr:
				POLICY_RATIO.append(policy1.pi(state, action)/policy0.pi(state, action))
				S.append(state)
				REW.append(reward)

		S = np.array(S)
		S_max = np.max(S, axis = 0)
		S_min = np.min(S, axis = 0)
		S = (S - S_min)/(S_max - S_min)
		POLICY_RATIO = np.array(POLICY_RATIO)
		REW = np.array(REW)
		DENR = self.get_density_ratio(S)
		T = DENR*POLICY_RATIO
		return np.sum(T*REW)/np.sum(T)


class Density_Ratio_GAN(object):
	def __init__(self, obs_dim, w_hidden, f_hidden, gau, Learning_rate, reg_weight):
		# place holder
		self.state = tf.placeholder(tf.float32, [None, obs_dim])
		self.next_state = tf.placeholder(tf.float32, [None, obs_dim])
		#self.start_state = tf.placeholder(tf.float32, [None, obs_dim])
		self.policy_ratio = tf.placeholder(tf.float32, [None])
		#self.isStart = tf.placeholder(tf.float32, [None])
					
		# density ratio for state and next state
		if gau == 0:
			w = self.state_to_w_batch_norm(self.state, obs_dim, w_hidden)
			w_next = self.state_to_w_batch_norm(self.next_state, obs_dim, w_hidden)
		else:
			w = self.state_to_w_gau_mix(self.state, obs_dim, w_hidden)
			w_next = self.state_to_w_gau_mix(self.next_state, obs_dim, w_hidden)
		f = self.state_to_f(self.state, obs_dim, f_hidden)
		f_next = self.state_to_f(self.next_state, obs_dim, f_hidden)
		norm_w = tf.reduce_mean(w)
		norm_f = tf.sqrt(tf.reduce_mean(f_next * f_next)) + 1e-15
		self.output = w

		# calculate loss function
		# discounted case
		# x = (1-self.isStart) * w * self.policy_ratio + self.isStart * norm_w - w_next
		x = w * self.policy_ratio - w_next
		y = f_next * self.policy_ratio - f
		
		self.reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'w'))
		self.reg_loss_f = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'f'))
		#self.loss = tf.square(tf.reduce_mean(x*f))/tf.square(norm_w)
		#self.loss1 = tf.reduce_mean(x * f_next)/norm_w
		#self.loss2 = tf.reduce_mean(w * y) / norm_w
		self.loss = tf.reduce_mean(x * f_next) / (norm_w * norm_f)
		with tf.variable_scope('optimizer'):
			optimizer = tf.train.AdamOptimizer(Learning_rate)
			f_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'f')
			w_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'w')
			self.train_op_f = optimizer.minimize(-self.loss, var_list = f_vars)
			self.train_op_w = optimizer.minimize( self.loss + reg_weight * self.reg_loss, var_list = w_vars)

		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

	def reset(self):
		self.sess.run(tf.global_variables_initializer())

	def close_Session(self):
		tf.reset_default_graph()
		self.sess.close()

	def state_to_w_gau_mix(self, state, obs_dim, num_component):
		with tf.variable_scope('w', reuse = tf.AUTO_REUSE):
			mean_state, var_state = tf.nn.moments(state,[0])
			state = (state - mean_state) / tf.sqrt(var_state)
			initial_mu = tf.random_normal(shape = [num_component, obs_dim], stddev = np.sqrt(1.))
			dpi = self.gaussian_mixture(state, obs_dim, num_component, 0.0, 'dpi', initial_mu)
			dpi0 = self.gaussian_mixture(state, obs_dim, num_component, 0.0, 'dpi0', initial_mu)
			return (dpi+1e-15)/(dpi0+1e-15)

	def gaussian_mixture(self, state, obs_dim, num_component, std_min, scope, initial_mu):
		with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
			#alpha = tf.get_variable('alpha', initializer = tf.random_normal(
			#	shape = [num_component], stddev = np.sqrt(1./num_component)))
			alpha = tf.get_variable('alpha', initializer = tf.zeros([num_component]))
			mu = tf.get_variable('mu', initializer = initial_mu)
			#sigma = tf.get_variable('sigma', initializer = tf.random_normal(
			#	shape = [num_component], stddev = np.sqrt(1.)))
			log_sigma = tf.get_variable('log_sigma', initializer = tf.zeros([num_component]))
			log_prob = alpha-tf.reduce_sum(tf.square(tf.expand_dims(state, 1) - tf.expand_dims(mu, 0)), axis = -1)/(20 * tf.exp(log_sigma) + std_min)
			prob = tf.reduce_mean(tf.exp(log_prob), axis = -1)
			return tf.squeeze(prob)

	def state_to_w_quadratic(self, state, obs_dim, hidden_dim_dr):
		with tf.variable_scope('w', reuse = tf.AUTO_REUSE):
			W = tf.get_variable('W1', ini)
	def state_to_w_batch_norm(self, state, obs_dim, hidden_dim_dr):
		'''
		obs_dim = obs_dim / 2
		w = np.zeros((obs_dim * 2, obs_dim))
		for i in range(obs_dim):
			w[i,i] = 1
			w[i+obs_dim, i] = 0.1
		W = tf.constant(w, dtype = tf.float32)
		state = tf.matmul(state, W)
		'''
		with tf.variable_scope('w', reuse = tf.AUTO_REUSE):
			# Normalized
			#mean_state, var_state = tf.nn.moments(state,[0])
			#state = (state - mean_state) / tf.sqrt(var_state + 1e-10)
			
			# First layer
			W1 = tf.get_variable('W1', initializer = tf.random_normal(shape = [obs_dim, hidden_dim_dr]))#, regularizer = tf.contrib.layers.l2_regularizer(1.))
			b1 = tf.get_variable('b1', initializer = tf.zeros([hidden_dim_dr]))#, regularizer = tf.contrib.layers.l2_regularizer(1.))
			z1 = tf.matmul(state, W1) + b1
			mean_z1, var_z1 = tf.nn.moments(z1, [0])
			scale_z1 = tf.get_variable('scale_z1', initializer = tf.ones([hidden_dim_dr]))
			beta_z1 = tf.get_variable('beta_z1', initializer = tf.zeros([hidden_dim_dr]))
			l1 = tf.tanh(tf.nn.batch_normalization(z1, mean_z1, var_z1, beta_z1, scale_z1, 1e-10))
			#l1 = tf.nn.dropout(l1, 0.5)

			# Second layer
			W2 = tf.get_variable('W2', initializer = 0.1 * tf.random_normal(shape = [hidden_dim_dr, 1]), regularizer = tf.contrib.layers.l2_regularizer(1.))
			b2 = tf.get_variable('b2', initializer = tf.zeros([1]), regularizer = tf.contrib.layers.l2_regularizer(1.))
			z2 = tf.matmul(l1, W2) + b2
			#return tf.exp(tf.squeeze(z2))
			return tf.log(1+tf.exp(tf.squeeze(z2)))
			'''
			mean_z2, var_z2 = tf.nn.moments(z2, [0])
			scale_z2 = tf.get_variable('scale_z2', initializer = tf.ones([20]))	#initial want to close to 0
			beta_z2 = tf.get_variable('beta_z2', initializer = tf.zeros([20]))
			l2 = tf.tanh(tf.nn.batch_normalization(z2, mean_z2, var_z2, beta_z2, scale_z2, 1e-10))
			#l2 = tf.tanh(0.01 * z2)

			W3 = tf.get_variable('W3', initializer = 0.1 * tf.random_normal(shape = [20, 1]), regularizer = tf.contrib.layers.l2_regularizer(1.))
			b3 = tf.get_variable('b3', initializer = tf.zeros([1]), regularizer = tf.contrib.layers.l2_regularizer(1.))
			z3 = tf.matmul(l2, W3) + b3
			'''

			#return tf.exp(tf.squeeze(z3))
			#return tf.log(1 + tf.exp(tf.squeeze(z3)))

			#alpha = tf.get_variable('alpha', initializer = tf.constant(5.))
			#return tf.exp(alpha * tf.squeeze(l2))
	def state_to_w(self, state, obs_dim, hidden_dim_dr):
		with tf.variable_scope('w', reuse = tf.AUTO_REUSE):
			# Normalized
			#mean_state, var_state = tf.nn.moments(state,[0])
			#state = (state - mean_state) / tf.sqrt(var_state)
			
			# First layer
			W1 = tf.get_variable('W1', initializer = 0.5 * tf.random_normal(shape = [obs_dim, hidden_dim_dr]), regularizer = tf.contrib.layers.l2_regularizer(1.))
			b1 = tf.get_variable('b1', initializer = tf.zeros([hidden_dim_dr]), regularizer = tf.contrib.layers.l2_regularizer(1.))
			z1 = tf.matmul(state, W1) + b1
			l1 = tf.tanh(z1)

			# Second layer
			W2 = tf.get_variable('W2', initializer = 0.5 * tf.random_normal(shape = [hidden_dim_dr, 1]), regularizer = tf.contrib.layers.l2_regularizer(1.))
			b2 = tf.get_variable('b2', initializer = tf.zeros([1]), regularizer = tf.contrib.layers.l2_regularizer(1.))
			z2 = tf.matmul(l1, W2) + b2
			#return tf.exp(tf.squeeze(z2))
			return tf.log(1+tf.exp(tf.squeeze(z2)))

	def state_to_f(self, state, obs_dim, hidden_dim_dr):
		'''
		obs_dim = obs_dim / 2
		w = np.zeros((obs_dim * 2, obs_dim))
		for i in range(obs_dim):
			w[i,i] = 1
			w[i+obs_dim, i] = 0.1
		W = tf.constant(w, dtype = tf.float32)
		state = tf.matmul(state, W)
		'''
		with tf.variable_scope('f', reuse = tf.AUTO_REUSE):
			#mean_state, var_state = tf.nn.moments(state,[0])
			#state = (state - mean_state) / tf.sqrt(var_state)

			W4 = tf.get_variable('W4', initializer = tf.random_normal(shape = [obs_dim, hidden_dim_dr]), regularizer = tf.contrib.layers.l2_regularizer(1.))
			b4 = tf.get_variable('b4', initializer = tf.zeros([hidden_dim_dr]), regularizer = tf.contrib.layers.l2_regularizer(1.))
			z1 = tf.matmul(state, W4) + b4
			mean_z1, var_z1 = tf.nn.moments(z1, [0])
			scale_z1 = tf.get_variable('scale_z1', initializer = tf.ones([hidden_dim_dr]))
			beta_z1 = tf.get_variable('beta_z1', initializer = tf.zeros([hidden_dim_dr]))
			l1 = tf.tanh(tf.nn.batch_normalization(z1, mean_z1, var_z1, beta_z1, scale_z1, 1e-10))
			#l1 = tf.tanh(z1)

			W5 = tf.get_variable('W5', initializer = tf.random_normal(shape = [hidden_dim_dr, 1]), regularizer = tf.contrib.layers.l2_regularizer(1.))
			b5 = tf.get_variable('b5', initializer = tf.zeros([1]), regularizer = tf.contrib.layers.l2_regularizer(1.))
			z2 = tf.matmul(l1, W5) + b5
			'''
			mean_z2, var_z2 = tf.nn.moments(z2, [0])
			scale_z2 = tf.get_variable('scale_z2', initializer = tf.ones([20]))
			beta_z2 = tf.get_variable('beta_z2', initializer = tf.zeros([20]))
			l2 = tf.tanh(tf.nn.batch_normalization(z2, mean_z2, var_z2, beta_z2, scale_z2, 1e-10))
			#W3 = tf.clip_by_value(W3, -1.0, 1.0)
			#b3 = tf.clip_by_value(b3, -1.0, 1.0)
			#W4 = tf.clip_by_value(W4, -1.0, 1.0)
			#b4 = tf.clip_by_value(b4, -1.0, 1.0)
			W6 = tf.get_variable('W6', initializer = tf.random_normal(shape = [20, 1]), regularizer = tf.contrib.layers.l2_regularizer(1.))
			b6 = tf.get_variable('b6', initializer = tf.zeros([1]), regularizer = tf.contrib.layers.l2_regularizer(1.))
			z3 = tf.matmul(l2, W6) + b6
			'''
			return tf.squeeze(z2)

	def get_density_ratio(self, states):
		return self.sess.run(self.output, feed_dict = {
			self.state : states
			})
		
	def train(self, SASR, policy0, policy1, batch_size = training_batch_size, max_iteration = training_maximum_iteration, test_num = TEST_NUM, fPlot = False):
		S = []
		SN = []
		POLICY_RATIO = []
		REW = []
		for sasr in SASR:
			for state, action, next_state, reward in sasr:
				S.append(state)
				SN.append(next_state)
				POLICY_RATIO.append(policy1.pi(state, action)/policy0.pi(state, action))
				REW.append(reward)
		if test_num > 0:
			S_test = np.array(S[:test_num])
			SN_test = np.array(SN[:test_num])
			POLICY_RATIO_test = np.array(POLICY_RATIO[:test_num])

		S = np.array(S[test_num:])
		SN = np.array(SN[test_num:])
		POLICY_RATIO = np.array(POLICY_RATIO[test_num:])
		REW = np.array(REW[test_num:])

		N = S.shape[0]
		for i in range(max_iteration):
			if test_num > 0 and i % 500 == 0:
				test_loss, test_reg_loss = self.sess.run([self.loss, self.reg_loss], feed_dict = {
					self.state: S_test,
					self.next_state: SN_test,
					self.policy_ratio: POLICY_RATIO_test
					})
				print ("Testing loss in iteration {} = {}".format(i, test_loss))
				print ('Regularization loss = {}'.format(test_reg_loss))
				DENR = self.get_density_ratio(S)
				T = DENR*POLICY_RATIO
				print('DENR = {}'.format(np.sum(T*REW)/np.sum(T)))
				sys.stdout.flush()
			
			subsamples = np.random.choice(N, batch_size)
			s = S[subsamples]
			sn = SN[subsamples]
			policy_ratio = POLICY_RATIO[subsamples]	
			for t in range(10):
				self.sess.run(self.train_op_f, feed_dict = {
					self.state: s,
					self.next_state: sn,
					self.policy_ratio: policy_ratio
					})
			
			self.sess.run(self.train_op_w, feed_dict = {
				self.state: s,
				self.next_state: sn,
				self.policy_ratio: policy_ratio
				})
		#if test_num == 0:
		

	def evaluate(self, SASR, policy0, policy1):
		S = []
		REW = []
		policy_ratio = []
		for sasr in SASR:
			for state, action, next_state, reward in sasr:
				S.append(state)
				REW.append(reward)
				policy_ratio.append(policy1.pi(state, action)/policy0.pi(state, action))
		w = self.get_density_ratio(np.array(S))
		policy_ratio = np.array(policy_ratio)
		REW = np.array(REW)
		T = w*policy_ratio
		return np.sum(T*REW)/np.sum(T)

	def learning_distribution(self, S0, S1, S11, policy0, policy1, dim_ID):
		w = self.get_density_ratio(np.array(S0))
		#print(w)
		#print(w[np.isnan(w)])
		# Only consider the marginal distribution of the first feature of state
		N = int(max(np.amax(S0[:,dim_ID]), np.amax(S1[:,dim_ID]), np.amax(S11[:, dim_ID]))) + 1
		
		p1_true = np.zeros(N, dtype = np.float32)
		p1_true_copy = np.zeros(N, dtype = np.float32)
		p1_estimate = np.zeros(N, dtype = np.float32)
		p0_true = np.zeros(N, dtype = np.float32)

		for s in S1:
			index = int(s[dim_ID])
			p1_true[index] += 1.0

		for s in S11:
			index = int(s[dim_ID])
			p1_true_copy[index] += 1.0

		for wi,s in zip(w,S0):
			index = int(s[dim_ID])
			p0_true[index] += 1.0
			p1_estimate[index] += wi
		return p1_true/np.sum(p1_true), p1_estimate/np.sum(p1_estimate), p0_true/np.sum(p0_true), p1_true_copy/np.sum(p1_true_copy)
