import numpy as np
import random
import os
import time
import json

# Code based on: 
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

# Simple replay buffer
class ReplayBuffer(object):
	def __init__(self):
		self.storage = []

	# Expects tuples of (state, next_state, action, reward, done)
	def add(self, data):
		self.storage.append(data)

	def sample(self, batch_size=100):
		ind = np.random.randint(0, len(self.storage), size=batch_size)
		x, y, u, r, d = [], [], [], [], []

		for i in ind: 
			X, Y, U, R, D = self.storage[i]
			x.append(np.array(X, copy=False))
			y.append(np.array(Y, copy=False))
			u.append(np.array(U, copy=False))
			r.append(np.array(R, copy=False))
			d.append(np.array(D, copy=False))

		return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


create_folder = lambda f: [ os.makedirs(f) if not os.path.exists(f) else False ]
class Logger(object):
      def __init__(self, experiment_name='', environment_name='', folder='./results' ):
            """
            Saves experimental metrics for use later.
            :param experiment_name: name of the experiment
            :param folder: location to save data
            : param environment_name: name of the environment
            """
            self.rewards = []
            self.save_folder = os.path.join(folder, experiment_name, environment_name, time.strftime('%y-%m-%d-%H-%M-%s'))
            create_folder(self.save_folder)


      def record_reward(self, reward_return):
            self.returns_eval = reward_return

      def training_record_reward(self, reward_return):
            self.returns_train = reward_return

      def save(self):
            np.save(os.path.join(self.save_folder, "returns_eval.npy"), self.returns_eval)

      def save_2(self):
            np.save(os.path.join(self.save_folder, "returns_train.npy"), self.returns_train)


      def save_args(self, args):
            """
            Save the command line arguments
            """
            with open(os.path.join(self.save_folder, 'params.json'), 'w') as f:
                  json.dump(dict(args._get_kwargs()), f)