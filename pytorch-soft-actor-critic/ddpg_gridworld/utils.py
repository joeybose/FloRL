import gym
import numpy as np
import torch
from datetime import datetime
import sys


import numpy as np
import random
import os
import time
import json
import torch
import torch.nn as nn


from collections import namedtuple
#from envs.double_pendulum_env_x import DoublePendulumEnvX
#from envs.cartpole_swingup_env_x import CartpoleSwingupEnvX
#from envs.half_cheetah_env_x import HalfCheetahEnvX, HalfCheetahEnvXLow1, HalfCheetahEnvXLow2, HalfCheetahEnvXHigh1
#from envs.mountain_car_env_x import MountainCarEnvX
#from rllab.misc import ext

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state',
                                       'done',))


class Memory(object):
    def __init__(self,
                 max_size = 5000):
        """
        can optimize further by using a numpy array and allocating it to zero
        """
        self.max_size = max_size
        self.store = [None] * self.max_size  # is a list, other possible data structures might be a queue
        self.count = 0
        self.current = 0


    def add(self, transition):
        """ insert one sample at a time """

        self.store[self.current] = transition

        # for taking care of how many total transitions have been inserted into the memory
        self.count = max(self.count, self.current + 1)

        # increase the counter
        self.current = (self.current + 1) % self.max_size

    def get_sample(self, index):
        # normalize index
        index = index % self.count

        return self.store[index]

    def get_minibatch(self, batch_size = 100):
        """
        a minibatch of random transitions without repetition
        """
        ind = np.random.randint(0, self.count, size=batch_size)
        samples = []

        for index in ind:
            samples.append(self.get_sample(index))

        return samples


def create_env(env_name = 'Swimmer-v1', init_seed=0):
    """
    create a copy of the environment and set seed
    """
    env = gym.make(env_name)
    env.seed(init_seed)

    return env


def log(msg):
    print("[%s]\t%s" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg))
    sys.stdout.flush()


def soft_update(target, source, tau):
    """
    do the soft parameter update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def create_rllab_env(env_name, init_seed):
    """
    create the rllab env
    """
    env = eval(env_name)()
    ext.set_seed(init_seed)
    return env




create_folder = lambda f: [ os.makedirs(f) if not os.path.exists(f) else False ]
class Logger(object):
      def __init__(self, experiment_name='', environment_name='', seed='', folder='./results' ):
            """
            Saves experimental metrics for use later.
            :param experiment_name: name of the experiment
            :param folder: location to save data
            : param environment_name: name of the environment
            """
            self.rewards = []
            self.save_folder = os.path.join(folder, experiment_name, environment_name, seed)
            create_folder(self.save_folder)


      def record_reward(self, reward_return):
            self.returns_eval = reward_return

      def record_train_reward(self, reward_return):
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


