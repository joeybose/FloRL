
import math
import numpy as np
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils
from torch.distributions import Normal

# NOTE: no batch norm

class StateDistributionGaussianVAE(nn.Module):
    def __init__(self, state_dim, latent_size):
        super(StateDistributionGaussianVAE, self).__init__()
        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.mu = nn.Linear(64, latent_size)
        self.log_var = nn.Linear(64, latent_size)
        self.l3 = nn.Linear(latent_size, state_dim)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def forward(self, x):
        x = F.tanh(self.l1(x))
        x = F.tanh(self.l2(x))
        mu = self.mu(x)
        log_var = self.log_var(x)
        z = self.reparameterize(mu, log_var)
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        dist = Normal(mu, log_var.exp())
        prob_state_prime = dist.rsample()
        state_dist_entropy = dist.entropy().mean()

        return prob_state_prime, state_dist_entropy, kl_div




# the architecture goes here
class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 64)
		self.n1 = nn.LayerNorm(64)
		self.l2 = nn.Linear(64, 64)
		self.n2 = nn.LayerNorm(64)
		self.l3 = nn.Linear(64, action_dim)

		self.max_action = max_action


	def forward(self, x):
		x = F.relu(self.l1(x))
		x = self.n1(x)
		x = F.relu(self.l2(x))
		x = self.n2(x)
		x = self.max_action * F.tanh(self.l3(x))
		return x


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(state_dim, 64)
		self.n1 = nn.LayerNorm(64)
		self.l2 = nn.Linear(64 + action_dim, 64)
		self.n2 = nn.LayerNorm(64)
		self.l3 = nn.Linear(64, 1)


	def forward(self, x, u):
		x = F.relu(self.l1(x))
		x = self.n1(x)
		x = F.relu(self.l2(torch.cat([x, u], 1)))
		x = self.n2(x)
		x = self.l3(x)
		return x
