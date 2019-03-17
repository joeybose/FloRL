import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils
from torch.distributions import Normal
import flows

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Re-tuned version of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, action_dim)

		self.max_action = max_action


	def forward(self, x):
		x = F.relu(self.l1(x))
		x = F.relu(self.l2(x))
		x = self.max_action * torch.tanh(self.l3(x))
		return x


class MuActor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(MuActor, self).__init__()

		self.l1 = nn.Linear(state_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.mu = nn.Linear(300, action_dim)
		self.log_var = nn.Linear(300, action_dim)
		#self.log_det_j = 0.
		self.z_size = action_dim

		self.num_flows = 15
		self.flow = flows.Planar


		# Amortized flow parameters
		self.amor_u = nn.Linear(300, self.num_flows * action_dim)
		self.amor_w = nn.Linear(300, self.num_flows * action_dim)
		self.amor_b = nn.Linear(300, self.num_flows)

		# Normalizing flow layers
		for k in range(self.num_flows):
			flow_k = self.flow()
			self.add_module('flow_' + str(k), flow_k)


		self.max_action = max_action


	def forward(self, x):
		batch_size = x.size(0)
		x = F.relu(self.l1(x))
		x = F.relu(self.l2(x))
		mu = self.max_action * torch.tanh(self.mu(x))
		log_var = self.max_action * torch.tanh(self.log_var(x))

		dist = Normal(mu, log_var.exp())
		action = dist.rsample()
		z = [action]

		u = self.amor_u(x).view(batch_size, self.num_flows, self.z_size, 1)
		w = self.amor_w(x).view(batch_size, self.num_flows, 1, self.z_size)
		b = self.amor_b(x).view(batch_size, self.num_flows, 1, 1)

		self.log_det_j = torch.zeros(batch_size).to(device)

		for k in range(self.num_flows):
			flow_k = getattr(self, 'flow_' + str(k))
			z_k, log_det_jacobian = flow_k(z[k], u[:, k, :, :], w[:, k, :, :], b[:, k, :, :])
			z.append(z_k)
			self.log_det_j += log_det_jacobian

		final_action = z[-1]


		log_prob = dist.log_prob(action)
		log_prob = log_prob.sum(-1, keepdim=True)
		log_prob_final_action = log_prob.squeeze() - self.log_det_j

		probability_final_action = torch.exp(log_prob_final_action)

		return final_action, probability_final_action, log_prob_final_action






class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(state_dim + action_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, 1)

	def forward(self, x, u):
		x = F.relu(self.l1(torch.cat([x, u], 1)))
		x = F.relu(self.l2(x))
		x = self.l3(x)
		return x


class FlowDDPG(object):
	def __init__(self, state_dim, action_dim, max_action):
		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = Actor(state_dim, action_dim, max_action).to(device)

		self.behaviour_actor = MuActor(state_dim, action_dim, max_action).to(device)
		self.behaviour_actor_target = MuActor(state_dim, action_dim, max_action).to(device)

		self.actor_target.load_state_dict(self.actor.state_dict())
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

		self.behaviour_actor_target.load_state_dict(self.behaviour_actor.state_dict())
		self.behaviour_actor_optimizer = torch.optim.Adam(self.behaviour_actor.parameters())

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = Critic(state_dim, action_dim).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters())


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def select_behaviour_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		action, _, _  = self.behaviour_actor(state)
		entropy = 0
		return action.cpu().data.numpy().flatten(), entropy


	def evaluate_behaviour_actions(self, state):
		sampled_action, probability_final_action, log_prob_final_action = self.behaviour_actor(state)
		entropy = - (probability_final_action * log_prob_final_action)
		entropy = entropy.mean()

		return entropy


	def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005):

		for it in range(iterations):

			# Sample replay buffer
			x, y, u, r, d = replay_buffer.sample(batch_size)
			state = torch.FloatTensor(x).to(device)
			action = torch.FloatTensor(u).to(device)
			next_state = torch.FloatTensor(y).to(device)
			done = torch.FloatTensor(1 - d).to(device)
			reward = torch.FloatTensor(r).to(device)

			# Compute the target Q value
			target_Q = self.critic_target(next_state, self.actor_target(next_state))
			target_Q = reward + (done * discount * target_Q).detach()

			# Get current Q estimate
			current_Q = self.critic(state, action)
			# Compute critic loss
			critic_loss = F.mse_loss(current_Q, target_Q)

			entropy_loss = self.evaluate_behaviour_actions(state)
			behaviour_actor_loss = - 1. * entropy_loss

			# Optimize the critic
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()

			# Compute actor loss
			actor_loss = -self.critic(state, self.actor(state)).mean()

			# Optimize the actor
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			self.behaviour_actor_optimizer.zero_grad()
			behaviour_actor_loss.backward()
			self.behaviour_actor_optimizer.step()


			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

			for param, target_param in zip(self.behaviour_actor.parameters(), self.behaviour_actor_target.parameters()):
				target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


	def save(self, filename, directory):
		torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
		torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))


	def load(self, filename, directory):
		self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
		self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
