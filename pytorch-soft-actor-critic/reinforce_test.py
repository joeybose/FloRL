from comet_ml import Experiment
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
import mujoco_py
import os
import gym
import ipdb
import numpy as np
import random
import ipdb

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

class Policy(nn.Module):
    def __init__(self, num_inputs, hidden_size, action_space):
        super(Policy,self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        self.linear = nn.Linear(num_inputs,hidden_size)
        self.mean = nn.Linear(hidden_size, num_outputs)
        self.log_std = nn.Linear(hidden_size, num_outputs)

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.linear(x))
        mean = self.mean(x)

        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX) #clamp log std into range
        std = log_std.exp()
        return mean, std

class REINFORCE(nn.Module):
    def __init__(self, num_inputs, hidden_size, action_space, lr = 1e-2):
        super(REINFORCE, self).__init__()
        self.action_space = action_space
        self.policy = Policy(num_inputs, hidden_size, action_space)
        self.reward_episode = []
        self.reward_history = []
        self.loss_history = []
        self.policy_history = Variable(torch.Tensor())
        self.gamma = 0.95

    def select_action(self, state, evaluate=False):
        state = torch.from_numpy(state).float().unsqueeze(0) # just to make it a Tensor obj
        # get mean and std
        mean, std = self.policy(state)
        # create normal distribution
        normal = Normal(mean, std)
        # sample action
        action = normal.sample()
        # get log prob of that action
        ln_prob = normal.log_prob(action)
        if not evaluate:
            if policy.policy_history.dim() != 0:
                self.policy_history = torch.cat([policy.policy_history, ln_prob])
            else:
                policy.policy_history = (c.log_prob(action))
        # ln_prob = ln_prob.sum()
        # squeeze action into [-1,1]
        # action = torch.tanh(action)
        return action, ln_prob, mean, std

def train(policy, trajectory):
    '''
       trajectory: a list of the form [(lnP(a_t|s_t), r(s_t,a_t) ),(lnP(a_{t+1}|s_{t+1}), r(s_{t+1},a_{t+1}))]
       Train the model by summing lnP(a_t|s_t)*r(s_t,a_t)
    '''
    optimizer = optim.Adam(policy.parameters())
    rewards = []
    loss = 0
    R = 0
    for r in policy.reward_episode[::-1]:
        R = r + policy.gamma * R
        rewards.insert(0,R)

    rewards = torch.FloatTensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

    # Calculate loss
    loss = (torch.sum(torch.mul(policy.policy_history.squeeze(), Variable(rewards)).mul(-1), -1))
    # for step in trajectory:
        # ln_prob, reward = step
        # loss -= ln_prob * reward

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    policy.loss_history.append(loss.item())
    policy.reward_history.append(np.sum(policy.reward_episode))
    policy.policy_history = Variable(torch.Tensor())
    policy.reward_episode= []

def evaluate_policy(policy, eval_episodes = 10):
    '''
        function to return the average reward of the policy over 10 runs
    '''
    avg_reward = 0.0
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, log_prob, mean, std = policy.select_action(np.array(obs),evaluate=True)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    print("the average reward is: {0}".format(avg_reward))
    #return avg_reward

def render_policy(policy):
    '''
        Function to see the policy in action
    '''
    obs = env.reset()
    done = False
    while not done:
        env.render()
        action,_,_,_ = policy.select_action(np.array(obs))
        obs, reward, done, _ = env.step(action)

    env.close()

if __name__ == '__main__':
    env = gym.make("MountainCarContinuous-v0")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space
    max_action = float(env.action_space.high[0])
    min_action = float(env.action_space.low[0])

    print("number of actions:{0}, dim of states: {1},\
          max_action:{2}, min_action: {3}".format(action_dim,state_dim,max_action,min_action))
    hidden_size = 256
    policy = REINFORCE(state_dim, hidden_size, action_dim)
    max_episodes = 200
    total_episodes = 0

    while total_episodes < max_episodes:
        obs = env.reset()
        done = False
        trajectory_info = []

        while not done:
            action, ln_prob, mean, std = policy.select_action(np.array(obs))
            next_state, reward, done, _ = env.step(action)
            policy.reward_episode.append(reward)
            trajectory_info.append([ln_prob, reward])
                #print("state:{0}, next_state:{1}, reward:{2}, action{3}, done:{4}".format(obs,next_state, reward, action,done))
            obs = next_state

        total_episodes += 1
        print("At episode:{0}".format(total_episodes))
        if total_episodes % 10 == 0:
            evaluate_policy(policy)

        train(policy,trajectory_info)
