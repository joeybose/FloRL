import argparse
from comet_ml import Experiment
import torch
from torch.autograd import Variable
import torch.autograd as autograd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import mujoco_py
import os
import gym
import ipdb

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

class Policy(nn.Module):

    def __init__(self, num_inputs, hidden_size, action_space):

       	super(Policy, self).__init__()

        self.action_space = action_space
        num_outputs = action_space.shape[0] # the number of output actions

        self.linear = nn.Linear(num_inputs, hidden_size)
        self.mean = nn.Linear(hidden_size, num_outputs)
        self.log_std = nn.Linear(hidden_size, num_outputs)

    def forward(self, inputs):

        # forward pass of NN
        x = inputs
        x = F.relu(self.linear(x))

        mean = self.mean(x)
        log_std = self.log_std(x) # if more than one action this will give you the diagonal elements of a diagonal covariance matrix
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX) # We limit the variance by forcing within a range of -2,20
        std = log_std.exp()

        return mean, std

class REINFORCE:

    def __init__(self, num_inputs, hidden_size, action_space, lr = 1e-2):

        self.action_space = action_space
        self.policy = Policy(num_inputs, hidden_size, action_space)
        self.optimizer = optim.Adam(self.policy.parameters(), lr = lr)

    def select_action(self,state):

        state = torch.from_numpy(state).float().unsqueeze(0) # just to make it a Tensor obj

        # get mean and std
        mean, std = self.policy(state)

        # create normal distribution
        normal = Normal(mean, std)

        # sample action
        action = normal.sample()

        # get log prob of that action
        ln_prob = normal.log_prob(action)
        ln_prob = ln_prob.sum()
	# squeeze action into [-1,1]
        action = torch.tanh(action)
        # turn actions into numpy array
        action = action.numpy()

        return action[0], ln_prob, mean, std

    def get_prob(self, action, state):

        state = torch.from_numpy(state).float().unsqueeze(0) # just to make it a Tensor obj

        # get mean and std
        mean, std = self.policy(state)

        # create normal distribution
        normal = Normal(mean, std)

        # get log prob of that action
        ln_prob = normal.log_prob(action)
        ln_prob = ln_prob.sum()

        return ln_prob

    def train(self, trajectory):

        '''
        trajectory: a list iof the form
        [(lnP(a_t|s_t), r(s_t,a_t) ),(lnP(a_{t+1}|s_{t+1}), r(s_{t+1},a_{t+1}))]

        Train the model by summing lnP(a_t|s_t)*r(s_t,a_t)
        '''


        loss = 0
        for step in trajectory:
            # look at one step
            ln_prob, reward = step
            # accumulate the loss
            loss = loss - ln_prob * reward

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

def evaluate_policy(policy, eval_episodes = 10):
    '''
        function to return the average reward of the policy over 10 runs
    '''
    avg_reward = 0.0
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, log_prob, mean, std = policy.select_action(np.array(obs) )
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

    """
    Process command-line arguments, then call main()
    """
    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
    parser.add_argument('--env-name', default="HalfCheetah-v2",
                        help='name of the environment to run')
    parser.add_argument('--seed', type=int, default=456, metavar='N',
                        help='random seed (default: 456)')

    args = parser.parse_args()



    #env = gym.make("MountainCarContinuous-v0") #
    env = gym.make(args.env_name)
    #env = gym.make("Pendulum-v0")
    #env = gym.make("InvertedPendulum-v1")
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)



    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space
    max_action = (env.action_space.high)
    min_action = (env.action_space.low)

    print("number of actions:{0}, dim of states: {1},\
          max_action:{2}, min_action: {3}".format(action_dim,state_dim,max_action,min_action))
    hidden_size = 256
    policy = REINFORCE(state_dim, hidden_size, action_dim)
    max_episodes = 1000
    total_episodes = 0
    # set up comet
    # experiment = Experiment(api_key="BUXbNT79Q2PEtRkuX9swzxspZ",
    #                        project_name="florl", workspace="nadeem-ward")

     #experiment.set_name("FLORL")


    while total_episodes < max_episodes:
        obs = env.reset()
        done = False
        trajectory_info = []
        episode_reward = 0
        episode_loss = 0

        while not done:
            action, ln_prob, mean, std = policy.select_action(np.array(obs))
            next_state, reward, done, _ = env.step(action)
            trajectory_info.append([ln_prob, reward])
                #print("state:{0}, next_state:{1}, reward:{2}, action{3}, done:{4}".format(obs,next_state, reward, action,done))
            obs = next_state
            episode_reward += reward


        total_episodes += 1
        #print("At episode:{0}".format(total_episodes))
        if total_episodes % 10 == 0:
            evaluate_policy(policy)

        loss = policy.train(trajectory_info)
#        experiment.log_metric("loss value", loss, step = total_episodes)
#        experiment.log_metric("episode reward", episode_reward, step =total_episodes)


    env.close()
