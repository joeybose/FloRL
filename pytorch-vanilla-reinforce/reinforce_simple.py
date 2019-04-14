from normalized_actions import NormalizedActions
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
from replay_memory import ReplayMemory
from sac import SAC

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class ValueNetwork(nn.Module):

    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, state):

        x = F.relu(self.linear1(state))
        x = self.linear2(x)

        return x

class REINFORCE:

    def __init__(self, num_inputs, hidden_size, action_space, lr_pi = 3e-4,\
                 lr_vf = 1e-3, baseline = False, gamma = 0.99, train_v_iters = 1):

        self.gamma = gamma
        self.action_space = action_space
        self.policy = Policy(num_inputs, hidden_size, action_space)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr = lr_pi)
        self.baseline = baseline
        self.train_v_iters = train_v_iters

        # create value network if we want to use baseline
        if self.baseline:
            self.value_function = ValueNetwork(num_inputs, hidden_size)
            self.value_optimizer = optim.Adam(self.value_function.parameters(), lr = lr_vf)


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

    def train(self, trajectory):

        '''
        trajectory: a list iof the form

        [( state , action , lnP(a_t|s_t), reward ), ...  ]

        '''

        log_probs = [item[2] for item in trajectory]
        rewards = [item[3] for item in trajectory]
        states = [item[0] for item in trajectory]
        actions = [item[1] for item in trajectory]

	#calculate rewards to go
        R = 0
        returns = []
        for r in rewards[::-1]:
            R = r + 0.99 * R
            returns.insert(0, R)

        returns = torch.tensor(returns)

        if self.baseline:

            # loop over this a couple of times
            for _ in range(self.train_v_iters):
                # calculate loss of value function
                value_estimates = []
                for state in states:
                    state = torch.from_numpy(state).float().unsqueeze(0) # just to make it a Tensor obj
                    value_estimates.append( self.value_function(state) )

                value_estimates = torch.stack(value_estimates).squeeze() # rewards to go for each step of env trajectory

                v_loss = F.mse_loss(value_estimates, returns)
                # update the weights
                self.value_optimizer.zero_grad()
                v_loss.backward()
                self.value_optimizer.step()

            # calculate advantage
            advantage = []
            for value, R in zip(value_estimates, returns):
                advantage.append(R - value)

            advantage = torch.Tensor(advantage)

            # caluclate policy loss
            policy_loss = []
            for log_prob, adv in zip(log_probs, advantage):
                policy_loss.append( - log_prob * adv)


        else:
            policy_loss = []
            for log_prob, R in zip(log_probs, returns):
                policy_loss.append( - log_prob * R)


        policy_loss = torch.stack( policy_loss ).sum()
        # update policy weights
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()


        if self.baseline:
            return policy_loss, v_loss

        else:
            return policy_loss

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
    parser.add_argument('--env-name', default="HalfCheetah-v1",
                        help='name of the environment to run')
    parser.add_argument('--seed', type=int, default=456, metavar='N',
                        help='random seed (default: 456)')
    parser.add_argument('--baseline', type=bool, default = False, help = 'Whether you want to add a baseline to Reinforce or not')
    parser.add_argument('--SAC', type=bool, default = False, help = 'Whether you want to use SAC model or not')
    parser.add_argument('--namestr', type=str, default='FloRL', \
            help='additional info in output filename to describe experiments')
    parser.add_argument('--num-episodes', type=int, default=2000, metavar='N',
                        help='maximum number of episodes (default:2000)')
#############################################################################################

    parser.add_argument('--policy', default="Gaussian",
                        help='algorithm to use: Gaussian | Deterministic')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default:True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.1, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.1)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Temperature parameter α automaically adjusted.')
    parser.add_argument('--batch_size', type=int, default=1024, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--clip', type=int, default=1, metavar='N',
                        help='Clipping for gradient norm')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')


###############################################################################################
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
    max_episodes = args.num_episodes
    total_episodes = 0

    # set up comet
    experiment = Experiment(api_key="BUXbNT79Q2PEtRkuX9swzxspZ",\
                            project_name="florl", workspace="nadeem-ward")

    experiment.set_name(args.namestr)

    if args.SAC:
        env = NormalizedActions(env)
        agent = SAC(env.observation_space.shape[0], env.action_space, args)
        # Memory
        memory = ReplayMemory(args.replay_size)

        while total_episodes < max_episodes:

            state = env.reset()
            done = False
            episode_reward = 0
            updates = 1

            # begining of episode
            while not done:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward

                mask = not done
                memory.push(state, action, reward, next_state, mask)  # Append transition to memory

                avg_val_loss = 0
                avg_policy_loss = 0
                avg_critic1_loss = 0
                avg_critic2_loss = 0
                avg_policy_loss = 0
                if len(memory) > args.batch_size:
                    for i in range(args.updates_per_step): # Number of updates per step in environment
                        # Sample a batch from memory
                        state_batch, action_batch, reward_batch, next_state_batch,\
                                mask_batch = memory.sample(args.batch_size)
                        # Update parameters of all the networks
                        value_loss, critic_1_loss, critic_2_loss, policy_loss,\
                                ent_loss, alpha = agent.update_parameters(state_batch,\
                                action_batch,reward_batch,next_state_batch,mask_batch, updates)
                        avg_val_loss += value_loss
                        avg_policy_loss += policy_loss
                        avg_critic1_loss += critic_1_loss
                        avg_critic2_loss += critic_2_loss
                        avg_policy_loss += policy_loss

                        updates +=1

            #end of episode
            total_episodes += 1

            avg_val_loss = avg_val_loss/updates
            avg_critic1_loss = avg_critic1_loss/updates
            avg_critic2_loss = avg_critic2_loss/updates
            avg_policy_loss = avg_policy_loss/updates
            if total_episodes % 10 == 0:
                print(total_episodes)
            experiment.log_metric("Loss Value", avg_val_loss, step =total_episodes)
            experiment.log_metric("Loss Critic 1",avg_critic1_loss,step = total_episodes)
            experiment.log_metric("Loss Critic 2",avg_critic2_loss,step=total_episodes)
            experiment.log_metric("Loss Policy",avg_policy_loss,step=total_episodes)
            experiment.log_metric("episode reward", episode_reward, step =total_episodes)



        env.close()


    else:

        policy = REINFORCE(state_dim, hidden_size, action_dim, baseline = args.baseline)

        while total_episodes < max_episodes:

            obs = env.reset()
            done = False
            trajectory = []
            episode_reward = 0

            while not done:
                action, ln_prob, mean, std = policy.select_action(np.array(obs))
                next_state, reward, done, _ = env.step(action)
                trajectory.append([np.array(obs), action, ln_prob, reward, next_state, done])

                obs = next_state
                episode_reward += reward

            total_episodes += 1

            if args.baseline:
                policy_loss, value_loss = policy.train(trajectory)
                experiment.log_metric("value function loss", value_loss, step = total_episodes)
            else:
                policy_loss = policy.train(trajectory)

            experiment.log_metric("policy loss",policy_loss, step = total_episodes)
            experiment.log_metric("episode reward", episode_reward, step =total_episodes)


        env.close()


