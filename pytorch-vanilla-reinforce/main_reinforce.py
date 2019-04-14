import argparse
from comet_ml import Experiment
import torch
from torch.autograd import Variable
import torch.autograd as autograd
import numpy as np
import torch.nn as nn
import json
from reinforce_simple import REINFORCE
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import mujoco_py
import os
import gym
import ipdb

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

def main(args):

    # create env
    env = gym.make(args.env_name)
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # get env info
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space
    max_action = (env.action_space.high)
    min_action = (env.action_space.low)

    print("number of actions:{0}, dim of states: {1},\
          max_action:{2}, min_action: {3}".format(action_dim,state_dim,max_action,min_action))

    # setup comet_ml to track experiments
    if os.path.isfile("settings.json"):
        with open('settings.json') as f:
                data = json.load(f)
        args.comet_apikey = data["apikey"]
        args.comet_username = data["username"]
    else:
        raise NotImplementedError

    experiment = Experiment(api_key=args.comet_apikey,\
    project_name="florl",auto_output_logging="None",\
    workspace=args.comet_username,auto_metric_logging=False,\
    auto_param_logging=False)
    experiment.set_name(args.namestr)
    args.experiment = experiment

    # construct model
    hidden_size = args.hidden_size
    policy = REINFORCE(state_dim, hidden_size, action_dim, baseline = args.baseline)

    # start of experiment: Keep looping until desired amount of episodes reached
    max_episodes = args.num_episodes
    total_episodes = 0 # keep track of amount of episodes that we have done

    while total_episodes < max_episodes:

        obs = env.reset()
        done = False
        trajectory = [] # trajectory info for reinforce update
        episode_reward = 0 # keep track of rewards per episode

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
    parser.add_argument('--namestr', type=str, default='FloRL', \
            help='additional info in output filename to describe experiments')
    parser.add_argument('--num-episodes', type=int, default=2000, metavar='N',
                        help='maximum number of episodes (default:2000)')
    parser.add_argument('--hidden-size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    args = parser.parse_args()

    main(args)

