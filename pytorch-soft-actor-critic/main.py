import argparse
import os
import time
from comet_ml import Experiment
import json
import gym
from tqdm import tqdm
import numpy as np
import itertools
import torch
import ipdb
from sac import SAC
from normalized_actions import NormalizedActions
from replay_memory import ReplayMemory
from continous_grids import GridWorld


def main(args):
    # Environment
    if args.make_cont_grid:
        if args.smol:
            dense_goals = []
            if args.dense_goals:
                dense_goals = [(13.0, 8.0), (18.0, 11.0), (20.0, 15.0), (22.0, 19.0)]
            env = GridWorld(max_episode_len=500, num_rooms=1, action_limit_max=1.0, silent_mode=args.silent,
                            start_position=(8.0, 8.0), goal_position=(22.0, 22.0), goal_reward=+100.0,
                            dense_goals=dense_goals, dense_reward=+5,
                            grid_len=30)
            env_name = "SmallGridWorld"
        elif args.tiny:
            env = GridWorld(max_episode_len=500, num_rooms=0, action_limit_max=1.0, silent_mode=args.silent,
                            start_position=(5.0, 5.0), goal_position=(15.0, 15.0), goal_reward=+100.0,
                            dense_goals=[], dense_reward=+0,
                            grid_len=20)
            env_name = "TinyGridWorld"
        elif args.twotiny:
            env = GridWorld(max_episode_len=500, num_rooms=1, action_limit_max=1.0, silent_mode=args.silent,
                            start_position=(5.0, 5.0), goal_position=(15.0, 15.0), goal_reward=+100.0,
                            dense_goals=[], dense_reward=+0,
                            grid_len=20, door_breadth=3)
            env_name = "TwoTinyGridWorld"
        elif args.threetiny:
            env = GridWorld(max_episode_len=500, num_rooms=0, action_limit_max=1.0, silent_mode=args.silent,
                            start_position=(8.0, 8.0), goal_position=(22.0, 22.0), goal_reward=+100.0,
                            dense_goals=[], dense_reward=+0,
                            grid_len=30)
            env_name = "ThreeGridWorld"
        else:
            dense_goals = []
            if args.dense_goals:
                dense_goals = [(35.0, 25.0), (45.0, 25.0), (55.0, 25.0), (68.0, 33.0), (75.0, 45.0), (75.0, 55.0),
                               (75.0, 65.0)]
            env = GridWorld(max_episode_len=1000, num_rooms=1, action_limit_max=1.0, silent_mode=args.silent,
                            dense_goals=dense_goals)
            env_name = "VeryLargeGridWorld"
    else:
        env = NormalizedActions(gym.make(args.env_name))

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Agent
    agent = SAC(env.observation_space.shape[0], env.action_space, args)

    # Memory
    memory = ReplayMemory(args.replay_size)

    # Training Loop
    rewards = []
    test_rewards = []
    total_numsteps = 0
    updates = 0

    if args.debug:
        args.use_logger = False

    # Check if settings file
    if os.path.isfile("settings.json"):
        with open('settings.json') as f:
                data = json.load(f)
        args.comet_apikey = data["apikey"]
        args.comet_username = data["username"]
    else:
        raise NotImplementedError

    if args.comet:
        experiment = Experiment(api_key=args.comet_apikey,\
        project_name="florl",auto_output_logging="None",\
        workspace=args.comet_username,auto_metric_logging=False,\
        auto_param_logging=False)
        experiment.set_name(args.namestr)
        args.experiment = experiment

    if args.make_cont_grid:
        # The following lines are for visual purposes
        traj = []
        imp_states = []

    for i_episode in itertools.count():
        state = env.reset()
        if args.make_cont_grid:
            traj.append(state)

        episode_reward = 0
        while True:
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)  # Sample action from policy
            time.sleep(.002)
            next_state, reward, done, _ = env.step(action)  # Step
            #Visual
            if args.make_cont_grid:
                traj.append(next_state)
                if total_numsteps % 100 == 0 and total_numsteps != 0:
                    imp_states.append(next_state)
            mask = not done  # 1 for not done and 0 for done
            memory.push(state, action, reward, next_state, mask)  # Append transition to memory
            if len(memory) > args.batch_size:
                for i in range(args.updates_per_step): # Number of updates per step in environment
                    # Sample a batch from memory
                    state_batch, action_batch, reward_batch, next_state_batch,\
                            mask_batch = memory.sample(args.batch_size)
                    # Update parameters of all the networks
                    value_loss, critic_1_loss, critic_2_loss, policy_loss,\
                            ent_loss, alpha = agent.update_parameters(state_batch,\
                            action_batch,reward_batch,next_state_batch,mask_batch, updates)

                    if args.comet:
                        args.experiment.log_metric("Loss Value", value_loss,step=updates)
                        args.experiment.log_metric("Loss Critic 1",critic_1_loss,step=updates)
                        args.experiment.log_metric("Loss Critic 2",critic_2_loss,step=updates)
                        args.experiment.log_metric("Loss Policy",policy_loss,step=updates)
                        args.experiment.log_metric("Entropy",ent_loss,step=updates)
                        args.experiment.log_metric("Entropy Temperature",alpha,step=updates)
                    updates += 1

            state = next_state
            total_numsteps += 1
            episode_reward += reward

            if done:
                break

        if total_numsteps > args.num_steps:
            break

        rewards.append(episode_reward)
        if args.comet:
            args.experiment.log_metric("Train Reward",episode_reward,step=i_episode)
            args.experiment.log_metric("Average Train Reward",\
                    np.round(np.mean(rewards[-100:]),2),step=i_episode)
        print("Episode: {}, total numsteps: {}, reward: {}, average reward: {}".format(i_episode,\
                        total_numsteps, np.round(rewards[-1],2),\
                        np.round(np.mean(rewards[-100:]),2)))

        if i_episode % 10 == 0 and args.eval == True:
            state = torch.Tensor([env.reset()])
            episode_reward = 0
            while True:
                action = agent.select_action(state, eval=True)
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward

                state = next_state
                if done:
                    break

            if args.comet:
                args.experiment.log_metric("Test Reward", episode_reward, step=i_episode)

            test_rewards.append(episode_reward)
            print("----------------------------------------")
            print("Test Episode: {}, reward: {}".format(i_episode, test_rewards[-1]))
            print("----------------------------------------")
    if args.make_cont_grid:
        experiment_id = None
        if args.comet:
            experiment_id = args.experiment.id
        #Visual
        # env.vis_trajectory(np.asarray(traj), args.namestr, experiment_id, np.asarray(imp_states))
        env.test_vis_trajectory(np.asarray(traj), args.namestr, experiment_id)

    env.close()

if __name__ == '__main__':
    """
    Process command-line arguments, then call main()
    """
    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
    parser.add_argument('--env-name', default="HalfCheetah-v2",
                        help='name of the environment to run')
    parser.add_argument('--policy', default="Gaussian",
                        help='algorithm to use: Gaussian | Deterministic')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default:True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.1, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.1)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Temperature parameter α automaically adjusted.')
    parser.add_argument('--seed', type=int, default=456, metavar='N',
                        help='random seed (default: 456)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--clip', type=int, default=1, metavar='N',
                        help='Clipping for gradient norm')
    parser.add_argument('--num_steps', type=int, default=1000000, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument("--comet", action="store_true", default=False,help='Use comet for logging')
    parser.add_argument('--debug', default=False, action='store_true',help='Debug')
    parser.add_argument('--namestr', type=str, default='FloRL', \
            help='additional info in output filename to describe experiments')
    parser.add_argument('--n_blocks', type=int, default=5,\
                        help='Number of blocks to stack in a model (MADE in MAF; Coupling+BN in RealNVP).')
    parser.add_argument('--n_components', type=int, default=1,\
                        help='Number of Gaussian clusters for mixture of gaussians models.')
    parser.add_argument('--flow_hidden_size', type=int, default=100,\
                        help='Hidden layer size for MADE (and each MADE block in an MAF).')
    parser.add_argument('--n_hidden', type=int, default=1, help='Number of hidden layers in each MADE.')
    parser.add_argument('--activation_fn', type=str, default='relu',\
                        help='What activation function to use in the MADEs.')
    parser.add_argument('--input_order', type=str, default='sequential',\
                        help='What input order to use (sequential | random).')
    parser.add_argument('--conditional', default=False, action='store_true',\
                        help='Whether to use a conditional model.')
    parser.add_argument('--no_batch_norm', action='store_true')
    parser.add_argument('--flow_model', default='maf', help='Which model to use: made, maf.')

    # flags for using reparameterization trick or not
    parser.add_argument('--reparam',dest='reparam',action='store_true')
    parser.add_argument('--no-reparam',dest='reparam',action='store_false')
    # flags for using a tanh activation or not
    parser.add_argument('--tanh', dest='tanh', action='store_true')
    parser.add_argument('--no-tanh', dest='tanh', action='store_false')
    # For different gridworld environments
    parser.add_argument('--make_cont_grid', default=False, action='store_true', help='Make GridWorld')
    parser.add_argument('--dense_goals', default=False, action='store_true', help='Create sub-goals')
    parser.add_argument("--smol", action="store_true", default=False, help='Change to a smaller sized gridworld')
    parser.add_argument("--tiny", action="store_true", default=False, help='Change to the smallest sized gridworld')
    parser.add_argument("--twotiny", action="store_true", default=False,
                        help='Change to 2x the smallest sized gridworld')
    parser.add_argument("--threetiny", action="store_true", default=False,
                        help='Change to 3x the smallest sized gridworld')
    parser.add_argument("--silent", action="store_true", default=False,
                        help='Display graphical output. Set to true when running on a server.')

    parser.set_defaults(reparam=True, tanh=True)

    args = parser.parse_args()
    args.cond_label_size = None
    main(args)

