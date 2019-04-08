from comet_ml import Experiment
from replay_memory import ReplayMemory
import itertools
import numpy as np
import torch
import gym
import argparse
import os
import json
import utils
import reinforce_simple
import ipdb

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


# Runs policy for X episodes and returns average reward
def evaluate_policy(policy,env, eval_episodes=10):
	avg_reward = 0.
	for _ in range(eval_episodes):
		obs = env.reset()
		done = False
		while not done:
			action = policy.select_action(np.array(obs))
			obs, reward, done, _ = env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print ("---------------------------------------")
	print ("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
	print ("---------------------------------------")
	return avg_reward


def main(args):

######## set up the environment ###############
    env = gym.make(args.env_name)
    # Set seeds
    seed = np.random.randint(10)
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

####### set up comet ##########################
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

    ####### environment info  #####################
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space
    max_action = env.action_space.high
    min_action = env.action_space.low

    print("number of actions:{0}, dim of states: {1}, max_action:{2}, min_action: {3}".format(action_dim,state_dim,max_action,min_action))

    ########## Initialize policy ################
    if args.policy == "REINFORCE": agent = reinforce_simple.REINFORCE(state_dim, args.hidden_size, action_dim)

    print("state dim:{0}, hidden dim: {1}, action_dim {2}".format(state_dim,
                                                                  args.hidden_size,
                                                                  action_dim) )
    ######### set up element to keep track #######

    rewards = []
    test_rewards = []
    total_numsteps = 0 # how many times steps over all, sum of time steps for each action
    timesteps_since_eval = 0 # time step to keep track how many times I've evaluated
    episode_num = 0 # number of episodes
    done = True # whether or not an episode is done or not

    ######### Main loop #######################

    for i_episode in itertools.count(): # keep looping through episodes

        state = env.reset()
        episode_reward = 0 # all the rewards for an episode
        trajectory = [] # keep track of info from trajectory

        while True:

            # use random actions at beggining
            if args.start_steps > total_numsteps:
                # random action
                action = env.action_space.sample()
                ln_prob = agent.get_prob(action, state)
            # use real action
            else:
                action, ln_prob, mean, std = agent.select_action(state)# Sample action from policy

            next_state, reward, done, _ = env.step(action)  # Step
            #mask = not done  # 1 for not done and 0 for done
            #memory.push(state, action, reward, next_state, mask)  # Append transition to memory
            trajectory.append([ln_prob, reward])


#            if len(memory) > args.batch_size:
#                for i in range(args.updates_per_step): # Number of updates per step in environment
#                    # Sample a batch from memory
#                    state_batch, action_batch, reward_batch, next_state_batch,\
#                            mask_batch = memory.sample(args.batch_size)
#                    # Update parameters of all the networks
#                    value_loss, critic_1_loss, critic_2_loss, policy_loss,\
#                            ent_loss, alpha = agent.update_parameters(state_batch,\
#                            action_batch,reward_batch,next_state_batch,mask_batch, updates)
#
#                    if args.comet:
#                        args.experiment.log_metric("Loss Value", value_loss,step=updates)
#                        args.experiment.log_metric("Loss Critic 1",critic_1_loss,step=updates)
#                        args.experiment.log_metric("Loss Critic 2",critic_2_loss,step=updates)
#                        args.experiment.log_metric("Loss Policy",policy_loss,step=updates)
#                        args.experiment.log_metric("Loss Entropy",ent_loss,step=updates)
#                        args.experiment.log_metric("Entropy Temperature",alpha,step=updates)
#                    updates += 1

            state = next_state
            total_numsteps += 1
            episode_reward += reward

            if done:
                break

        # train agent and keep track of loss
        loss = agent.train(trajectory)
        args.experiment.log_metric("Loss Value", loss, step = i_episode)


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

        # evaluate the policy at intervals
        if i_episode % 10 == 0 and args.eval == True:
            state = env.reset()
            episode_reward = 0
            while True:
                action = agent.select_action(np.array(state) )
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward

                state = next_state
                if done:
                    break

            if args.comet:
                args.experiment.log_metric("Test Reward",episode_reward,step=i_episode)

            test_rewards.append(episode_reward)
            print("----------------------------------------")
            print("Test Episode: {}, reward: {}".format(i_episode, test_rewards[-1]))
            print("----------------------------------------")

    env.close()





if __name__ == "__main__":

    """
    Process command-line arguments, then call main()
    """
    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
    parser.add_argument('--env-name', default="HalfCheetah-v2",
                        help='name of the environment to run')
    parser.add_argument('--policy', default="REINFORCE",
                        )
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default:True)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--seed', type=int, default=456, metavar='N',
                        help='random seed (default: 456)')
    parser.add_argument('--batch_size', type=int, default=1024, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--clip', type=int, default=1, metavar='N',
                        help='Clipping for gradient norm')
    parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument("--comet", action="store_true", default=False,help='Use comet for logging')
    parser.add_argument('--debug', default=False, action='store_true',help='Debug')
    parser.add_argument('--namestr', type=str, default='FloRL', \
            help='additional info in output filename to describe experiments')

    args = parser.parse_args()
    args.cond_label_size = None
    main(args)







############################################################################

#    print("BEFORE ENTERING LOOP")
#    print("------ \n \n \n ")
#    while total_timesteps < args.max_timesteps: # loop until desired timesteps
#
#        if done: # if we are at the end of an episode
#
#            if total_timesteps != 0: # if we actually at end of episode
#                # update polcies and print info
#                print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f")\
#                            % (total_timesteps,episode_num,episode_timesteps,episode_reward))
#
##                ipdb.set_trace()
#                policy.train(trajectory)
#
#            # Evaluate episode
#            if timesteps_since_eval >= args.eval_freq:
#                timesteps_since_eval %= args.eval_freq
#                evaluations.append(evaluate_policy(policy, env))
#                if args.comet:
#                    args.experiment.log_metric(
#                        "Evaluation Reward", evaluations[-1],step=global_step)
#
#            # Reset environment
#            obs = env.reset()
#            done = False
#            training_evaluations.append(episode_reward)
#            if args.comet:
#                args.experiment.log_metric(
#                    "Training Reward", episode_reward, step=global_step)
#
#            if args.use_logger:
#                logger.training_record_reward(training_evaluations)
#                logger.save_2()
#
#            episode_reward = 0 # entire reward of an episode
#            episode_timesteps = 0 # number of timesteps in this episode
#            episode_num += 1
#
#        # Select action randomly or according to policy
#        if total_timesteps < args.start_timesteps: # we want to run random policy at start
#            action = env.action_space.sample()
#        else:
#            action = policy.select_action(np.array(obs))            # Perform action
#
#        new_obs, reward, done, _ = env.step(action)
#        done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
#        episode_reward += reward
#
#        # Store data in replay buffer
#        replay_buffer.add((obs, new_obs, action, reward, done_bool))
#
#        obs = new_obs
#
#        episode_timesteps += 1
#        total_timesteps += 1
#        timesteps_since_eval += 1
#        global_step += 1
#
#    # Final evaluation
#    evaluations.append(evaluate_policy(policy,env))
#    training_evaluations.append(episode_reward)
#    if args.comet:
#            args.experiment.log_metric("Evaluation Reward", evaluations[-1],step=global_step)


