import math
import numpy as np
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random


from tensorboardX import SummaryWriter

from utils import *
#from utils import Memory
from exploration_models import *
from models import Actor, Critic, StateDistributionGaussianVAE

from PIL import Image
from torchvision.transforms import ToTensor


class Agent(object):
    """
    The DDPG Agent
    """
    def __init__(self,
                 args,
                 env,
                 exploration,
                 latent_size,
                 logger,
                 writer_dir = None):
        """
        init agent
        """
        self.env = env
        self.args = args
        self.exploration = exploration

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.max_action = int(env.action_space.high[0])

        self.device = torch.device("cuda" if (torch.cuda.is_available() and  self.args.gpu) else "cpu")

        self.logger = logger

        self.latent_size = latent_size

        # TODO: set the random seed in the main launcher
        # set the seeds here again
        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        if self.args.gpu:
            torch.cuda.manual_seed(self.args.seed )


        # create the models and target networks
        self.actor = Actor(self.state_dim, self.action_dim, self.max_action).to(self.device)
        self.actor_target = Actor(self.state_dim, self.action_dim, self.max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        # NOTE: the fix  LR for DDPG agen
        self.critic = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_target = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), weight_decay=1e-2)

        if self.args.policy_name == "LearnMuDDPG":
            self.state_distribution_vae = StateDistributionGaussianVAE(self.state_dim, self.latent_size).to(self.device)
            self.state_distribution_vae_optimizer = torch.optim.Adam(self.state_distribution_vae.parameters(),lr=1e-4)

            # create the models and target networks
            self.actor_mu = Actor(self.state_dim, self.action_dim, self.max_action).to(self.device)
            self.actor_target_mu = Actor(self.state_dim, self.action_dim, self.max_action).to(self.device)
            self.actor_target_mu.load_state_dict(self.actor_mu.state_dict())
            self.actor_optimizer_mu = torch.optim.Adam(self.actor_mu.parameters(), lr=1e-4)

            # NOTE: the fix  LR for DDPG agen
            self.critic_mu = Critic(self.state_dim, self.action_dim).to(self.device)
            self.critic_target_mu = Critic(self.state_dim, self.action_dim).to(self.device)
            self.critic_target_mu.load_state_dict(self.critic_mu.state_dict())
            self.critic_optimizer_mu = torch.optim.Adam(self.critic_mu.parameters(), weight_decay=1e-2)


        self.memory = Memory(self.args.replay_size)
        self.memory_mu = Memory(self.args.replay_size)
        self.writer = SummaryWriter(log_dir=writer_dir) if  writer_dir is not None else None
        self.total_steps = 0

    def compute_per_step_entropy(self, state):
        # state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        state_log_probs, state_dist_entropy, _ = self.state_distribution_vae(state)
        return state_log_probs, state_dist_entropy

    def compute_stationary_distribution_entropy(self, state):
        state_log_probs, state_dist_entropy, kl_div = self.state_distribution_vae(state)        
        return state_dist_entropy, kl_div


    def pi(self, state):
        """
        take the action based on the current policy
        """
        self.actor.eval()
        action =  self.actor(state).detach().cpu().numpy().flatten()
        self.actor.train()
        return action

    def mu(self, state):
        """
        take the action based on the current policy
        """
        self.actor_mu.eval()
        action =  self.actor_mu(state).detach().cpu().numpy().flatten()
        self.actor_mu.train()
        return action



    def exp_pi(self, state, step):
        """
        gives the noisy action
        """
        if self.args.policy_name == "DDPG":
            if step >= self.args.num_steps/3:
                action = self.pi(state)
            else:
                ## take random actions for the first 1/3rd timesteps
                action = self.env.action_space.sample()

        elif self.args.policy_name == "LearnMuDDPG":
            if step >= self.args.num_steps/3:
                action = self.pi(state)
            else:
                ## take actions from Behaviour Policy for the first 1/3rd timesteps
                action = self.mu(state)           

        # get action according to the policy
        # action = self.pi(state)
        if self.exploration is not None:
            prev_action = action
            if isinstance(self.exploration, RandomWalkNoise):
                action = self.exploration()

            else:
                noise = self.exploration()
                assert noise.shape == action.shape
                action += noise
        # clip the action
        action = action.clip(self.env.action_space.low, self.env.action_space.high)
        return action


    def off_policy_update(self, epoch):
        """
        the update to network
        """
        actor_loss_list = []
        critic_loss_list = []
        pred_v_list = []

        for _ in range(self.args.updates_per_step):

            # sample a transition buffer
            transitions = self.memory.get_minibatch(self.args.batch_size)
            batch = Transition(*zip(*transitions))
            bs = self.args.batch_size

            state = torch.from_numpy(np.asarray(batch.state).reshape(bs, -1)).float().to(self.device)
            action = torch.from_numpy(np.asarray(batch.action).reshape(bs, -1)).float().to(self.device)
            reward = torch.from_numpy(np.asarray(batch.reward).reshape(bs, -1)).float().to(self.device)
            next_state = torch.from_numpy(np.asarray(batch.next_state).reshape(bs, -1)).float().to(self.device)
            done = torch.from_numpy(np.asarray(batch.done).reshape(bs, -1)).float().to(self.device)

            Q_value = self.critic(state, action)
            Target_Q_estimate = self.critic_target(next_state, self.actor_target(next_state))
            Q_target_value = (reward + ( (1. - done) * self.args.gamma * Target_Q_estimate)).detach()

            td_error = Q_target_value - Q_value

            # critic loss
            critic_loss = F.mse_loss(Q_value, Q_target_value)

            # optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # compute the actor loss
            actor_loss = - self.critic(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            soft_update(self.actor_target, self.actor, self.args.tau)
            soft_update(self.critic_target, self.critic, self.args.tau)

            # append to the list
            actor_loss_list.append(actor_loss.item())
            critic_loss_list.append(critic_loss.item())
            pred_v_list.append(Q_value.mean().item())

            if self.total_steps % 1000 == 0:
                self.writer.add_scalar("actor_loss", np.mean(actor_loss_list), epoch)
                self.writer.add_scalar("critic_loss", np.mean(critic_loss_list), epoch)
                self.writer.add_scalar("pred_v", np.mean(pred_v_list), epoch)



            if self.args.policy_name == "LearnMuDDPG":
                transitions = self.memory_mu.get_minibatch(self.args.batch_size)
                batch = Transition(*zip(*transitions))
                bs = self.args.batch_size                
                state = torch.from_numpy(np.asarray(batch.state).reshape(bs, -1)).float().to(self.device)
                action = torch.from_numpy(np.asarray(batch.action).reshape(bs, -1)).float().to(self.device)
                cumulant = torch.from_numpy(np.asarray(batch.reward).reshape(bs, -1)).float().to(self.device)
                next_state = torch.from_numpy(np.asarray(batch.next_state).reshape(bs, -1)).float().to(self.device)
                done = torch.from_numpy(np.asarray(batch.done).reshape(bs, -1)).float().to(self.device)



                Q_value = self.critic_mu(state, action)
                Target_Q_estimate = self.critic_target_mu(next_state, self.actor_target(next_state))
                Q_target_value = (cumulant + ( (1. - done) * self.args.gamma * Target_Q_estimate)).detach()

                td_error = Q_target_value - Q_value

                # critic loss
                critic_loss_mu = F.mse_loss(Q_value, Q_target_value)

                # optimize the critic
                self.critic_optimizer_mu.zero_grad()
                critic_loss_mu.backward()
                self.critic_optimizer_mu.step()

                # compute the actor loss
                actor_loss_mu = - self.critic_mu(state, self.actor_mu(state)).mean()

                # Optimize the actor
                self.actor_optimizer_mu.zero_grad()
                actor_loss_mu.backward()
                self.actor_optimizer_mu.step()

                # Update the frozen target models
                soft_update(self.actor_target_mu, self.actor_mu, self.args.tau)
                soft_update(self.critic_target_mu, self.critic_mu, self.args.tau)

                _, kl_div = self.compute_stationary_distribution_entropy(state)
                kl_loss = kl_div.mean()
                self.state_distribution_vae_optimizer.zero_grad()
                kl_loss.backward()
                self.state_distribution_vae_optimizer.step()




        return [np.mean(actor_loss_list), np.mean(critic_loss_list), np.mean(pred_v_list)]

    def run(self):
        """
        the actual ddpg algorithm here
        """
        results_dict = {
        "train_rewards" : [],
        "eval_rewards" : [],
        }
        update_steps = 0
        eval_steps = 0
        self.total_steps = 0
        num_episodes = 0

        # reset state and env
        # reset exploration porcess
        state = self.env.reset()
        done = False
        if self.exploration is not None and not isinstance(self.exploration, RandomWalkNoise):
            self.exploration.reset()
        ep_reward = 0
        ep_len = 0
        start_time = time.time()

        #The following lines are for visual purposes
        traj=[]
        imp_states=[]
        traj.append(state)

        timesteps_since_eval = 0

        evaluations = []
        all_train_rewards = []

        for step in range(self.args.num_steps):
            # convert the state to tensor
            state_tensor = torch.from_numpy(state).float().to(self.device).view(-1, self.state_dim)
            # get the expl action
            action = self.exp_pi(state_tensor, step)

            next_state, reward, done, _ = self.env.step(action)

            if self.args.policy_name == "LearnMuDDPG":
                _, state_dist_entropy = self.compute_per_step_entropy(state_tensor)
                cumulant = state_dist_entropy.data.cpu().numpy()

            ep_reward += reward
            ep_len += 1
            self.total_steps += 1

            #Visual
            traj.append(next_state)
            if step % 100 == 0 and step != 0:
                imp_states.append(next_state)

            # hard reset done for rllab envs
            done = done or ep_len >= self.args.max_path_len
            # add the transition in the memory
            transition = Transition(state = state, action = action,reward = reward, next_state = next_state,done = float(done))
            self.memory.add(transition)

            if self.args.policy_name == "LearnMuDDPG":
                transition_mu = Transition(state = state, action = action,reward = cumulant, next_state = next_state,done = float(done))
                self.memory_mu.add(transition_mu)

            # update the state
            state = next_state

            if done :
                ### evaluate the policy
                if timesteps_since_eval >= self.args.eval_every:
                    timesteps_since_eval %= self.args.eval_every
                    eval_reward, eval_length = self.evaluate_policy()
                    evaluations.append(eval_reward)                
                    if self.args.use_logger:
                        self.logger.record_reward(evaluations)
                        self.logger.save()

                # log
                results_dict["train_rewards"].append(ep_reward)
                self.writer.add_scalar("ep_reward", ep_reward, num_episodes)
                self.writer.add_scalar("ep_len", ep_len, num_episodes)
                self.writer.add_scalar("reward_step", ep_reward, self.total_steps)

                log(
                    'Num Episode {}\t'.format(num_episodes) + \
                    'Time: {:.2f}\t'.format(time.time() - start_time) + \
                    'E[R]: {:.2f}\t'.format(ep_reward) +\
                    'E[t]: {}\t'.format(ep_len) +\
                    'Step: {}\t'.format(self.total_steps) +\
                    'Epoch: {}\t'.format(self.total_steps // 10000) +\
                    'avg_train_reward: {:.2f}\t'.format(np.mean(results_dict["train_rewards"][-100:]))
                    )

                if self.args.use_logger:
                    all_train_rewards.append(np.mean(results_dict["train_rewards"][-100:]))
                    self.logger.record_train_reward(all_train_rewards)
                    self.logger.save_2()

                if self.args.comet:
                    self.args.experiment.log_metric("Num Episode", num_episodes, step=self.total_steps)
                    self.args.experiment.log_metric("Time", time.time(), step=self.total_steps)
                    self.args.experiment.log_metric("Episode Reward", ep_reward, step=self.total_steps)
                    self.args.experiment.log_metric("Episode Length", ep_len, step=self.total_steps)
                    self.args.experiment.log_metric("Step", self.total_steps, step=self.total_steps)
                    self.args.experiment.log_metric("Epoch", self.total_steps // 10000 , step=self.total_steps)
                    self.args.experiment.log_metric("avg_train_reward", np.mean(results_dict["train_rewards"][-100:]), step=self.total_steps)

                # reset
                state = self.env.reset()
                done = False
                if self.exploration is not None and not isinstance(self.exploration, RandomWalkNoise):
                    self.exploration.reset()
                ep_reward = 0
                ep_len = 0
                start_time = time.time()
                # update counters
                num_episodes += 1

            # update here
            if self.memory.count > self.args.batch_size * 5:
                self.off_policy_update(update_steps)
                update_steps += 1

            if self.total_steps % self.args.checkpoint_interval == 0:
                self.save_models()

            timesteps_since_eval += 1

        # save the models
        self.save_models()
        # save the results
        torch.save(results_dict, os.path.join(self.args.out, 'results_dict.pt'))

        #Visual
        img = self.env.vis_trajectory(np.asarray(traj), self.args.namestr, np.asarray(imp_states))
        if self.args.comet:
            self.args.experiment.log_image("%s.png"%(self.args.namestr), file_name= None, overwrite = False)


        if not self.args.silent:
            im=Image.open(img)
            im.show()

        
    def evaluate_policy(self):
        """
        evaluate the current policy and log it
        """
        avg_reward = []
        avg_len = []

        for _ in range(self.args.eval_n):
            state = self.env.reset()
            done = False
            ep_reward = 0
            ep_len = 0
            start_time = time.time()

            while not done:
                # convert the state to tensor
                state_tensor = torch.from_numpy(state).float().to(self.device).view(-1, self.state_dim)
                # get the policy action
                action = self.pi(state_tensor)
                next_state, reward, done, _ = self.env.step(action)
                ep_reward += reward
                ep_len += 1
                # update the state
                state = next_state
                done = done or ep_len >= self.args.max_path_len
            avg_reward.append(ep_reward)
            avg_len.append(ep_len)

        return np.mean(avg_reward), np.mean(avg_len)



    def save_models(self):
        """create results dict and save"""
        models = {
        "actor" : self.actor.state_dict(),
        "critic" : self.critic.state_dict(),
        }
        torch.save(models,os.path.join(self.args.out, 'models.pt'))


    def load_models(self):
        models = torch.load(os.path.join(self.args.out, 'models.pt'))
        self.actor.load_state_dict(models["actor"])
        self.critic.load_state_dict(models["critic"])
