"""
Implementaiton of ACER.
"""
import math

import numpy as np
import torch

from benchmarks.classic.neural_networks import ActorCritic  # pylint: disable=unused-import
from benchmarks.classic.replay_buffers import EpisodicReplayMemory  # pylint: disable=unused-import

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def compute_acer_loss(
        policies, q_values, values, actions, rewards, retrace,
        masks, behavior_policies, gamma=0.99, truncation_clip=10,
        entropy_weight=0.0001):
    """
    Compute the loss for ACER.

    Args:
        policies: Policy at each step.
        q_values:
        values:
        actions:
        rewards:
        retrace:
        masks:
        behavior_policies:
        gamma:
        truncation_clip:
        entropy_weight:
    """
    loss = 0

    for step in reversed(range(len(rewards))):
        importance_weight = policies[step].detach() / behavior_policies[
            step].detach()
        retrace = rewards[step] + gamma * retrace * masks[step]
        advantage = retrace - values[step]

        log_policy_action = policies[step].gather(1, actions[step]).log()
        truncated_importance_weight = importance_weight.gather(1, actions[
            step]).clamp(max=truncation_clip)
        actor_loss = -(
        truncated_importance_weight * log_policy_action * advantage.detach()).mean(
            0)

        correction_weight = (1 - truncation_clip / importance_weight).clamp(
            min=0)
        actor_loss -= (correction_weight * policies[step].log() * (
        q_values[step] - values[step]).detach()).sum(1).mean(0)

        entropy = entropy_weight * -(policies[step].log() * policies[step]).sum(
            1).mean(0)

        q_value = q_values[step].gather(1, actions[step])
        critic_loss = ((retrace - q_value) ** 2 / 2).mean(0)

        truncated_rho = importance_weight.gather(1, actions[step]).clamp(max=1)
        retrace = truncated_rho * (retrace - q_value.detach()) + values[
            step].detach()

        loss += actor_loss + critic_loss - entropy
        if torch.isnan(loss):
            print('NAN LOSS!!!!!!')

    return loss


def off_policy_update(
        replay_buffer, model, batch_size, optimizer,
        replay_ratio=4, gamma=0.99, entropy_weight=0.0001):
    """
    Make an ACER off-policy update.

    Args:
        replay_buffer:
        model:
        batch_size:
        replay_ratio: TODO
    """
    if batch_size > len(replay_buffer) + 1:
        return

    for _ in range(np.random.poisson(replay_ratio)):
        trajs = replay_buffer.sample(batch_size)
        state, action, reward, old_policy, mask = map(torch.stack, zip(
            *(map(torch.cat, zip(*traj)) for traj in trajs)))

        q_values = []
        values = []
        policies = []

        for step in range(state.size(0)):
            policy, q_value, value = model(state[step])
            q_values.append(q_value)
            policies.append(policy)
            values.append(value)

        _, _, retrace = model(state[-1])
        retrace = retrace.detach()
        loss = compute_acer_loss(
                policies, q_values, values, action,
                reward, retrace, mask, old_policy, gamma=gamma,
                entropy_weight=entropy_weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

