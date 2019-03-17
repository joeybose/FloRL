"""
Parts for the DQN and double DQN algorithm.
"""
import math

import numpy as np
import torch

from benchmarks.classic.neural_networks import DQN  # pylint: disable=unused-import
from benchmarks.classic.replay_buffers import ReplayBuffer  # pylint: disable=unused-import

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_epsilon_by_frame_function(epsilon_final, epsilon_start, epsilon_decay):
    """Returns a function that gives the epsilon to use in each frame."""
    delta_epsilon = epsilon_start - epsilon_final
    def epsilon_by_frame(frame_idx):
        decay_magnitude = math.exp(-1. * frame_idx / epsilon_decay)
        return epsilon_final + delta_epsilon * decay_magnitude
    return epsilon_by_frame


def update_target(current_model, target_model):
    """ Update the target model using the current.
    Args:
        current_model: The current model.
        target_model: The target model.
    """
    target_model.load_state_dict(current_model.state_dict())



def soft_update(local_model, target_model, tau=1e-3):
    """Soft update model parameters.
    target_model = tau * local_model + (1 - tau) * target_model

    Params
    ======
        local_model (PyTorch model): weights will be copied from
        target_model (PyTorch model): weights will be copied to
        tau (float): interpolation parameter 
    """
    ##tau = 1e-3              # for soft update of target parameters
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


def compute_td_loss(
        replay_buffer,
        model,
        batch_size=32,
        gamma=0.99):
    """Computes the DQN TD Loss from the replay_buffer.

    Args:
        replay_buffer: The replay_buffer object to sample from.
        model: The current model for Q-values.
        batch_size: The size of the batch to sample.
    """
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = torch.FloatTensor(np.float32(state)).to(device)
    next_state = torch.FloatTensor(np.float32(next_state)).to(device)
    action = torch.LongTensor(action).to(device)
    reward = torch.FloatTensor(reward).to(device)
    done = torch.FloatTensor(done).to(device)

    q_values = model(state)
    next_q_values = model(next_state)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = (reward + gamma * next_q_value * (1 - done)).detach()

    loss = (q_value - expected_q_value).pow(2).mean()

    return loss


def compute_double_td_loss(
        replay_buffer,
        current_model,
        target_model,
        batch_size=32,
        gamma=0.99):
    """Computes the double DQN TD Loss from the replay_buffer.

    Args:
        replay_buffer: The replay_buffer object to sample from.
        current_model: The current model for Q-values.
        target_model: the target model for Q-values.
        batch_size: The size of the batch to sample.
    """
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = torch.FloatTensor(np.float32(state)).to(device)
    next_state = torch.FloatTensor(np.float32(next_state)).to(device)
    action = torch.LongTensor(action).to(device)
    reward = torch.FloatTensor(reward).to(device)
    done = torch.FloatTensor(done).to(device)

    q_values = current_model(state)
    next_q_values = current_model(next_state)
    next_q_state_values = target_model(next_state)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_state_values.gather(
        1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)

    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = (q_value - expected_q_value.detach()).pow(2).mean()


    return loss
