"""
Implementations of replay buffers.
"""
import random
from collections import deque

import numpy as np

class ReplayBuffer(object):
    """
    Used in DQN.
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        (state, action, reward,
         next_state, done) = zip(*random.sample(self.buffer, batch_size))
        return (np.concatenate(state),
                action,
                reward,
                np.concatenate(next_state),
                done)

    def __len__(self):
        return len(self.buffer)


class EpisodicReplayMemory(object):
    """
    Used in ACER.
    """
    def __init__(self, capacity, max_episode_length):
        self.num_episodes = capacity // max_episode_length
        self.buffer = deque(maxlen=self.num_episodes)
        self.position = 0
        self._insert_new = True

    def push(self, state, action, reward, policy, mask, done):
        if self._insert_new:
            self.buffer.append([])
            self._insert_new = False
        self.buffer[self.position].append((state, action, reward, policy, mask))
        if done:
            self._insert_new = True
            self.position = min(self.position + 1, self.num_episodes - 1)

    def sample(self, batch_size, max_len=None):
        min_len = 0
        while min_len == 0:
            rand_episodes = random.sample(
                self.buffer, min(batch_size, len(self.buffer)))
            min_len = min(len(episode) for episode in rand_episodes)

        if max_len:
            max_len = min(max_len, min_len)
        else:
            max_len = min_len

        episodes = []
        for episode in rand_episodes:
            if len(episode) > max_len:
                rand_idx = random.randint(0, len(episode) - max_len)
            else:
                rand_idx = 0

            episodes.append(episode[rand_idx:rand_idx + max_len])

        return list(map(list, zip(*episodes)))

    def __len__(self):
        return len(self.buffer)
