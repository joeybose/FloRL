import random

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    """
    Simple Actor Critic implementation.
    """
    def __init__(self, num_inputs, num_actions, hidden_size=400):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
            nn.Softmax(dim=1)
        )

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
        )

    def forward(self, x):
        policy = self.actor(x).clamp(
                max=1-1e-20,
                min=1e-20)
        q_value = self.critic(x)
        value= (policy * q_value).sum(-1, keepdim=True)
        return policy, q_value, value


class DQN(nn.Module):
    """
    Implementation of Deep Q Network.
    """
    def __init__(self, num_inputs, num_actions, hidden_size=256):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), #used 512 here before
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)) #used 512 here before

        self.num_actions = num_actions

    def forward(self, x):
        return self.layers(x)

    def act(self, state, epsilon):
        with torch.no_grad():
            # Epsilon Greedy.
            if random.random() > epsilon:
                state = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_value = self.forward(state)
                action = q_value.max(1)[1].data.cpu().numpy()
                action = action[0]

            else:
                action = random.randrange(self.num_actions)

        return action
