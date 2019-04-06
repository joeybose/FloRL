import sys
import os
import torch
import ipdb
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Exponential, LogNormal, Laplace

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Policy weights
def weights_init_(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        x1 = torch.cat([state, action], 1)
        x1 = F.relu(self.linear1(x1))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = torch.cat([state, action], 1)
        x2 = F.relu(self.linear4(x2))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, args):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

    def encode(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def forward(self, state):
        mean, log_std = self.encode(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, x_t, mean, log_std

class ExponentialPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, args):
        super(ExponentialPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.rate_linear = nn.Linear(hidden_dim, num_actions)
        self.apply(weights_init_)

    def encode(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        log_rate = self.rate_linear(x)
        return log_rate

    def forward(self, state):
        log_rate = self.encode(state)
        log_rate = torch.clamp(log_rate, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        rate = torch.exp(log_rate)
        exponential = Exponential(rate)
        x_t = exponential.rsample()
        action = torch.tanh(x_t)
        log_prob = xponential.log_prob(x_t)
        mean = exponential.mean
        std = torch.sqrt(exponential.variance)
        log_std = torch.log(std)
        # Enforcing Action Bound
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, x_t, mean, log_std

class LogNormalPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, args):
        super(LogNormalPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

    def encode(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def forward(self, state):
        mean, log_std = self.encode(state)
        std = torch.exp(log_std)
        log_normal = LogNormal(mean, std)
        x_t = log_normal.rsample()
        action = torch.tanh(x_t)
        log_prob = log_normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, x_t, mean, log_std

class LaplacePolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, args):
        super(LaplacePolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_scale_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

    def encode(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_scale = self.log_scale_linear(x)
        log_scale = torch.clamp(log_scale, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_scale

    def forward(self, state):
        mean, log_scale = self.encode(state)
        scale = torch.exp(log_scale)
        laplace = Laplace(mean, scale)
        x_t = laplace.rsample()
        action = torch.tanh(x_t)
        log_prob = laplace.log_prob(x_t)
        std = torch.sqrt(laplace.variance)
        log_std = torch.log(std)
        # Enforcing Action Bound
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, x_t, mean, log_std

class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

    def encode(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x))
        return mean

    def forward(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), torch.tensor(0.), mean, torch.tensor(0.)

