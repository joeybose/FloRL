import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Normal
from flow_helpers import *
import ipdb

#Reference: https://github.com/ritheshkumar95/pytorch-normalizing-flows/blob/master/modules.py
# Initialize Policy weights
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6
def weights_init_(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class PlanarBase(nn.Module):
    def __init__(self, state_enc, n_blocks, state_size, input_size,
                 hidden_size, n_hidden, device, num_entropy_samples):
        super().__init__()
        self.l1 = nn.Linear(state_size, hidden_size)
        self.l2 = nn.Linear(hidden_size,hidden_size)
        self.mu = nn.Linear(hidden_size, input_size)
        self.log_std = nn.Linear(hidden_size, input_size)
        self.device = device
        self.z_size = input_size
        self.num_flows = n_blocks
        self.state_enc = state_enc
        self.flow = Planar
        self.N = num_entropy_samples
        # Amortized flow parameters
        self.amor_u = nn.Linear(hidden_size, self.num_flows * input_size)
        self.amor_w = nn.Linear(hidden_size, self.num_flows * input_size)
        self.amor_b = nn.Linear(hidden_size, self.num_flows)

        # Normalizing flow layers
        for k in range(self.num_flows):
            flow_k = self.flow()
            self.add_module('flow_' + str(k), flow_k)

        self.apply(weights_init_)

    def encode(self, state):
        # x = self.state_enc(state)
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        mean = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        mean = torch.clamp(mean, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std, x
        # return x

    def forward(self, state):
        batch_size = state.size(0)
        mean, log_std, hidden_state = self.encode(state)
        std = log_std.exp()
        # mean = torch.zeros(batch_size,self.z_size).to(self.device)
        # std = torch.ones(batch_size,self.z_size).to(self.device)
        # encoded_state, hidden_state = self.encode(state)
        normal = Normal(mean, std)
        noise = normal.rsample()
        action = noise
        log_prob_prior = normal.log_prob(noise)
        z = [action]
        u = self.amor_u(hidden_state).view(batch_size, self.num_flows, self.z_size, 1)
        w = self.amor_w(hidden_state).view(batch_size, self.num_flows, 1, self.z_size)
        b = self.amor_b(hidden_state).view(batch_size, self.num_flows, 1, 1)

        self.log_det_j = torch.zeros(batch_size).to(self.device)

        for k in range(self.num_flows):
            flow_k = getattr(self, 'flow_' + str(k))
            # if k == 1:
                # z_k, log_det_jacobian = flow_k(z[k], u[:, k, :, :],\
                                               # w[:, k, :, :], b[:, k, :,:],\
                                               # encoded_state=encoded_state)
            # else:
            z_k, log_det_jacobian = flow_k(z[k], u[:, k, :, :],\
                                           w[:, k, :, :], b[:, k, :,:])
            z.append(z_k)
            self.log_det_j += log_det_jacobian

        action = z[-1]
        # log_prob_final_action = log_prob_prior.squeeze() - self.log_det_j
        log_prob_final_action = log_prob_prior

        # normalized_action = torch.tanh(action)
        # Enforcing Action Bound
        # log_prob_final_action -= torch.log(1 - action.pow(2) + epsilon)
        log_prob_final_action = log_prob_final_action.sum(1, keepdim=True)
        log_prob_final_action -= self.log_det_j.unsqueeze(1)
        np_action = action.cpu().data.numpy().flatten()
        if np.isnan(np_action[0]):
            ipdb.set_trace()
        # return normalized_action, log_prob_final_action, action , mean, log_std
        return action, log_prob_final_action, action , mean, log_std

    def calc_entropy(self,state):
        mean_log_prob = 0
        for i in range(0,self.N):
            _, log_prob, _ , _ , _ = self.forward(state)
            mean_log_prob += log_prob
        mean_log_prob = mean_log_prob / self.N
        return mean_log_prob

class PlanarFlow(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.D = D

    def forward(self, z, lamda):
        '''
        z - latents from prev layer
        lambda - Flow parameters (b, w, u)
        b - scalar
        w - vector
        u - vector
        '''
        b = lamda[:, :1]
        w, u = lamda[:, 1:].chunk(2, dim=1)

        # Forward
        # f(z) = z + u tanh(w^T z + b)
        transf = F.tanh(
            z.unsqueeze(1).bmm(w.unsqueeze(2))[:, 0] + b
        )
        f_z = z + u * transf

        # Inverse
        # psi_z = tanh' (w^T z + b) w
        psi_z = (1 - transf ** 2) * w
        log_abs_det_jacobian = torch.log(
            (1 + psi_z.unsqueeze(1).bmm(u.unsqueeze(2))).abs()
        )

        return f_z, log_abs_det_jacobian

class NormalizingFlow(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.flows = nn.ModuleList([PlanarFlow(D) for i in range(K)])

    def forward(self, z_k, flow_params):
        # ladj -> log abs det jacobian
        sum_ladj = 0
        for i, flow in enumerate(self.flows):
            z_k, ladj_k = flow(z_k, flow_params[i])
            sum_ladj += ladj_k

        return z_k, sum_ladj

class Planar(nn.Module):
    """
    PyTorch implementation of planar flows as presented in "Variational Inference with Normalizing Flows"
    by Danilo Jimenez Rezende, Shakir Mohamed. Model assumes amortized flow parameters.
    """

    def __init__(self):

        super(Planar, self).__init__()

        self.h = nn.Tanh()
        self.softplus = nn.Softplus()

    def der_h(self, x):
        """ Derivative of tanh """

        return 1 - self.h(x) ** 2

    def forward(self, zk, u, w, b, encoded_state=None):
        """
        Forward pass. Assumes amortized u, w and b. Conditions on diagonals of u and w for invertibility
        will be be satisfied inside this function. Computes the following transformation:
        z' = z + u h( w^T z + b)
        or actually
        z'^T = z^T + h(z^T w + b)u^T
        Assumes the following input shapes:
        shape u = (batch_size, z_size, 1)
        shape w = (batch_size, 1, z_size)
        shape b = (batch_size, 1, 1)
        shape z = (batch_size, z_size).
        """
        if encoded_state is not None:
            zk = zk + encoded_state

        zk = zk.unsqueeze(2)

        # reparameterize u such that the flow becomes invertible (see appendix paper)
        uw = torch.bmm(w, u)
        m_uw = -1. + self.softplus(uw)
        w_norm_sq = torch.sum(w ** 2, dim=2, keepdim=True)
        u_hat = u + ((m_uw - uw) * w.transpose(2, 1) / w_norm_sq)

        # compute flow with u_hat
        wzb = torch.bmm(w, zk) + b
        z = zk + u_hat * self.h(wzb)
        z = z.squeeze(2)

        # compute logdetJ
        if encoded_state is not None:
            wzb = torch.bmm(w, zk - encoded_state.unsqueeze(2)) + b
            psi = w * self.der_h(wzb)
        else:
            psi = w * self.der_h(wzb)
        log_det_jacobian = torch.log(torch.abs(1 + torch.bmm(psi, u_hat)))
        log_det_jacobian = log_det_jacobian.squeeze(2).squeeze(1)

        return z, log_det_jacobian

# All code below this line is taken from
# https://github.com/kamenbliznashki/normalizing_flows/blob/master/maf.py

class FlowSequential(nn.Sequential):
    """ Container for layers of a normalizing flow """
    def forward(self, x, y):
        sum_log_abs_det_jacobians = 0
        for module in self:
            x, log_abs_det_jacobian = module(x, y)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
        return x, sum_log_abs_det_jacobians

    def inverse(self, u, y):
        sum_log_abs_det_jacobians = 0
        for module in reversed(self):
            u, log_abs_det_jacobian = module.inverse(u, y)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
        return u, sum_log_abs_det_jacobians

# --------------------
# Models
# --------------------

class MADE(nn.Module):
    def __init__(self, input_size, hidden_size, n_hidden, cond_label_size=None, activation='relu', input_order='sequential', input_degrees=None):
        """
        Args:
            input_size -- scalar; dim of inputs
            hidden_size -- scalar; dim of hidden layers
            n_hidden -- scalar; number of hidden layers
            activation -- str; activation function to use
            input_order -- str or tensor; variable order for creating the autoregressive masks (sequential|random)
                            or the order flipped from the previous layer in a stack of mades
            conditional -- bool; whether model is conditional
        """
        super().__init__()
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))

        # create masks
        masks, self.input_degrees = create_masks(input_size, hidden_size, n_hidden, input_order, input_degrees)

        # setup activation
        if activation == 'relu':
            activation_fn = nn.ReLU()
        elif activation == 'tanh':
            activation_fn = nn.Tanh()
        else:
            raise ValueError('Check activation function.')

        # construct model
        self.net_input = MaskedLinear(input_size, hidden_size, masks[0], cond_label_size)
        self.net = []
        for m in masks[1:-1]:
            self.net += [activation_fn, MaskedLinear(hidden_size, hidden_size, m)]
        self.net += [activation_fn, MaskedLinear(hidden_size, 2 * input_size, masks[-1].repeat(2,1))]
        self.net = nn.Sequential(*self.net)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        # MAF eq 4 -- return mean and log std
        m, loga = self.net(self.net_input(x, y)).chunk(chunks=2, dim=1)
        u = (x - m) * torch.exp(-loga)
        # MAF eq 5
        log_abs_det_jacobian = - loga
        return u, log_abs_det_jacobian

    def inverse(self, u, y=None, sum_log_abs_det_jacobians=None):
        # MAF eq 3
        D = u.shape[1]
        x = torch.zeros_like(u)
        # run through reverse model
        for i in self.input_degrees:
            m, loga = self.net(self.net_input(x, y)).chunk(chunks=2, dim=1)
            x[:,i] = u[:,i] * torch.exp(loga[:,i]) + m[:,i]
        log_abs_det_jacobian = -loga
        return x, log_abs_det_jacobian

    def log_prob(self, x, y=None):
        u, log_abs_det_jacobian = self.forward(x, y)
        return torch.sum(self.base_dist.log_prob(u) + log_abs_det_jacobian, dim=1)


class MADEMOG(nn.Module):
    """ Mixture of Gaussians MADE """
    def __init__(self, n_components, input_size, hidden_size, n_hidden, cond_label_size=None, activation='relu', input_order='sequential', input_degrees=None):
        """
        Args:
            n_components -- scalar; number of gauassian components in the mixture
            input_size -- scalar; dim of inputs
            hidden_size -- scalar; dim of hidden layers
            n_hidden -- scalar; number of hidden layers
            activation -- str; activation function to use
            input_order -- str or tensor; variable order for creating the autoregressive masks (sequential|random)
                            or the order flipped from the previous layer in a stack of mades
            conditional -- bool; whether model is conditional
        """
        super().__init__()
        self.n_components = n_components

        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))

        # create masks
        masks, self.input_degrees = create_masks(input_size, hidden_size, n_hidden, input_order, input_degrees)

        # setup activation
        if activation == 'relu':
            activation_fn = nn.ReLU()
        elif activation == 'tanh':
            activation_fn = nn.Tanh()
        else:
            raise ValueError('Check activation function.')

        # construct model
        self.net_input = MaskedLinear(input_size, hidden_size, masks[0], cond_label_size)
        self.net = []
        for m in masks[1:-1]:
            self.net += [activation_fn, MaskedLinear(hidden_size, hidden_size, m)]
        self.net += [activation_fn, MaskedLinear(hidden_size, n_components * 3 * input_size, masks[-1].repeat(n_components * 3,1))]
        self.net = nn.Sequential(*self.net)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        # shapes
        N, L = x.shape
        C = self.n_components
        # MAF eq 2 -- parameters of Gaussians - mean, logsigma, log unnormalized cluster probabilities
        m, loga, logr = self.net(self.net_input(x, y)).view(N, C, 3 * L).chunk(chunks=3, dim=-1)  # out 3 x (N, C, L)
        # MAF eq 4
        x = x.repeat(1, C).view(N, C, L)  # out (N, C, L)
        u = (x - m) * torch.exp(-loga)  # out (N, C, L)
        # MAF eq 5
        log_abs_det_jacobian = - loga  # out (N, C, L)
        # normalize cluster responsibilities
        self.logr = logr - logr.logsumexp(1, keepdim=True)  # out (N, C, L)
        return u, log_abs_det_jacobian

    def inverse(self, u, y=None, sum_log_abs_det_jacobians=None):
        # shapes
        N, C, L = u.shape
        # init output
        x = torch.zeros(N, L).to(u.device)
        # MAF eq 3
        # run through reverse model along each L
        for i in self.input_degrees:
            m, loga, logr = self.net(self.net_input(x, y)).view(N, C, 3 * L).chunk(chunks=3, dim=-1)  # out 3 x (N, C, L)
            # normalize cluster responsibilities and sample cluster assignments from a categorical dist
            logr = logr - logr.logsumexp(1, keepdim=True)  # out (N, C, L)
            z = D.Categorical(logits=logr[:,:,i]).sample().unsqueeze(-1)  # out (N, 1)
            u_z = torch.gather(u[:,:,i], 1, z).squeeze()  # out (N, 1)
            m_z = torch.gather(m[:,:,i], 1, z).squeeze()  # out (N, 1)
            loga_z = torch.gather(loga[:,:,i], 1, z).squeeze()
            x[:,i] = u_z * torch.exp(loga_z) + m_z
        log_abs_det_jacobian = - loga
        return x, log_abs_det_jacobian

    def log_prob(self, x, y=None):
        u, log_abs_det_jacobian = self.forward(x, y)  # u = (N,C,L); log_abs_det_jacobian = (N,C,L)
        # marginalize cluster probs
        log_probs = torch.logsumexp(self.logr + self.base_dist.log_prob(u) + log_abs_det_jacobian, dim=1)  # sum over C; out (N, L)
        return log_probs.sum(1)  # sum over L; out (N,)


class MAF(nn.Module):
    def __init__(self, n_blocks, state_size, input_size, hidden_size, n_hidden,
                 cond_label_size=None, activation='relu',
                 input_order='sequential', batch_norm=True):
        super().__init__()
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))
        self.linear1 = nn.Linear(state_size, input_size)

        # construct model
        modules = []
        self.input_degrees = None
        for i in range(n_blocks):
            modules += [MADE(input_size, hidden_size, n_hidden,
                             cond_label_size, activation, input_order,
                             self.input_degrees)]
            self.input_degrees = modules[-1].input_degrees.flip(0)
            modules += batch_norm * [BatchNorm(input_size)]

        self.net = FlowSequential(*modules)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        ''' Projecting the State to the same dim as actions '''
        action_proj = F.relu(self.linear1(x))
        # action_proj = action_proj.view(1,-1)
        if action_proj.size()[0] == 1 and len(action_proj.size()) > 2:
            action, sum_log_abs_det_jacobians = self.net(action_proj[0], y)
        else:
            action, sum_log_abs_det_jacobians = self.net(action_proj, y)
        log_prob = torch.sum(self.base_dist.log_prob(action) + sum_log_abs_det_jacobians, dim=1)
        normalized_action = torch.tanh(action)
        # TODO: Find the mean and log std deviation of a Normalizing Flow
        return normalized_action, log_prob, action , action, 0

    def inverse(self, u, y=None):
        action_proj = F.relu(self.linear1(u))
        action, sum_log_abs_det_jacobians = self.net.inverse(action_proj, y)
        log_prob = torch.sum(self.base_dist.log_prob(action) + sum_log_abs_det_jacobians, dim=1)
        normalized_action = torch.tanh(action)
        return normalized_action, log_prob, action , action, 0
        # return self.net.inverse(action_proj, y)

    def log_prob(self, x, y=None):
        u, sum_log_abs_det_jacobians = self.forward(x, y)
        return torch.sum(self.base_dist.log_prob(u) + sum_log_abs_det_jacobians, dim=1)

class MAFMOG(nn.Module):
    """ MAF on mixture of gaussian MADE """
    def __init__(self, n_blocks, n_components, input_size, hidden_size, n_hidden, cond_label_size=None, activation='relu',
                 input_order='sequential', batch_norm=True):
        super().__init__()
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))

        self.maf = MAF(n_blocks, input_size, hidden_size, n_hidden, cond_label_size, activation, input_order, batch_norm)
        # get reversed input order from the last layer (note in maf model, input_degrees are already flipped in for-loop model constructor
        input_degrees = self.maf.input_degrees#.flip(0)
        self.mademog = MADEMOG(n_components, input_size, hidden_size, n_hidden, cond_label_size, activation, input_order, input_degrees)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        u, maf_log_abs_dets = self.maf(x, y)
        u, made_log_abs_dets = self.mademog(u, y)
        sum_log_abs_det_jacobians = maf_log_abs_dets.unsqueeze(1) + made_log_abs_dets
        return u, sum_log_abs_det_jacobians

    def inverse(self, u, y=None):
        x, made_log_abs_dets = self.mademog.inverse(u, y)
        x, maf_log_abs_dets = self.maf.inverse(x, y)
        sum_log_abs_det_jacobians = maf_log_abs_dets.unsqueeze(1) + made_log_abs_dets
        return x, sum_log_abs_det_jacobians

    def log_prob(self, x, y=None):
        u, log_abs_det_jacobian = self.forward(x, y)  # u = (N,C,L); log_abs_det_jacobian = (N,C,L)
        # marginalize cluster probs
        log_probs = torch.logsumexp(self.mademog.logr + self.base_dist.log_prob(u) + log_abs_det_jacobian, dim=1)  # out (N, L)
        return log_probs.sum(1)  # out (N,)


class RealNVP(nn.Module):
    def __init__(self, n_blocks, input_size, hidden_size, n_hidden, cond_label_size=None, batch_norm=True):
        super().__init__()

        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))

        # construct model
        modules = []
        mask = torch.arange(input_size).float() % 2
        for i in range(n_blocks):
            modules += [LinearMaskedCoupling(input_size, hidden_size, n_hidden, mask, cond_label_size)]
            mask = 1 - mask
            modules += batch_norm * [BatchNorm(input_size)]

        self.net = FlowSequential(*modules)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        return self.net(x, y)

    def inverse(self, u, y=None):
        return self.net.inverse(u, y)

    def log_prob(self, x, y=None):
        u, sum_log_abs_det_jacobians = self.forward(x, y)
        return torch.sum(self.base_dist.log_prob(u) + sum_log_abs_det_jacobians, dim=1)

