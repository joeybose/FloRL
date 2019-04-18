
import numpy as np
from action_sampling import *


# action noise models
class ActionNoise(object):
    def reset(self):
        pass


class RandomWalkNoise(object):
    def __init__(self, action_dim, max_action_limit):
        self.action_dim = action_dim
        self.max_action_limit = max_action_limit

    def __call__(self):
        return np.around(np.random.uniform(-self.max_action_limit,+self.max_action_limit, (self.action_dim,)), decimals = 10) 


class NormalActionNoise(ActionNoise):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        return np.random.normal(self.mu, self.sigma)

    def __repr__(self):
        return 'NormalActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise(ActionNoise):
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)




# TODO: Adaptive param noise: https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
class AdaptiveParamNoiseSpec(object):
    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.1, adoption_coefficient=1.01):
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adoption_coefficient = adoption_coefficient

        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance > self.desired_action_stddev:
            # Decrease stddev.
            self.current_stddev /= self.adoption_coefficient
        else:
            # Increase stddev.
            self.current_stddev *= self.adoption_coefficient

    def get_stats(self):
        stats = {
            'param_noise_stddev': self.current_stddev,
        }
        return stats

    def __repr__(self):
        fmt = 'AdaptiveParamNoiseSpec(initial_stddev={}, desired_action_stddev={}, adoption_coefficient={})'
        return fmt.format(self.initial_stddev, self.desired_action_stddev, self.adoption_coefficient)



class PolyNoise(object):
    def __init__(self,
                 L_p,
                 b_0,
                 action_dim,
                 ou_noise,
                 sigma = 0.2,):
        """
        params for the L_p formulation:
        L_p: the persistance length
        b_0: movement distance
        signma: correlaion_variance
        blind: disregard the current action and only use the previous action
        """
        action_noise = NormalActionNoise(mu=np.zeros(action_dim), sigma=float(sigma) * np.ones(action_dim))
        self.L_p = L_p
        self.b_0 = b_0
        self.sigma = sigma
        self.action_dim = action_dim
        self.ou_noise = ou_noise

        # calculate the angle here
        self.n = int(L_p/b_0)
        self.lambda_ = np.arccos(np.exp((-1. * b_0)/L_p))
        # initialize and reset traj-specific stats
        self.reset()


    def reset(self):
        """
        reset the chain history
        """
        self.H = None
        self.a_p = None
        #self.ou_noise.reset()

        self.i = 0
        self.t = 0
        self.rand_or_poly = []


    def __call__(self, a):
        """
        the
        s: the current state
        a: the current action
        t: the time step
        """
        new_a = a

        if self.t==0:
            # return original a
            pass
        elif self.t==1:
            # create randonm trajectory vector
            H = np.random.rand(self.action_dim)
            self.H = (H * self.b_0) / np.linalg.norm(H, 2)
            # append the new H to the previous actions
            new_a = self.a_p + self.H
            self.i += 1
        else:
            # done with polyRL noise
            if self.i == self.n:
                # intialize
                noise = self.ou_noise()

                # add the noise
                new_a = a + noise

                #rest i and H
                self.i = 0
                self.H = new_a - self.a_p
                self.rand_or_poly.append(False)
            else:
                eta = abs(np.random.normal(self.lambda_, self.sigma, 1))
                B = sample_persistent_action(self.action_dim, self.H, self.a_p, eta)
                self.rand_or_poly.append(True)

                #update the trajectory
                self.H = self.b_0 * B

                new_a = self.a_p + self.H
                self.i += 1

        #update the previous a_p
        self.a_p = new_a
        self.t += 1



        return new_a


class GyroPolyNoise(object):
    def __init__(self,
                 L_p,
                 b_0,
                 action_dim,
                 state_dim,
                 ou_noise,
                 sigma = 0.2,):
        """
        params for the L_p formulation:
        L_p: the persistance length
        b_0: movement distance
        signma: correlaion_variance
        blind: disregard the current action and only use the previous action
        """
        self.L_p = L_p
        self.b_0 = b_0
        self.sigma = sigma
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.ou_noise = ou_noise

        # calculate the angle here
        self.n = int(L_p/b_0)
        self.lambda_ = np.arccos(np.exp((-1. * b_0)/L_p))
        # initialize and reset traj-specific stats
        self.reset()


    def reset(self):
        """
        reset the chain history
        """
        self.H = None
        self.a_p = None
        self.ou_noise.reset()

        self.i = 0
        self.t = 0

        # raius of gyration
        self.g = 0
        self.delta_g = 0

        # centre of mass of gyration
        self.C = np.zeros(self.state_dim)


        self.rand_or_poly = []
        self.g_history = []
        self.avg_delta_g = 0


    def __call__(self, a, s):
        """
        the
        s: the current state
        a: the current action
        t: the time step
        """
        new_a = a

        if self.t==0:
            # return original a
            pass
        elif self.t==1:
            # create randonm trajectory vector
            H = np.random.rand(self.action_dim)
            self.H = (H * self.b_0) / np.linalg.norm(H, 2)

            # append the new H to the previous actions
            new_a = self.a_p + self.H
            self.i += 1
        else:
            # done with polyRL noise
            if self.delta_g < 0:
                # intialize
                noise = self.ou_noise()

                # add the noise
                new_a = a + noise

                #rest i and H
                self.i = 0
                self.g = 0
                self.C = np.zeros(self.state_dim)
                self.delta_g = 0
                self.H = new_a - self.a_p
                self.rand_or_poly.append(False)
            else:
                eta = abs(np.random.normal(self.lambda_, self.sigma, 1))
                B = sample_persistent_action(self.action_dim, self.H, self.a_p, eta)
                self.rand_or_poly.append(True)

                #update the trajectory
                self.H = self.b_0 * B

                new_a = self.a_p + self.H

                if self.i == self.n:
                    self.i = 0
                    self.g = 0
                    self.C = np.zeros(self.state_dim)
                    self.delta_g = 0
                else:
                    self.i += 1

        #update the previous a_p
        self.a_p = new_a

        if self.i != 0:
            g = np.sqrt(((float(self.i-1.0)/self.i) * self.g**2) + (1.0/(self.i+1.0) * np.linalg.norm(s - self.C, 2)**2) )
            self.delta_g = g - self.g
            self.g = g

            # add to history
            self.avg_delta_g += self.delta_g
            self.g_history.append(self.g)


        self.C = (self.i * self.C + s)/(self.i + 1.0)
        self.t += 1


        return new_a
class GyroPolyNoiseActionTraj (object):
    def __init__(self,
                 lambd,
                 action_dim,
                 state_dim,
                 ou_noise,
                 sigma = 0.2,
                 max_action_limit = 1.0):
        self.lambd = lambd
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.ou_noise = ou_noise
        self.sigma = sigma
        self.max_action_limit = max_action_limit


        # initialize and reset traj-specific stats
        self.reset()

    def reset(self):

        """
        reset the chain history
        """
        self.a_p = None
        self.ou_noise.reset()

        self.i = 0
        self.t = 0

        # raius of gyration
        self.g = 0
        self.delta_g = 0

        # centre of mass 
        self.C = np.zeros(self.state_dim)


        self.rand_or_poly = []
        self.g_history = []
        self.avg_delta_g = 0


    def __call__(self, a, s):
        """
        the
        s: the current state
        a: the current action
        t: the time step
        """
        new_a = a

        if self.t==0:
            # return original a
            pass
        else:
            # done with polyRL noise
            if self.delta_g < 0:
                # intialize
                noise = self.ou_noise()

                # add the noise
                new_a = a + noise

                #rest i and H
                self.i = 0
                self.g = 0
                self.C = np.zeros(self.state_dim)
                self.delta_g = 0
                self.rand_or_poly.append(False)
            else:
                eta = abs(np.random.normal(self.lambd, self.sigma, 1))
                A = sample_persistent_action_noHvector(self.action_dim, self.a_p, eta, self.max_action_limit)
                self.rand_or_poly.append(True)

                #update the trajectory
                new_a = A

                self.i +=1


        #update the previous a_p
        self.a_p = new_a

        if self.i != 0:
            g = np.sqrt(((float(self.i-1.0)/self.i) * self.g**2) + (1.0/(self.i+1.0) * np.linalg.norm(s - self.C, 2)**2) )
            self.delta_g = g - self.g
            self.g = g

            # add to history
            self.avg_delta_g += self.delta_g
            self.g_history.append(self.g)


        self.C = (self.i * self.C + s)/(self.i + 1.0)
        self.t += 1
        return new_a

