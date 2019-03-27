from comet_ml import Experiment
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import mujoco_py
import os
import gym
class Policy(nn.Module):

    def __init__(self, num_inputs, hidden_size, action_space):

        self.action_space = action_space #
        num_outputs = action_space.shape[0] # the number of output actions

        self.linear = nn.Linear(num_inputs, hidden_size)
        self.mean = nn.Linear(hidden_size, num_outputs)
        self.std = nn.Linear(hidden_size, num_outputs)

    def forward(self, inputs):

        x = inputs
        x = F.relu(self.linear(x))

        mu = self.mean(x)
        std = self.std(x) # if more an one action this will give you the diagonal elements of a diagonal covariance matrix

        return mu, std

class REINFORCE:

    def __init__(self, num_inputs, hidden_size, action_space):

        self.action_space = action_space
        self.policy = Policy(num_inputs, hidden_size, action_space)
        self.optimizer = optim.Adam(self.policy.parameters(), lr = 1e-2)

    def select_action(self,state):

        # get the normal distribution parameters from our model
        state = torch.from_numpy(state).float().unsqueeze(0)
        mu, std = policy(state)

        # sample from the normal distribution
        normal_dist = Normal(mu, std)
        sample_action = normal_dist.sample()

        # get the log prob of this action
        ln_prob = normal_dist.log_prob(sample_action) # could be nan watch out

        return sample_action, ln_prob



if __name__ == "__main__":


    print("hello")

    policy = Policy()
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)



#experiment = Experiment(api_key="BUXbNT79Q2PEtRkuX9swzxspZ",
#                                                project_name="general",
#                        workspace="nadeem-ward")

#print(env.action_space)


'''


for i_episode in range(20):
    print("here")
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
'''
#    mj_path, _ = mujoco_py.utils.discover_mujoco()
# xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')# model = mujoco_py.load_model_from_path(xml_path)
# sim = mujoco_py.MjSim(model)

# env = gym.make("InvertedPendulum-v1")
# env.reset()
# env.render()

