"""
Standardized evaluation as described in Khetarpal, Khimya, et al.
"RE-EVALUATE: Reproducibility in Evaluating Reinforcement Learning
Algorithms." (2018).
"""
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Evaluator:
    def __init__(self, env, wrap_function, n_episodes=10):
        """
        The evaluator object provides a standard interface for evaluating
        agents in an environment.

        Args:
            env: The gym environment to evaluate.
            wrap_function: A function that takes in a policy/model and returns
                a callabe function that takes in a unmodified state from the
                environment and returns an action suitable to be executed in
                the env. Wrapping functions are meant to standardize interfaces
                to agents implemented in different ways.
            n_episodes: The number of episodes to evaluate for.
        """
        self.env = env
        self.n_episodes = n_episodes
        self.wrap_function = wrap_function

    def evaluate_policy(self, policy, pbar=None):
        """
        Evaluate the policy in the environment.

        Args:
            policy: The policy to evaluate.
            pbar: The pbar object.
        """
        policy_wrapped = self.wrap_function(policy)
        with torch.no_grad():
            return self._evaluate_policy(policy_wrapped, pbar)

    def _evaluate_policy(self, policy, pbar=None):
        """
        Args:
            policy: The wrapped policy that takes in a state from the
                environment returns an action suitable to execute in the env.
            pbar: The progress bar object.
        """
        avg_reward = 0
        for _ in range(self.n_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = policy(state)
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                avg_reward += reward
        avg_reward /= self.n_episodes

        if pbar is not None:
            msg = "Eval {} episodes {}".format(self.n_episodes, avg_reward)
            pbar.print_end_epoch(msg)
        return avg_reward

def wrap_acer_policy(model):
    """Standardize the call to the acer policy."""
    def get_action(state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        policy, _, _ = model(state)
        action = policy.argmax().cpu()
        return action.item()
    return get_action

def wrap_dqn_policy(policy, epsilon=0.001):
    """Standardize the call the the DQN policy."""
    def get_action(state):
        action = policy.act(state, epsilon)
        return action
    return get_action

def wrap_discrete_td3_policy(policy):
    """Standardize the call to the TD3 policy."""
    def get_action(state):
        action = policy.deterministic_action(np.array(state))
        return action
    return get_action

