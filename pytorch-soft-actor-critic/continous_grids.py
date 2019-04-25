"""
code from here: https://github.com/junhyukoh/value-prediction-network/blob/master/maze.py
"""

import copy
import pandas as pd

import seaborn as sns

import gym.spaces
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.pyplot as plt
import numpy as np
from exploration_models import *
from graphics import *
from gym import spaces


class GridWorld(gym.Env):
    """
    empty grid world
    """

    def __init__(self,
                 num_rooms=0,
                 start_position=(25.0, 25.0),
                 goal_position=(75.0, 75.0),
                 goal_reward=+100.0,
                 dense_goals=None,
                 dense_reward=+5,
                 goal_radius=1.0,
                 per_step_penalty=-0.01,
                 max_episode_len=1000,
                 grid_len=100,
                 wall_breadth=1,
                 door_breadth=5,
                 action_limit_max=1.0,
                 silent_mode=False):
        """
        params:
        """

        # num of rooms
        self.num_rooms = num_rooms
        self.silent_mode = silent_mode

        # grid size
        self.grid_len = float(grid_len)
        self.wall_breadth = float(wall_breadth)
        self.door_breadth = float(door_breadth)
        self.min_position = 0.0
        self.max_position = float(grid_len)

        # goal stats
        self.goal_position = np.array(goal_position)
        self.goal_radius = goal_radius
        self.start_position = np.array(start_position)

        # Dense reward stuff:
        self.dense_reward = dense_reward
        # List of dense goal coordinates
        self.dense_goals = dense_goals

        # rewards
        self.goal_reward = goal_reward
        self.per_step_penalty = per_step_penalty

        self.max_episode_len = max_episode_len

        # observation space
        self.low_state = np.array([self.min_position, self.min_position])
        self.high_state = np.array([self.max_position, self.max_position])

        # how much the agent can move in a step (dx,dy)
        self.min_action = np.array([-action_limit_max, -action_limit_max])
        self.max_action = np.array([+action_limit_max, +action_limit_max])

        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state)
        self.action_space = spaces.Box(low=self.min_action, high=self.max_action)
        self.nb_actions = self.action_space.shape[-1]

        # add the walls here
        self.create_walls()
        self.scale = 5

        # This code enables live visualization of trajectories
        # Susan added these lines for visual purposes
        if not self.silent_mode:
            self.win1 = GraphWin("2DGrid", self.max_position * self.scale + 40, self.max_position * self.scale + 40)
            rectangle1 = Rectangle(Point(self.min_position * self.scale + 20, self.min_position * self.scale + 20),
                                   Point(self.max_position * self.scale + 20, self.max_position * self.scale + 20))
            rectangle1.setOutline('red')
            rectangle1.draw(self.win1)

            if self.num_rooms > 0:
                wall1 = Rectangle(Point(self.min_position * self.scale + 20,
                                        self.max_position * self.scale / 2 + 20 - self.wall_breadth * self.scale),
                                  Point(self.max_position * self.scale / 2 + 20,
                                        self.max_position * self.scale / 2 + 20 + self.wall_breadth * self.scale))
                wall1.draw(self.win1)
                wall1.setFill('aquamarine')

                wall2 = Rectangle(Point(self.max_position * self.scale / 2 + 20 - self.wall_breadth * self.scale,
                                        self.min_position * self.scale + 20),
                                  Point(self.max_position * self.scale / 2 + 20 + self.wall_breadth * self.scale,
                                        self.max_position * self.scale / 4 + 20 - self.door_breadth * self.scale))
                wall2.draw(self.win1)
                wall2.setFill('aquamarine')

                wall3 = Rectangle(Point(self.max_position * self.scale / 2 + 20 - self.wall_breadth * self.scale,
                                        self.max_position * self.scale / 4 + 20 + self.door_breadth * self.scale),
                                  Point(self.max_position * self.scale / 2 + 20 + self.wall_breadth * self.scale,
                                        self.max_position * self.scale / 2 + 20 + self.wall_breadth * self.scale))
                wall3.draw(self.win1)
                wall3.setFill('aquamarine')
            start_point = Circle(Point(start_position[0] * self.scale + 20, start_position[1] * self.scale + 20),
                                 goal_radius * self.scale)
            start_point.draw(self.win1)
            start_point.setFill('red')
            goal_point = Circle(Point(goal_position[0] * self.scale + 20, goal_position[1] * self.scale + 20),
                                goal_radius * self.scale)
            goal_point.draw(self.win1)
            goal_point.setFill('green')

            # Drawing the dense goals:
            for idx, mini_goal in enumerate(self.dense_goals):
                mini_goal_point = Circle(Point(mini_goal[0] * self.scale + 20, mini_goal[1] * self.scale + 20),
                                         goal_radius * self.scale)
                mini_goal_point.draw(self.win1)
                mini_goal_point.setFill('blue')

            # self.win1.getMouse()

        self.seed()
        self.reset()

    def reset(self):
        self.state = copy.deepcopy(self.start_position)
        self.t = 0
        self.done = False

        return self._get_obs()

    def _get_obs(self):
        return copy.deepcopy(self.state)

    def step(self, a):
        """
        take the action here
        """

        # check if the action is valid
        assert self.action_space.contains(a)
        assert self.done is False

        # Susan added this line
        self.state_temp = copy.deepcopy(self.state)

        self.t += 1

        # check if collides, if it doesn't then update the state
        if self.num_rooms == 0 or not self.collides((self.state[0] + a[0], self.state[1] + a[1])):
            # move the agent and update the state
            self.state[0] += a[0]
            self.state[1] += a[1]

        # clip the state if out of bounds
        self.state[0] = np.clip(self.state[0], self.min_position, self.max_position)
        self.state[1] = np.clip(self.state[1], self.min_position, self.max_position)

        # the reward logic
        reward = self.per_step_penalty

        # Adding dense Rewards:
        for idx, mini_goal in enumerate(self.dense_goals):
            if np.linalg.norm(np.array(self.state) - np.array(mini_goal), 2) <= self.goal_radius:
                reward = self.dense_reward

        # if reached goal (within a radius of 1 unit)
        if np.linalg.norm(np.array(self.state) - np.array(self.goal_position), 2) <= self.goal_radius:
            # episode done
            self.done = True
            reward = self.goal_reward

        if self.t >= self.max_episode_len:
            self.done = True

        line = Line(Point(self.state_temp[0] * self.scale + 20, self.state_temp[1] * self.scale + 20),
                    Point(self.state[0] * self.scale + 20, self.state[1] * self.scale + 20))

        if not self.silent_mode:
            line.draw(self.win1)
            line.setOutline('black')
            # self.win1.getMouse()
        self.state_temp = self.state

        if self.silent_mode:
            return self._get_obs(), reward, self.done, None

        # return self.win1,self._get_obs(), reward, self.done, None
        return self._get_obs(), reward, self.done, None

    def sample(self, exploration, b_0, l_p, ou_noise, stddev):
        """ take a random sample """
        if exploration == 'RandomWalk':
            return np.random.uniform(low=self.min_action[0], high=self.max_action[0], size=(2,))
        elif exploration == 'PolyRL':
            return PolyNoise(L_p=float(l_p), b_0=float(b_0), action_dim=self.nb_actions, ou_noise=ou_noise,
                             sigma=float(stddev))
        else:
            raise Exception("The exploration method " + self.exploration + " is not defined!")

    def create_walls(self):
        """
        create the walls here, the polygons
        """
        self.walls = []

        # codes for drawing the polygons in matplotlib
        codes = [path.Path.MOVETO,
                 path.Path.LINETO,
                 path.Path.LINETO,
                 path.Path.LINETO,
                 path.Path.CLOSEPOLY,
                 ]

        if self.num_rooms == 0:
            # no walls required
            return
        elif self.num_rooms == 1:
            # create one room with one opening

            # a wall parallel to x-axis, at (0,grid_len/2), (grid_len/2,grid_len/2)
            self.walls.append(path.Path([(0, self.grid_len / 2.0 + self.wall_breadth),
                                         (0, self.grid_len / 2.0 - self.wall_breadth),
                                         (self.grid_len / 2.0, self.grid_len / 2.0 - self.wall_breadth),
                                         (self.grid_len / 2.0, self.grid_len / 2.0 + self.wall_breadth),
                                         (0, self.grid_len / 2.0 + self.wall_breadth)
                                         ], codes=codes))

            # the top part  of wall on (0,grid_len/2), parallel to y -axis containg
            self.walls.append(path.Path([(self.grid_len / 2.0 - self.wall_breadth, self.grid_len / 2.0),
                                         (self.grid_len / 2.0 - self.wall_breadth,
                                          self.grid_len / 4.0 + self.door_breadth),
                                         (self.grid_len / 2.0 + self.wall_breadth,
                                          self.grid_len / 4.0 + self.door_breadth),
                                         (self.grid_len / 2.0 + self.wall_breadth, self.grid_len / 2.0),
                                         (self.grid_len / 2.0 - self.wall_breadth, self.grid_len / 2.0),
                                         ], codes=codes))

            # the bottom part  of wall on (0,grid_len/2), parallel to y -axis containg
            self.walls.append(
                path.Path([(self.grid_len / 2.0 - self.wall_breadth, self.grid_len / 4.0 - self.door_breadth),
                           (self.grid_len / 2.0 - self.wall_breadth, 0.),
                           (self.grid_len / 2.0 + self.wall_breadth, 0.),
                           (self.grid_len / 2.0 + self.wall_breadth, self.grid_len / 4.0 - self.door_breadth),
                           (self.grid_len / 2.0 - self.wall_breadth, self.grid_len / 4.0 - self.door_breadth),
                           ], codes=codes))

        elif self.num_rooms == 4:
            # create 4 rooms
            raise Exception("Not implemented yet :(")
        else:
            raise Exception("Logic for current number of rooms " +
                            str(self.num_rooms) + " is not implemented yet :(")

    def collides(self, pt):
        """
        to check if the point (x,y) is in the area defined by the walls polygon (i.e. collides)
        """
        wall_edge_low = self.grid_len / 2 - self.wall_breadth
        wall_edge_high = self.grid_len / 2 + self.wall_breadth
        for w in self.walls:
            if w.contains_point(pt):
                return True
            elif pt[0] <= self.min_position and pt[1] > wall_edge_low and pt[1] < wall_edge_high:
                return True
            elif pt[1] <= self.min_position and pt[0] > wall_edge_low and pt[0] < wall_edge_high:
                return True
        return False

    def vis_trajectory(self, traj, name_plot, experiment_id=None, imp_states=None):
        """
        creates the trajectory and return the plot

        trj: numpy_array


        Code taken from: https://discuss.pytorch.org/t/example-code-to-put-matplotlib-graph-to-tensorboard-x/15806
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        # convert the environment to the image
        ax.set_xlim(0.0, self.max_position)
        ax.set_ylim(0.0, self.max_position)

        # add the border here
        # for i in ax.spines.itervalues():
        #     i.set_linewidth(0.1)

        # plot any walls if any
        for w in self.walls:
            patch = patches.PathPatch(w, facecolor='gray', lw=2)
            ax.add_patch(patch)

        # plot the start and goal points
        ax.scatter([self.start_position[0]], [self.start_position[1]], c='g')
        ax.scatter([self.goal_position[0]], [self.goal_position[1]], c='y')

        # Plot the dense rewards:
        for idx, mini_goal in enumerate(self.dense_goals):
            ax.scatter([mini_goal[0]], [mini_goal[1]], c='b')

        # add the trajectory here
        # https://stackoverflow.com/questions/36607742/drawing-phase-space-trajectories-with-arrows-in-matplotlib

        ax.quiver(traj[:-1, 0], traj[:-1, 1],
                  traj[1:, 0] - traj[:-1, 0], traj[1:, 1] - traj[:-1, 1],
                  scale_units='xy', angles='xy', scale=1, color='black')

        # plot the decision points/states
        if imp_states is not None:
            ax.scatter(imp_states[:, 0], imp_states[:, 1], c='r')

        # return the image buff

        ax.set_title("grid")
        # fig.savefig(buf, format='jpeg') # maybe png
        fig.savefig('install/{}_{}'.format(name_plot, experiment_id), dpi=300)  # maybe png

    def test_vis_trajectory(self, traj, name_plot, experiment_id=None):

        # Trajectory heatmap
        x = np.array([point[0] for point in traj])
        y = np.array([point[1] for point in traj])

        # Save heatmap for different bin scales
        for num in range(1, 5):
            fig, ax = plt.subplots()

            bin_scale = num * 0.1
            h = ax.hist2d(x, y, bins=[np.arange(self.min_position, self.max_position, bin_scale),
                                      np.arange(self.min_position, self.max_position, bin_scale)],
                          cmap='magma')
            image = h[3]
            plt.colorbar(image, ax=ax)

            # Build graph barriers and start and goal positions
            start_circle = plt.Circle((self.start_position[0] * self.scale + 20, self.start_position[1] * self.scale + 20),
                           self.goal_radius * self.scale)

            ax.set_title('Continuous GridWorld Trajectories')
            plt.savefig('install/{}_{}_{}.pdf'.format(name_plot, experiment_id, num))

