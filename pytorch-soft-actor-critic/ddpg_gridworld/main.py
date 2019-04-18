import argparse
from itertools import chain
import datetime
import os
import json
from comet_ml import Experiment
import time
import shutil
from utils import log
from tensorboardX import SummaryWriter
import torch
from continous_grids import GridWorld
import gym.spaces
from PIL import Image
import numpy as np
from exploration_models import *
from action_sampling import *
from agent import Agent
from utils import Logger
from utils import create_folder


parser = argparse.ArgumentParser(description='DDPG')

# Environment settings
parser.add_argument('--env-name', default='GridWorld',help='environment to train on (default: HalfCheetah-v1), "rllab:DoublePendulumEnvX"')
parser.add_argument('--num-steps', type=int, default=1000000,help='total number of steps during training (default: 1M)')
parser.add_argument('--gamma', type=float, default=0.99,help='discount factor for reward (default: 0.99)')

parser.add_argument('--updates-per-step', type=int, default=1,help='model updates per simulator step (default: 1)')
parser.add_argument('--batch-size', type=int, default=128,help='batch size (default: 128)')
parser.add_argument('--tau', type=float, default=0.001,help='discount factor for model (default: 0.001)')
parser.add_argument('--replay-size', type=int, default=100000,help='size of replay buffer (default: 100000)')
parser.add_argument("--silent",action="store_true",default=False,help='Display graphical output. Set to true when running on a server.')


# Algorithm hyper-parameters
# parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 0.0001)')
parser.add_argument('--noise-type', type=str, default='ou_1',help="format: ou_<stdvar>/normal_<stddev>/none")
parser.add_argument('--co-var', type=float, default=0.3,help='Diagonal covariance for the multi variate normal policy.')


# run speific args
parser.add_argument('--log-interval', type=int, default=10,help='interval between training status logs (default: 10)')
parser.add_argument('--checkpoint-interval', type=int, default=100000,help='interval between saving the model (default: 100k)')
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--out', type=str, default='./results/models')
parser.add_argument('--log-dir', type=str, default='./results/logs',help='The logging directory to record the logs and tensorboard summaries')
parser.add_argument('--reset-dir', action='store_true',help="give this argument to delete the existing logs for the current set of parameters")
parser.add_argument('--eval_every', type=int, default=1000,help='Evaluate the policy every N steps (test agent)')
parser.add_argument('--eval-n', type=int, default=20,help='Number of times to eval')
parser.add_argument("--comet",action="store_true",default=False,help='Use comet for logging')
parser.add_argument('--namestr',type=str,default='Gridworld_DDPG',help='additional info in output filename to describe experiments')

### for different gridworld environments
parser.add_argument("--smol",action="store_true",default=False,help='Change to a smaller sized gridworld')
parser.add_argument("--tiny",action="store_true",default=False,help='Change to the smallest sized gridworld')
parser.add_argument("--twotiny",action="store_true",default=False,help='Change to 2x the smallest sized gridworld')
parser.add_argument("--threetiny",action="store_true",default=False,help='Change to 3x the smallest sized gridworld')

### logger for saving results
parser.add_argument("--folder", type=str, default='./results/')
parser.add_argument("--use_logger", action="store_true", default=False, help='whether to use logging or not')
parser.add_argument("--policy_name", default="DDPG", help = "DDPG | LearnMuDDPG")
parser.add_argument('--latent_size', type=int, default=64,help='Latent space dimension')

args = parser.parse_args()

# Check if settings file
if os.path.isfile("settings.json"):
    with open('settings.json') as f:
        data = json.load(f)
    args.comet_apikey = data["apikey"]
    args.comet_username = data["username"]
    args.comet_project = data["project"]

if args.comet:
    experiment = Experiment(api_key=args.comet_apikey,project_name=args.comet_project,auto_output_logging="None",workspace=args.comet_username,auto_metric_logging=False,auto_param_logging=False)
    experiment.set_name(args.namestr)
    args.experiment = experiment


if not args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

# check the device here
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = np.random.randint(1,1000)
args.seed = seed

log(args)

# set np and cuda seeds here
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.gpu:
    torch.cuda.manual_seed(args.seed)

args_param = vars(args)
toprint = ['noise_type']


name = ''
for arg in toprint:
    name += '_{}_{}'.format(arg, args_param[arg])

out_dir = os.path.join(args.out, args.env_name, name)
args.out = out_dir

# create the directory here
os.makedirs(args.out, exist_ok=True)

# create the tensorboard summary writer here
tb_log_dir = os.path.join(args.log_dir, args.env_name, name, 'tb_logs')

print("Log dir", tb_log_dir)
print("Out dir", args.out)

if args.reset_dir:
    shutil.rmtree(tb_log_dir, ignore_errors=True)
os.makedirs(tb_log_dir, exist_ok=True)
tb_writer = SummaryWriter(log_dir=tb_log_dir)


if args.smol:
    env = GridWorld(max_episode_len = 500,num_rooms=1,action_limit_max = 1.0, silent_mode = args.silent, \
                    start_position = (8.0, 8.0),goal_position = (22.0, 22.0),goal_reward = +100.0, \
                    dense_goals = [(13.0,8.0),(18.0,11.0),(20.0,15.0),(22.0, 19.0),], dense_reward = +5,\
                    grid_len = 30)
    env_name = "SmallGridWorld"
elif args.tiny:
    env = GridWorld(max_episode_len = 500,num_rooms=0,action_limit_max = 1.0, silent_mode = args.silent, \
                    start_position = (5.0, 5.0),goal_position = (15.0, 15.0),goal_reward = +100.0, \
                    dense_goals = [], dense_reward = +0,\
                    grid_len = 20)
    env_name = "TinyGridWorld"
elif args.twotiny:
    env = GridWorld(max_episode_len = 500,num_rooms=1,action_limit_max = 1.0, silent_mode = args.silent, \
                    start_position = (5.0, 5.0),goal_position = (15.0, 15.0),goal_reward = +100.0, \
                    dense_goals = [], dense_reward = +0,\
                    grid_len = 20, door_breadth = 3)
    env_name = "TwoTinyGridWorld"
elif args.threetiny:
    env = GridWorld(max_episode_len = 500,num_rooms=0,action_limit_max = 1.0, silent_mode = args.silent, \
                    start_position = (8.0, 8.0),goal_position = (22.0, 22.0),goal_reward = +100.0, \
                    dense_goals = [], dense_reward = +0,\
                    grid_len = 30)
    env_name = "ThreeGridWorld"
else:
    env = GridWorld(max_episode_len = 1000, num_rooms=1,action_limit_max = 1.0, silent_mode = args.silent)
    env_name = "VeryLargeGridWorld"

args.max_path_len=1e6

# dump all the arguments in the tb_log_dir
print(args, file=open(os.path.join(tb_log_dir, "arguments"), "w"))


if args.use_logger:
  file_name = "%s_%s_%s" % (args.policy_name, env_name, str(args.seed))

  logger = Logger(experiment_name = args.policy_name, environment_name = env_name, seed=str(seed), folder = args.folder)
  # logger.save_args(args)

  print ('Saving to', logger.save_folder)




nb_actions = env.action_space.shape[-1]
state_dim = env.observation_space.shape[0]

current_noise_type = args.noise_type.strip()

action_noise = None
if current_noise_type == 'none':
    pass
elif 'RandomWalk' in current_noise_type:
    action_noise = RandomWalkNoise(action_dim = nb_actions, max_action_limit = env.max_action[0])
elif 'adaptive-param' in current_noise_type:
    pass
    # _, stddev = current_noise_type.split('_')
elif 'normal' in current_noise_type:
    # param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
    _, stddev = current_noise_type.split('_')
    action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
elif 'ou' in current_noise_type:
    _, stddev = current_noise_type.split('_')
    action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
else:
    raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))


# create the agent here now
agent = Agent(args=args,
              env=env,
              exploration=action_noise,
              latent_size = args.latent_size,
              logger=logger,
              writer_dir=tb_log_dir)

# run the agent
agent.run()
