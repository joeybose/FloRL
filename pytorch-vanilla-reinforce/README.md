This implements basic reinforce with and without a baseline value network for continuous control using Gaussian policies.

An example of how to run reinforce:

python main_reinforce.py --namestr="name of experiment" --env-name <Name_of_{gym/mujoco}_env> --baseline {True/False} --num-episodes 4000

