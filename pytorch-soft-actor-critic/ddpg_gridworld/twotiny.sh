#!/usr/bin/env bash

#set -x

#echo "Getting into the script"

# Script to run multiple batches of experiments together.
# Can also be used for different hyperparameter settings.

# Run the following scripts in parallel:
# COMET_DISABLE_AUTO_LOGGING=1 Python3 Main.Py --twotiny --comet --namestr="ContinuousGrid_ouDDPG_doubletiny_0" --silent --noise-type ou_1 &
# COMET_DISABLE_AUTO_LOGGING=1 python3 main.py --twotiny --comet --namestr="ContinuousGrid_ouDDPG_doubletiny_1" --silent --noise-type ou_1 &
# COMET_DISABLE_AUTO_LOGGING=1 python3 main.py --twotiny --comet --namestr="ContinuousGrid_ouDDPG_doubletiny_2" --silent --noise-type ou_1 &
# COMET_DISABLE_AUTO_LOGGING=1 python3 main.py --twotiny --comet --namestr="ContinuousGrid_ouDDPG_doubletiny_3" --silent --noise-type ou_1 &
# COMET_DISABLE_AUTO_LOGGING=1 python3 main.py --twotiny --comet --namestr="ContinuousGrid_ouDDPG_doubletiny_4" --silent --noise-type ou_1 &


COMET_DISABLE_AUTO_LOGGING=1 python main.py --noise-type ou_1 --eval_every 1000 --twotiny --use_logger --policy_name "LearnMuDDPG" --comet --namestr="ContinuousGrid_MuDDPG_twotiny_default_0" --silent &
COMET_DISABLE_AUTO_LOGGING=1 python main.py --noise-type ou_1 --eval_every 1000 --twotiny --use_logger --policy_name "LearnMuDDPG" --comet --namestr="ContinuousGrid_MuDDPG_twotiny_default_1" --silent &
COMET_DISABLE_AUTO_LOGGING=1 python main.py --noise-type ou_1 --eval_every 1000 --twotiny --use_logger --policy_name "LearnMuDDPG" --comet --namestr="ContinuousGrid_MuDDPG_twotiny_default_2" --silent &

COMET_DISABLE_AUTO_LOGGING=1 python main.py --noise-type ou_1 --eval_every 1000 --twotiny --use_logger --policy_name "DDPG" --comet --namestr="ContinuousGrid_DDPG_twotiny_default_0" --silent &
COMET_DISABLE_AUTO_LOGGING=1 python main.py --noise-type ou_1 --eval_every 1000 --twotiny --use_logger --policy_name "DDPG" --comet --namestr="ContinuousGrid_DDPG_twotiny_default_1" --silent &
COMET_DISABLE_AUTO_LOGGING=1 python main.py --noise-type ou_1 --eval_every 1000 --twotiny --use_logger --policy_name "DDPG" --comet --namestr="ContinuousGrid_DDPG_twotiny_default_2" --silent &
