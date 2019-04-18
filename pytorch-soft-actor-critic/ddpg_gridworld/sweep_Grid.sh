#!/usr/bin/env bash

#set -x

#echo "Getting into the script"

# Script to run multiple batches of experiments together.
# Can also be used for different hyperparameter settings.

# Run the following scripts in parallel:
COMET_DISABLE_AUTO_LOGGING=1 python3 main.py --comet --namestr="ContinuousGrid_DDPG_default_0" --silent &
COMET_DISABLE_AUTO_LOGGING=1 python3 main.py --comet --namestr="ContinuousGrid_DDPG_default_1" --silent &
COMET_DISABLE_AUTO_LOGGING=1 python3 main.py --comet --namestr="ContinuousGrid_DDPG_default_2" --silent &
COMET_DISABLE_AUTO_LOGGING=1 python3 main.py --comet --namestr="ContinuousGrid_DDPG_default_3" --silent &
COMET_DISABLE_AUTO_LOGGING=1 python3 main.py --comet --namestr="ContinuousGrid_DDPG_default_4" --silent &
