#!/usr/bin/env bash

#set -x

#echo "Getting into the script"

# Script to run multiple batches of experiments together.
# Can also be used for different hyperparameter settings.

# Run the following scripts in parallel:

COMET_DISABLE_AUTO_LOGGING=1 python main.py --namestr=E-LL --env-name LunarLanderContinuous-v2 --batch_size=1024 --policy=Exponential --comet &
COMET_DISABLE_AUTO_LOGGING=1 python main.py --namestr=E-LN --env-name LunarLanderContinuous-v2 --batch_size=1024 --policy=LogNormal --comet &
COMET_DISABLE_AUTO_LOGGING=1 python main.py --namestr=L-LL --env-name LunarLanderContinuous-v2 --batch_size=1024 --policy=Laplace --comet &

