#!/usr/bin/env bash

set -x

#echo "Getting into the script"

# Script to run multiple batches of experiments together.
# Can also be used for different hyperparameter settings.

# Run the following scripts in parallel:

#COMET_DISABLE_AUTO_LOGGING=1 python main.py --namestr=G-S-CG --make_cont_grid --batch_size=128 --silent --replay_size=100000 --hidden_size=64 --num_steps=100000 --policy=Gaussian --smol --comet &
#COMET_DISABLE_AUTO_LOGGING=1 python main.py --namestr=G-S-CG --make_cont_grid --batch_size=128 --silent --replay_size=100000 --hidden_size=64 --num_steps=100000 --policy=Gaussian --smol --comet &
#COMET_DISABLE_AUTO_LOGGING=1 python main.py --namestr=G-S-CG --make_cont_grid --batch_size=128 --silent --replay_size=100000 --hidden_size=64 --num_steps=100000 --policy=Gaussian --smol --comet &

COMET_DISABLE_AUTO_LOGGING=1 python main.py --namestr=G-NT-S-CG --make_cont_grid --batch_size=128 --silent --replay_size=100000 --hidden_size=64 --num_steps=100000 --policy=Gaussian --smol --tanh=False --comet &
COMET_DISABLE_AUTO_LOGGING=1 python main.py --namestr=G-NT-S-CG --make_cont_grid --batch_size=128 --silent --replay_size=100000 --hidden_size=64 --num_steps=100000 --policy=Gaussian --smol --tanh=False --comet &
COMET_DISABLE_AUTO_LOGGING=1 python main.py --namestr=G-NT-S-CG --make_cont_grid --batch_size=128 --silent --replay_size=100000 --hidden_size=64 --num_steps=100000 --policy=Gaussian --smol --tanh=False --comet &