#!/usr/bin/env bash

set -x

#echo "Getting into the script"

# Script to run multiple batches of experiments together.
# Can also be used for different hyperparameter settings.

# Run the following scripts in parallel:

COMET_DISABLE_AUTO_LOGGING=1 python main.py --namestr=G-S-DG-CG --make_cont_grid --batch_size=128 --replay_size=100000 --hidden_size=64 --num_steps=100000 --policy=Gaussian --smol --comet --dense_goals --silent &
COMET_DISABLE_AUTO_LOGGING=1 python main.py --namestr=G-S-DG-CG --make_cont_grid --batch_size=128 --replay_size=100000 --hidden_size=64 --num_steps=100000 --policy=Gaussian --smol --comet --dense_goals --silent

#COMET_DISABLE_AUTO_LOGGING=1 python main.py --namestr=G-S-CG --make_cont_grid --batch_size=128 --replay_size=100000 --hidden_size=64 --num_steps=200000 --policy=Gaussian --smol --comet --alpha=0.1 --silent &
#COMET_DISABLE_AUTO_LOGGING=1 python main.py --namestr=G-S-CG --make_cont_grid --batch_size=128 --replay_size=100000 --hidden_size=64 --num_steps=200000 --policy=Gaussian --smol --comet --alpha=0.2 --silent &
#COMET_DISABLE_AUTO_LOGGING=1 python main.py --namestr=G-S-CG --make_cont_grid --batch_size=128 --replay_size=100000 --hidden_size=64 --num_steps=200000 --policy=Gaussian --smol --comet --alpha=0.3 --silent &
#COMET_DISABLE_AUTO_LOGGING=1 python main.py --namestr=G-S-CG --make_cont_grid --batch_size=128 --replay_size=100000 --hidden_size=64 --num_steps=200000 --policy=Gaussian --smol --comet --alpha=0.4 --silent &
#COMET_DISABLE_AUTO_LOGGING=1 python main.py --namestr=G-S-CG --make_cont_grid --batch_size=128 --replay_size=100000 --hidden_size=64 --num_steps=200000 --policy=Gaussian --smol --comet --alpha=0.5 --silent