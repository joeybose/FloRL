#!/usr/bin/env bash

set -x

#echo "Getting into the script"

# Script to run multiple batches of experiments together.
# Can also be used for different hyperparameter settings.

# Run the following scripts in parallel:

#COMET_DISABLE_AUTO_LOGGING=1 python main.py --namestr=E-S-DG-CG --make_cont_grid --batch_size=128 --replay_size=100000 --hidden_size=64 --num_steps=100000 --policy=Exponential --smol --comet --dense_goals &
#COMET_DISABLE_AUTO_LOGGING=1 python main.py --namestr=E-S-DG-CG --make_cont_grid --batch_size=128 --replay_size=100000 --hidden_size=64 --num_steps=100000 --policy=Exponential --smol --comet --dense_goals

#COMET_DISABLE_AUTO_LOGGING=1 python main.py --namestr=E-S-CG --make_cont_grid --batch_size=128 --replay_size=100000 --hidden_size=64 --num_steps=150000 --policy=Exponential --smol --comet --silent --alpha=0.1 &
#COMET_DISABLE_AUTO_LOGGING=1 python main.py --namestr=E-S-CG --make_cont_grid --batch_size=128 --replay_size=100000 --hidden_size=64 --num_steps=150000 --policy=Exponential --smol --comet --silent --alpha=0.2
COMET_DISABLE_AUTO_LOGGING=1 python main.py --namestr=E-S-CG --make_cont_grid --batch_size=128 --replay_size=100000 --hidden_size=64 --num_steps=150000 --policy=Exponential --smol --comet --silent --alpha=0.3 &
COMET_DISABLE_AUTO_LOGGING=1 python main.py --namestr=E-S-CG --make_cont_grid --batch_size=128 --replay_size=100000 --hidden_size=64 --num_steps=150000 --policy=Exponential --smol --comet --silent --alpha=0.4 &
COMET_DISABLE_AUTO_LOGGING=1 python main.py --namestr=E-S-CG --make_cont_grid --batch_size=128 --replay_size=100000 --hidden_size=64 --num_steps=150000 --policy=Exponential --smol --comet --silent --alpha=0.5 &