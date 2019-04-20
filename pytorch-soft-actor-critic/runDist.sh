#!/usr/bin/env bash

#set -x

#echo "Getting into the script"

# Script to run multiple batches of experiments together.
# Can also be used for different hyperparameter settings.

# Run the following scripts in parallel:

COMET_DISABLE_AUTO_LOGGING=1 python main.py --namestr=Reinforce-GaussianSAC-Pendulum-v0 --env-name Pendulum-v0 --batch_size=1024 --policy=Gaussian --comet --reparam=False --seed=1 &
COMET_DISABLE_AUTO_LOGGING=1 python main.py --namestr=Reinforce-GaussianSAC-Pendulum-v0 --env-name Pendulum-v0 --batch_size=1024 --policy=LogNormal --comet --reparam=False --seed=2 &
COMET_DISABLE_AUTO_LOGGING=1 python main.py --namestr=Reinforce-GaussianSAC-Pendulum-v0 --env-name Pendulum-v0 --batch_size=1024 --policy=Laplace --comet --reparam=False --seed=3 &

