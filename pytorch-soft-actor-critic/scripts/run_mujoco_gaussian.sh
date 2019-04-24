#!/usr/bin/env bash

set -x

#echo "Getting into the script"

# Script to run multiple batches of experiments together.
# Can also be used for different hyperparameter settings.

# Run the following scripts in parallel:

#COMET_DISABLE_AUTO_LOGGING=1 python main.py --namestr=G-HC --env-name HalfCheetah-v1 --batch_size=1024 --policy=Gaussian --seed=0 --comet &
#COMET_DISABLE_AUTO_LOGGING=1 python main.py --namestr=G-HC --env-name HalfCheetah-v1 --batch_size=1024 --policy=Gaussian --seed=0 --comet &

COMET_DISABLE_AUTO_LOGGING=1 python main.py --namestr=G-NT-HC --env-name HalfCheetah-v1 --batch_size=1024 --policy=Gaussian --seed=0 --tanh_off --comet &
COMET_DISABLE_AUTO_LOGGING=1 python main.py --namestr=G-NT-HC --env-name HalfCheetah-v1 --batch_size=1024 --policy=Gaussian --seed=0 --tanh_off --comet &