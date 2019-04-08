#!/usr/bin/env bash

#set -x

#echo "Getting into the script"

# Run the following scripts in parallel:

# HyperParam Sweep
COMET_DISABLE_AUTO_LOGGING=1 python main.py --namestr="Planar 3 Flow Lunar Lander" --env-name LunarLanderContinuous-v2 --batch_size=1024 --policy='Flow' --flow_model='planar' --n_blocks=3 --comet --seed=1 &
COMET_DISABLE_AUTO_LOGGING=1 python main.py --namestr="Planar 5 Flow Lunar Lander" --env-name LunarLanderContinuous-v2 --batch_size=1024 --policy='Flow' --flow_model='planar' --n_blocks=5 --comet --seed=2 &
COMET_DISABLE_AUTO_LOGGING=1 python main.py --namestr="Planar 7 Flow Lunar Lander" --env-name LunarLanderContinuous-v2 --batch_size=1024 --policy='Flow' --flow_model='planar' --n_blocks=7 --comet --seed=3 &
COMET_DISABLE_AUTO_LOGGING=1 python main.py --namestr="Planar 9 Flow Lunar Lander" --env-name LunarLanderContinuous-v2 --batch_size=1024 --policy='Flow' --flow_model='planar' --n_blocks=9 --comet --seed=4&
