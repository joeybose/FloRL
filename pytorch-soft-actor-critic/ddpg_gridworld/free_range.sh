#!/usr/bin/env bash

#set -x

#echo "Getting into the script"

# Script to run multiple batches of experiments together.
# Can also be used for different hyperparameter settings.

# Run the following scripts in parallel:
COMET_DISABLE_AUTO_LOGGING=1 python3 main.py --tiny --comet --namestr="ContinuousGrid_ouDDPG_tiny_0" --silent --noise-type ou_1 &
COMET_DISABLE_AUTO_LOGGING=1 python3 main.py --tiny --comet --namestr="ContinuousGrid_ouDDPG_tiny_1" --silent --noise-type ou_1 &
COMET_DISABLE_AUTO_LOGGING=1 python3 main.py --tiny --comet --namestr="ContinuousGrid_ouDDPG_tiny_2" --silent --noise-type ou_1 &
COMET_DISABLE_AUTO_LOGGING=1 python3 main.py --tiny --comet --namestr="ContinuousGrid_ouDDPG_tiny_3" --silent --noise-type ou_1 &
COMET_DISABLE_AUTO_LOGGING=1 python3 main.py --tiny --comet --namestr="ContinuousGrid_ouDDPG_tiny_4" --silent --noise-type ou_1 &
