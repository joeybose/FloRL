#!/usr/bin/env


# Script to run multiple batches of experiments together.
# Can also be used for different hyperparameter settings.

# Run the following scripts in parallel:

#COMET_DISABLE_AUTO_LOGGING=1 python3 launcher.py --policy_name MaxEntDDPG --comet --namestr="InvertedPendulum_MaxEntDDPG_beta=0.8" --use_logger True --beta 0.9 --env_name InvertedPendulum-v1 &

#COMET_DISABLE_AUTO_LOGGING=1 python3 launcher.py --policy_name MaxEntDDPG --comet --namestr="InvertedDoublePendulum_MaxEntDDPG_beta=0.8" --use_logger True --beta 0.9 --env_name InvertedDoublePendulum-v1 &

#COMET_DISABLE_AUTO_LOGGING=1 python3 launcher.py --policy_name DDPG --comet --namestr="Ant_DDPG_beta=0.9" --use_logger True --beta 0.9 --env_name Ant-v1 &
#COMET_DISABLE_AUTO_LOGGING=1 python3 launcher.py --policy_name MaxEntDDPG --comet --namestr="Ant_MaxEntDDPG_beta=0.9" --use_logger True --beta 0.9 --env_name Ant-v1 &
#COMET_DISABLE_AUTO_LOGGING=1 python3 launcher.py --policy_name FlowDDPG --comet --namestr="Ant_FlowDDPG_beta=0.9" --use_logger True --beta 0.9 --env_name Ant-v1 &
COMET_DISABLE_AUTO_LOGGING=1 python3 main_learn_behaviour.py --policy_name DDPG --comet --namestr="Ant_DDPG_beta=0.9" --beta 0.9 --env_name Ant-v1 &
COMET_DISABLE_AUTO_LOGGING=1 python3 main_learn_behaviour.py --policy_name MaxEntDDPG --comet --namestr="Ant_MaxEntDDPG_beta=0.9" --beta 0.9 --env_name Ant-v1 &
COMET_DISABLE_AUTO_LOGGING=1 python3 main_learn_behaviour.py --policy_name FlowDDPG --comet --namestr="Ant_FlowDDPG_beta=0.9" --beta 0.9 --env_name Ant-v1 &


# Cycling through hyperparameters, HalfCheetah:

# COMET_DISABLE_AUTO_LOGGING=1 python3 launcher.py --policy_name MaxEntDDPG --comet --namestr="HalfCheetah_MaxEntDDPG_beta=0.9" --use_logger True --beta 0.9 --env_name HalfCheetah-v1 &

# COMET_DISABLE_AUTO_LOGGING=1 python3 launcher.py --policy_name MaxEntDDPG --comet --namestr="HalfCheetah_MaxEntDDPG_beta=0.7" --use_logger True --beta 0.7 --env_name HalfCheetah-v1 &

# COMET_DISABLE_AUTO_LOGGING=1 python3 launcher.py --policy_name MaxEntDDPG --comet --namestr="HalfCheetah_MaxEntDDPG_beta=0.6" --use_logger True --beta 0.6 --env_name HalfCheetah-v1 &

# COMET_DISABLE_AUTO_LOGGING=1 python3 launcher.py --policy_name MaxEntDDPG --comet --namestr="HalfCheetah_MaxEntDDPG_beta=0.5" --use_logger True --beta 0.5 --env_name HalfCheetah-v1 &
