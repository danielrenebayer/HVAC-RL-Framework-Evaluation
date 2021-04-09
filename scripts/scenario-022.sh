#!/bin/bash

#
# Scenario 020
#
# Fairbanks
# January
#
# Smaller critic hidden size
#


cd $(dirname $0)

datestr=$(date +"%Y%m%d-%H%M")
checkpoint_dir=$(realpath "../checkpoints/s022")"/$datestr"

mkdir -p $checkpoint_dir

python ../code/TrainingController.py \
	--model 5ZoneAirCooled_SmallAgents \
	--lr 0.0005 \
	--discount_factor 0.97 \
	--use_cuda \
	--episodes_count 80 \
	--lambda_rwd_energy 0.0 \
	--lambda_rwd_mstpc 0.7 \
	--critic_hidden_activation LeakyReLU \
	--critic_hidden_size 5 \
	--network_storage_frequency 20 \
	--ou_theta 0.05 \
	--ou_sigma 0.15 \
	--agent_w_l2 0.000001 \
	--critic_w_l2 0.000001 \
	--checkpoint_dir $checkpoint_dir \
	--epw_file ../../COBS/cobs/data/weathers/8.epw \
	--episode_start_month 1
	#--load_models_from_path $(realpath "../checkpoints/s020/pretrained/") \
	#--load_models_episode 10


