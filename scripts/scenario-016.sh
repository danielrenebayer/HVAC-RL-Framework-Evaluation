#!/bin/bash

#
# Scenario 016
#
# Fairbanks
# January
#
# Smaller critic hidden size
#

cd $(dirname $0)

datestr=$(date +"%Y%m%d-%H%M")
checkpoint_dir=$(realpath "../checkpoints/s016")"/$datestr"

mkdir -p $checkpoint_dir

python ../code/TrainingController.py \
	--model 5ZoneAirCooled_SmallSingleAgent \
	--lr 0.001 \
	--discount_factor 0.97 \
	--use_cuda \
	--episodes_count 400 \
	--lambda_rwd_energy 0.001 \
	--lambda_rwd_mstpc 0.6 \
	--critic_hidden_activation LeakyReLU \
	--critic_hidden_size 8 \
	--network_storage_frequency 40 \
	--ou_theta 0.05 \
	--ou_sigma 0.20 \
	--agent_w_l2 0.000002 \
	--critic_w_l2 0.000002 \
	--checkpoint_dir $checkpoint_dir \
	--epw_file ../../COBS/cobs/data/weathers/8.epw \
	--episode_start_month 1


