#!/bin/bash

#
# Scenario 010
#
# as scenario 009, but:
# with other reward: only mstpc, on energy reward
# higher discount factor, smaller weight regularisation
#
# Fairbanks
# January
#

cd $(dirname $0)

datestr=$(date +"%Y%m%d-%H%M")
checkpoint_dir=$(realpath "../checkpoints/s010")"/$datestr"

mkdir -p $checkpoint_dir

python ../code/TrainingController.py \
	--model 5ZoneAirCooled_SingleAgent \
	--lr 0.0005 \
	--discount_factor 0.97 \
	--use_cuda \
	--episodes_count 80 \
	--lambda_rwd_energy 0.0 \
	--lambda_rwd_mstpc 0.7 \
	--critic_hidden_activation LeakyReLU \
	--critic_hidden_size 30 \
	--network_storage_frequency 20 \
	--ou_theta 0.3 \
	--ou_sigma 0.15 \
	--agent_w_l2 0.000001 \
	--critic_w_l2 0.000001 \
	--checkpoint_dir $checkpoint_dir \
	--epw_file ../../COBS/cobs/data/weathers/8.epw \
	--episode_start_month 1


