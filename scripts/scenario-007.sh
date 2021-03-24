#!/bin/bash

#
# Scenario 007
#
# Fairbanks
# January
#

cd $(dirname $0)

datestr=$(date +"%Y%m%d-%H%M")
checkpoint_dir=$(realpath "../checkpoints/s007")"/$datestr"

mkdir -p $checkpoint_dir

python ../code/TrainingController.py \
	--model 5ZoneAirCooled \
	--lr 0.0005 \
	--discount_factor 0.93 \
	--use_cuda \
	--episodes_count 100 \
	--lambda_rwd_energy 0.001 \
	--lambda_rwd_mstpc 0.13 \
	--critic_hidden_activation LeakyReLU \
	--critic_hidden_size 30 \
	--network_storage_frequency 20 \
	--ou_theta 0.3 \
	--ou_sigma 0.15 \
	--agent_w_l2 0.000002 \
	--critic_w_l2 0.000002 \
	--checkpoint_dir $checkpoint_dir \
	--epw_file ../../COBS/cobs/data/weathers/8.epw \
	--episode_start_month 1


