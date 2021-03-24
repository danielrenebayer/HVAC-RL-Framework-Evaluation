#!/bin/bash

#
# Scenario 004
#
# like scenario 1, but with ou_theta = 0.3, ou_sigma = 0.05
# train january until june
#
# Chicago O'Hare
# July
#

cd $(dirname $0)

datestr=$(date +"%Y%m%d-%H%M")
checkpoint_dir=$(realpath "../checkpoints/s004")"/$datestr"

mkdir -p $checkpoint_dir

python ../code/TrainingController.py \
	--model 5ZoneAirCooled_SingleAgent \
	--lr 0.0005 \
	--use_cuda \
	--lambda_rwd_energy 0.001 \
	--lambda_rwd_mstpc 0.13 \
	--critic_hidden_activation LeakyReLU \
	--critic_hidden_size 30 \
	--episodes_count 160 \
	--network_storage_frequency 20 \
	--episode_start_month 1 \
	--episode_length 170 \
	--ou_theta 0.3 \
	--ou_sigma 0.05 \
	--checkpoint_dir $checkpoint_dir


