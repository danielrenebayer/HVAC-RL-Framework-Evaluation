#!/bin/bash

#
# Scenario 002
#
# like scenario 1, but with ou_theta = 0.3, ou_sigma = 0.05
# longer training
#
# Chicago O'Hare
# July
#

cd $(dirname $0)

datestr=$(date +"%Y%m%d-%H%M")
checkpoint_dir=$(realpath "../checkpoints/s002")"/$datestr"

mkdir -p $checkpoint_dir

python ../code/TrainingController.py \
	--model 5ZoneAirCooled_SingleAgent \
	--lr 0.0005 \
	--use_cuda \
	--episodes_count 160 \
	--lambda_rwd_energy 0.001 \
	--lambda_rwd_mstpc 0.13 \
	--critic_hidden_activation LeakyReLU \
	--critic_hidden_size 30 \
	--network_storage_frequency 20 \
	--ou_theta 0.3 \
	--ou_sigma 0.05 \
	--checkpoint_dir $checkpoint_dir


