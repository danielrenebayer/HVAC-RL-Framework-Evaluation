#!/bin/bash

#
# Scenario 001
#
# Chicago O'Hare
# July
#

cd $(dirname $0)

datestr=$(date +"%Y%m%d-%H%M")
checkpoint_dir=$(realpath "../checkpoints/s001")"/$datestr"

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
	--checkpoint_dir $checkpoint_dir


