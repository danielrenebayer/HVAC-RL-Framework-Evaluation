#!/bin/bash

cd $(dirname $0)

datestr=$(date +"%Y%m%d-%H%M")
checkpoint_dir=$(realpath "../checkpoints/s104")"/$datestr"

mkdir -p $checkpoint_dir

python ../code/TrainingController.py \
	--algorithm ddqn \
	--model 5ZoneAirCooled_SmallAgents \
	--lr 0.01 \
	--discount_factor 0.97 \
	--use_cuda \
	--episodes_count 280 \
	--lambda_rwd_energy 0.0001 \
	--lambda_rwd_mstpc 1.0 \
	--network_storage_frequency 40 \
	--epsilon 0.05 \
	--agent_w_l2 0.000001 \
	--checkpoint_dir $checkpoint_dir \
	--epw_file ../../COBS/cobs/data/weathers/8.epw \
	--episode_start_month 1

