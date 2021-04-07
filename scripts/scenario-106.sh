#!/bin/bash

cd $(dirname $0)

datestr=$(date +"%Y%m%d-%H%M")
checkpoint_dir=$(realpath "../checkpoints/s106")"/$datestr"

mkdir -p $checkpoint_dir

python ../code/TrainingController.py \
	--algorithm ddqn \
	--model Building_5ZoneAirCooled_SingleSetpoint \
	--lr 0.005 \
	--discount_factor 0.97 \
	--use_cuda \
	--episodes_count 320 \
	--alternate_reward \
	--network_storage_frequency 40 \
	--epsilon 0.05 \
	--agent_w_l2 0.000001 \
	--checkpoint_dir $checkpoint_dir \
	--epw_file ../../COBS/cobs/data/weathers/8.epw \
	--episode_start_month 1


