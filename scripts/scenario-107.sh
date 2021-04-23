#!/bin/bash

cd $(dirname $0)

datestr=$(date +"%Y%m%d-%H%M")
checkpoint_dir=$(realpath "../checkpoints/s107")"/${datestr}"

mkdir -p $checkpoint_dir

for i in $(seq 4); do
    python ../code/TrainingController.py \
	--algorithm ddqn \
	--model Building_5ZoneAirCooled_SingleSetpoint_SmallAgents \
	--lr 0.001 \
	--discount_factor 0.91 \
	--batch_size 512 \
	--use_cuda \
	--episodes_count 80 \
	--reward_function "rulebased_agent_output" \
	--network_storage_frequency 40 \
	--epsilon 0.05 \
	--agent_w_l2 0.0005 \
	--checkpoint_dir $checkpoint_dir \
	--epw_file ../../COBS/cobs/data/weathers/8.epw \
	--episode_start_month 1 \
	--continue_training
    if ! (( $? == 0 )); then
	echo "Error during scenario run."
	exit 1
    fi
done

