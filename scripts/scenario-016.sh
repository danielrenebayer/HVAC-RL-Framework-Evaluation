#!/bin/bash

cd $(dirname $0)

datestr=$(date +"%Y%m%d-%H%M")
checkpoint_dir=$(realpath "../checkpoints/s016")"/$datestr"

mkdir -p $checkpoint_dir

for i in $(seq 35); do
    python ../code_t1/TrainingController.py \
	--model Building_5ZoneAirCooled_SingleSetpoint_SingleSmallAgent \
	--ts_per_hour 1 \
	--lr 0.001 \
	--discount_factor 0.92 \
	--use_cuda \
	--episodes_count 320 \
	--reward_function "rulebased_agent_output" \
	--lambda_rwd_energy 0.0001 \
	--lambda_rwd_mstpc 1.0 \
	--critic_hidden_activation LeakyReLU \
	--critic_hidden_size 8 \
	--network_storage_frequency 40 \
	--ou_theta 0.26 \
	--ou_sigma 0.35 \
	--epsilon_final_step 11000 \
	--agent_w_l2 0.000002 \
	--critic_w_l2 0.000002 \
	--checkpoint_dir $checkpoint_dir \
	--epw_file ../../COBS/cobs/data/weathers/8.epw \
	--episode_start_month 1 \
	--continue_training
    if ! (( $? == 0 )); then
	echo "Error during scenario run."
	exit 1
    fi
done

