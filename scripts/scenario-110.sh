#!/bin/bash
set -o pipefail

cd $(dirname $0)

datestr=$(date +"%Y%m%d-%H%M")
checkpoint_dir=$(realpath "../checkpoints/s110")"/${datestr}"

mkdir -p $checkpoint_dir

for i in $(seq 190); do
    python ../code/TrainingController.py \
	--algorithm ddqn \
	--model "Building_5ZoneAirCooled_SingleSetpoint" \
	--ts_per_hour 1 \
	--eplus_storage_mode \
	--lr 0.001 \
	--discount_factor 0.91 \
	--batch_size 512 \
	--use_cuda \
	--episodes_count 88 \
	--reward_function "rulebased_agent_output" \
	--network_storage_frequency 88 \
	--epsilon 0.05 \
	--epsilon_final_step 11000 \
	--agent_w_l2 0.000003 \
	--checkpoint_dir $checkpoint_dir \
	--epw_file ../../COBS/cobs/data/weathers/8.epw \
	--episode_start_month 1 \
	--continue_training
    if ! (( $? == 0 )); then
	echo "Error during scenario run."
	exit 1
    fi
done

