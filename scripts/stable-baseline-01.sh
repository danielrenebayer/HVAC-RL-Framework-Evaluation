#!/bin/bash
set -o pipefail

cd $(dirname $0)

datestr=$(date +"%Y%m%d-%H%M")
checkpoint_dir=$(realpath "../checkpoints/sb01")"/${datestr}"

mkdir -p $checkpoint_dir

#for i in $(seq 190); do
    python ../code/TrainingController_StableBaseline.py \
	--algorithm ddqn \
	--model Building_5ZoneAirCooled_SingleSetpoint_SingleAgent \
	--ts_per_hour 1 \
	--lr 0.02 \
	--discount_factor 0.91 \
	--batch_size 512 \
	--episodes_count 88 \
	--reward_function "rulebased_agent_output" \
	--network_storage_frequency 88 \
	--epsilon 0.05 \
	--epsilon_final_step 16000 \
	--agent_w_l2 0.000004 \
	--checkpoint_dir $checkpoint_dir \
	--idf_file $(realpath 5ZoneAirCooled_HigherWinterSetpoint.idf) \
	--epw_file ../../COBS/cobs/data/weathers/8.epw \
	--episode_start_month 1
#    if ! (( $? == 0 )); then
#	echo "Error during scenario run."
#	exit 1
#    fi
#done

