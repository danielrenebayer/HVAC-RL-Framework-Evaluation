#!/bin/bash
set -o pipefail

cd $(dirname $0)

datestr=$(date +"%Y%m%d-%H%M")
checkpoint_dir=$(realpath "../checkpoints/s131")"/${datestr}"

mkdir -p $checkpoint_dir

for i in $(seq 310); do
    python ../code/TrainingController.py \
	--algorithm ddqn \
	--model "Building_5ZoneAirCooled_SingleSetpoint" \
	--shared_network_per_agent_class \
	--ts_per_hour 1 \
	--lr 0.015 \
	--discount_factor 0.91 \
	--batch_size 256 \
	--use_cuda \
	--episodes_count 88 \
	--lambda_rwd_energy 0.0001 \
	--lambda_rwd_mstpc 1.0 \
	--network_storage_frequency 88 \
	--target_network_update_freq 6 \
	--epsilon 0.05 \
	--epsilon_final_step 27000 \
	--agent_w_l2 0.000005 \
	--checkpoint_dir $checkpoint_dir \
	--idf_file $(realpath 5ZoneAirCooled_HigherWinterSetpoint.idf) \
	--epw_file ../../COBS/cobs/data/weathers/1A.epw \
	--episode_start_month 1 \
	--continue_training | sed -n -e '/^Ep.*/p' -e '/^Lo.*/p'
    if ! (( $? == 0 )); then
	echo "Error during scenario run."
	exit 1
    fi
done

