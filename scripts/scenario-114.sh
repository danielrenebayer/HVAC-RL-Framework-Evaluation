#!/bin/bash
set -o pipefail

cd $(dirname $0)

datestr=$(date +"%Y%m%d-%H%M")
checkpoint_dir=$(realpath "../checkpoints/s114")"/${datestr}"

mkdir -p $checkpoint_dir

for i in $(seq 300); do
    python ../code/TrainingController.py \
	--algorithm ddqn \
	--model "Building_5ZoneAirCooled_SingleSetpoint" \
	--shared_network_per_agent_class \
	--ts_per_hour 1 \
	--lr 0.001 \
	--discount_factor 0.91 \
	--batch_size 256 \
	--use_cuda \
	--episodes_count 80 \
	--reward_function "rulebased_roomtemp" \
	--network_storage_frequency 80 \
	--epsilon 0.05 \
	--epsilon_final_step 23000 \
	--agent_w_l2 0.000005 \
	--checkpoint_dir $checkpoint_dir \
	--epw_file ../../COBS/cobs/data/weathers/8.epw \
	--episode_start_month 1 \
	--continue_training | sed -n -e '/^Ep.*/p' -e '/^Lo.*/p'
    if ! (( $? == 0 )); then
	echo "Error during scenario run."
	exit 1
    fi
done

