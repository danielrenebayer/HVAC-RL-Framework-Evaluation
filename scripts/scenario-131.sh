#!/bin/bash
set -o pipefail

cd $(dirname $0)

datestr=$(date +"%Y%m%d-%H%M")
checkpoint_dir=$(realpath "../checkpoints/s131")"/${datestr}"

mkdir -p $checkpoint_dir

for i in $(seq 250); do
    python ../code/TrainingController.py \
	--algorithm ddqn \
	--model "Building_5ZoneAirCooled_SingleSetpoint_SingleSmallAgent" \
	--ts_per_hour 1 \
	--eplus_storage_mode \
	--lr 0.05 \
	--discount_factor 0.7 \
	--batch_size 512 \
	--episodes_count 88 \
	--reward_offset 0.5 \
	--reward_scale 0.25 \
	--stp_reward_function "linear" \
	--stp_reward_step_offset 1.0 \
	--reward_function "rulebased_agent_output" \
	--network_storage_frequency 88 \
	--target_network_update_freq 7 \
	--epsilon 0.05 \
	--epsilon_final_step 22000 \
	--agent_network "1HiddenBigLayer,SiLU" \
	--agent_init_fn "xavier_normal" \
	--agent_init_gain 0.3 \
	--agent_w_l2 0.000001 \
	--checkpoint_dir $checkpoint_dir \
	--idf_file $(realpath 5ZoneAirCooled_HigherWinterSetpoint.idf) \
	--epw_file ../../COBS/cobs/data/weathers/8.epw \
	--episode_start_month 1 \
	--continue_training | sed -n -e '/^Ep.*/p' -e '/^Lo.*/p'
    if ! (( $? == 0 )); then
	echo "Error during scenario run."
	exit 1
    fi
done
for i in $(seq 1); do
    python ../code/TrainingController.py \
	--algorithm ddqn \
	--model "Building_5ZoneAirCooled_SingleSetpoint_SingleSmallAgent" \
	--ts_per_hour 1 \
	--eplus_storage_mode \
	--lr 0.005 \
	--discount_factor 0.7 \
	--batch_size 512 \
	--episodes_count 88 \
	--reward_offset 0.5 \
	--reward_scale 0.25 \
	--stp_reward_function "linear" \
	--stp_reward_step_offset 1.0 \
	--reward_function "rulebased_agent_output" \
	--network_storage_frequency 88 \
	--target_network_update_freq 7 \
	--epsilon 0.05 \
	--epsilon_final_step 30000 \
	--agent_network "1HiddenBigLayer,SiLU" \
	--agent_w_l2 0.0000005 \
	--checkpoint_dir $checkpoint_dir \
	--idf_file $(realpath 5ZoneAirCooled_HigherWinterSetpoint.idf) \
	--epw_file ../../COBS/cobs/data/weathers/8.epw \
	--episode_start_month 1 \
	--output_Q_vals_iep \
	--continue_training | sed -n -e '/^Ep.*/p' -e '/^Lo.*/p'
    if ! (( $? == 0 )); then
	echo "Error during scenario run."
	exit 1
    fi
done

