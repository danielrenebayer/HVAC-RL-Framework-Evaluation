#!/bin/bash
set -o pipefail

cd $(dirname $0)

datestr=$(date +"%Y%m%d-%H%M")
checkpoint_dir=$(realpath "../checkpoints/s133")"/${datestr}"

mkdir -p $checkpoint_dir

for i in $(seq 150); do
    python ../code/TrainingController.py \
	--algorithm ddqn \
	--model "Building_5ZoneAirCooled_SingleSetpoint_SingleAgent" \
	--ts_per_hour 1 \
	--eplus_storage_mode \
	--lr 0.06 \
	--discount_factor 0.7 \
	--batch_size 256 \
	--episodes_count 88 \
	--reward_offset 0.0 \
	--reward_scale 1.0 \
	--stp_reward_function "linear" \
	--stp_reward_step_offset 1.5 \
	--reward_function "rulebased_agent_output" \
	--network_storage_frequency 88 \
	--target_network_update_freq 6 \
	--epsilon 0.05 \
	--epsilon_final_step 30000 \
	--agent_network "2HiddenLayer,Trapezium" \
	--agent_w_l2 0.0000015 \
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
for i in $(seq 175); do
    python ../code/TrainingController.py \
	--algorithm ddqn \
	--model "Building_5ZoneAirCooled_SingleSetpoint_SingleAgent" \
	--ts_per_hour 1 \
	--eplus_storage_mode \
	--lr 0.01 \
	--discount_factor 0.7 \
	--batch_size 256 \
	--episodes_count 88 \
	--reward_offset 0.0 \
	--reward_scale 1.0 \
	--stp_reward_function "linear" \
	--stp_reward_step_offset 1.5 \
	--reward_function "rulebased_agent_output" \
	--network_storage_frequency 88 \
	--target_network_update_freq 6 \
	--epsilon 0.05 \
	--epsilon_final_step 30000 \
	--agent_network "2HiddenLayer,Trapezium" \
	--agent_w_l2 0.0000015 \
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
	--model "Building_5ZoneAirCooled_SingleSetpoint_SingleAgent" \
	--ts_per_hour 1 \
	--eplus_storage_mode \
	--lr 0.01 \
	--discount_factor 0.7 \
	--batch_size 128 \
	--episodes_count 88 \
	--reward_offset 0.0 \
	--reward_scale 1.0 \
	--stp_reward_function "linear" \
	--stp_reward_step_offset 1.5 \
	--reward_function "rulebased_agent_output" \
	--network_storage_frequency 88 \
	--target_network_update_freq 6 \
	--epsilon 0.05 \
	--epsilon_final_step 30000 \
	--agent_network "2HiddenLayer,Trapezium" \
	--agent_w_l2 0.0000015 \
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

