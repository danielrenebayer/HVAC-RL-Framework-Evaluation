#!/bin/bash
set -o pipefail

cd $(dirname $0)

datestr=$(date +"%Y%m%d-%H%M")
checkpoint_dir=$(realpath "../checkpoints/s143")"/${datestr}"

mkdir -p $checkpoint_dir

for i in $(seq 42); do
    python ../code/TrainingController.py \
	--algorithm ddqn \
	--model "Building_5ZoneAirCooled_SingleSetpoint" \
	--shared_network_per_agent_class \
	--ts_per_hour 1 \
	--lr 0.025 \
	--discount_factor 0.75 \
	--batch_size 256 \
	--episodes_count 88 \
	--reward_offset 0.1 \
	--stp_reward_function "linear" \
	--stp_reward_step_offset 1.0 \
	--lambda_rwd_energy 0.00001 \
	--lambda_rwd_mstpc  0.2 \
	--network_storage_frequency 88 \
	--target_network_update_freq 4 \
	--epsilon 0.05 \
	--epsilon_final_step 6330 \
	--agent_network "2HiddenLayer,Trapezium,SiLU" \
	--agent_init_fn "xavier_normal" \
	--agent_init_gain 0.6 \
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
for i in $(seq 30); do
    python ../code/TrainingController.py \
	--algorithm ddqn \
	--model "Building_5ZoneAirCooled_SingleSetpoint" \
	--ts_per_hour 1 \
	--lr 0.015 \
	--discount_factor 0.75 \
	--batch_size 256 \
	--episodes_count 88 \
	--reward_offset 0.1 \
	--stp_reward_function "linear" \
	--stp_reward_step_offset 1.0 \
	--lambda_rwd_energy 0.00001 \
	--lambda_rwd_mstpc  0.2 \
	--network_storage_frequency 88 \
	--target_network_update_freq 4 \
	--epsilon 0.05 \
	--epsilon_final_step 6330 \
	--agent_network "2HiddenLayer,Trapezium,SiLU" \
	--agent_init_fn "xavier_normal" \
	--agent_init_gain 0.6 \
	--agent_w_l2 0.0000005 \
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
	--model "Building_5ZoneAirCooled_SingleSetpoint" \
	--ts_per_hour 1 \
	--lr 0.005 \
	--discount_factor 0.75 \
	--batch_size 256 \
	--episodes_count 88 \
	--reward_offset 0.1 \
	--stp_reward_function "linear" \
	--stp_reward_step_offset 1.0 \
	--lambda_rwd_energy 0.00001 \
	--lambda_rwd_mstpc  0.2 \
	--network_storage_frequency 88 \
	--target_network_update_freq 4 \
	--epsilon 0.05 \
	--epsilon_final_step 6330 \
	--agent_network "2HiddenLayer,Trapezium,SiLU" \
	--agent_init_fn "xavier_normal" \
	--agent_init_gain 0.6 \
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

