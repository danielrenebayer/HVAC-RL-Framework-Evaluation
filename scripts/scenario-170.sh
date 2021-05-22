#!/bin/bash
set -o pipefail

cd $(dirname $0)

datestr=$(date +"%Y%m%d-%H%M")
checkpoint_dir=$(realpath "../checkpoints/s170")"/${datestr}"
num_iters=660
num_episodes_per_iter=160

let epsilon_final=$num_iters*$num_episodes_per_iter
let num_iters_half=$num_iters/2
let num_iters_quart=$num_iters/4
let num_iters_threequart=3*$num_iters/4
mkdir -p $checkpoint_dir

for i in $(seq $num_iters); do
    arguments=()
    arguments+=( "--algorithm" "ddqn" )
    arguments+=( "--model" "Building_5ZoneAirCooled_SingleSetpoint_SingleBIGAgent" )
    arguments+=( "--ts_per_hour" 1 )
    arguments+=( "--ts_until_regulation" 0 )
    arguments+=( "--use_cuda" )
    if   (( $i < $num_iters_quart )); then
        arguments+=( "--lr" 0.08 )
    elif (( $i < $num_iters_half )); then
        arguments+=( "--lr" 0.04 )
    elif (( $i < $num_iters_threequart )); then
        arguments+=( "--lr" 0.02 )
    else
        arguments+=( "--lr" 0.008 )
    fi
    arguments+=( "--discount_factor" 0.7 )
    arguments+=( "--next_occ_horizont" 2 )
    arguments+=( "--batch_size" 256 )
    arguments+=( "--episodes_count" $num_episodes_per_iter )
    arguments+=( "--reward_offset" 0.05 )
    arguments+=( "--stp_reward_function" "linear" )
    arguments+=( "--stp_reward_step_offset" 1.0 )
    arguments+=( "--lambda_rwd_energy" 0.0000025 )
    arguments+=( "--lambda_rwd_mstpc"  0.15 )
    arguments+=( "--network_storage_frequency" $num_episodes_per_iter )
    arguments+=( "--target_network_update_freq" 6 )
    arguments+=( "--epsilon" 0.05 )
    arguments+=( "--epsilon_final_step" $epsilon_final )
    arguments+=( "--agent_network" "2HiddenLayer,Trapezium" )
    arguments+=( "--agent_init_fn" "xavier_normal" )
    arguments+=( "--agent_init_gain" 0.7 )
    arguments+=( "--agent_w_l2" 0.000001 )
    arguments+=( "--checkpoint_dir" $checkpoint_dir )
    arguments+=( "--idf_file" $(realpath 5ZoneAirCooled_HigherWinterSetpoint.idf) )
    arguments+=( "--epw_file" "../../COBS/cobs/data/weathers/8.epw" )
    arguments+=( "--episode_start_month" 1 )
    arguments+=( "--continue_training" )
    if (( $i >= $num_iters - 1 )); then arguments+=( "--output_Q_vals_iep" ); fi

    python ../code/TrainingController.py "${arguments[@]}" | sed -n -e '/^Ep.*/p' -e '/^Lo.*/p'
    if ! (( $? == 0 )); then
	echo "Error during scenario run."
	exit 1
    fi
done

