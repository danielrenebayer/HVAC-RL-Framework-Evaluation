#!/bin/bash
set -o pipefail

cd $(dirname $0)

datestr=$(date +"%Y%m%d-%H%M")
checkpoint_dir=$(realpath "../checkpoints/s201")"/${datestr}"
num_iters=100
num_episodes_per_iter=160

let epsilon_final=$num_iters*$num_episodes_per_iter
let num_iters_half=$num_iters/2
let num_iters_quart=$num_iters/4
let num_iters_threequart=3*$num_iters/4
mkdir -p $checkpoint_dir

for i in $(seq $num_iters); do
    arguments=()
    arguments+=( "--algorithm" "ddpg" )
    arguments+=( "--model" "Building_5ZoneAirCooled_SingleSetpoint" )
    arguments+=( "--ts_per_hour" 1 )
    arguments+=( "--ts_until_regulation" 0 )
    arguments+=( "--single_setpoint_agent_count" "one" )
    arguments+=( "--use_layer_normalization" )
    arguments+=( "--use_cuda" )
    if   (( $i < $num_iters_quart )); then
        arguments+=( "--lr" 0.040 )
    elif (( $i < $num_iters_half )); then
        arguments+=( "--lr" 0.025 )
    elif (( $i < $num_iters_threequart )); then
        arguments+=( "--lr" 0.013 )
    else
        arguments+=( "--lr" 0.008 )
    fi
    arguments+=( "--discount_factor" 0.85 )
    arguments+=( "--next_occ_horizont" 2 )
    arguments+=( "--batch_size" 256 )
    arguments+=( "--ou_theta" 0.1 )
    arguments+=( "--ou_sigma" 0.3 )
    arguments+=( "--episodes_count" $num_episodes_per_iter )
    arguments+=( "--stp_reward_function" "linear" )
    arguments+=( "--stp_reward_step_offset" 1.0 )
    arguments+=( "--lambda_rwd_energy" 0.000005 )
    arguments+=( "--lambda_rwd_mstpc"  0.1 )
    arguments+=( "--network_storage_frequency" $num_episodes_per_iter )
    arguments+=( "--target_network_update_freq" 6 )
    arguments+=( "--epsilon" 0.05 )
    arguments+=( "--epsilon_final_step" $epsilon_final )
    arguments+=( "--critic_network" "2HiddenLayer,FastPyramid" )
    arguments+=( "--agent_network" "2HiddenLayer,Trapezium" )
    arguments+=( "--agent_init_fn" "xavier_normal" )
    arguments+=( "--agent_init_gain" 0.7 )
    arguments+=( "--agent_w_l2" 0.000001 )
    arguments+=( "--critic_w_l2" 0.000002 )
    arguments+=( "--checkpoint_dir" $checkpoint_dir )
    arguments+=( "--idf_file" $(realpath 5ZoneAirCooled_HigherWinterSetpoint.idf) )
    arguments+=( "--epw_file" "../../COBS/cobs/data/weathers/8.epw" )
    arguments+=( "--episode_start_month" 1 )
    arguments+=( "--continue_training" )

    #python -m ipdb ../code/TrainingController.py "${arguments[@]}"
    python ../code/TrainingController.py "${arguments[@]}" | sed -n -e '/^Ep.*/p' -e '/^Lo.*/p'
    if ! (( $? == 0 )); then
	echo "Error during scenario run."
	exit 1
    fi
done

