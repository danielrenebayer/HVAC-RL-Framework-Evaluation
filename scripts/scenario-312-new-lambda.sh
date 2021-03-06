#!/bin/bash
set -o pipefail

cd $(dirname $0)

datestr=$(date +"%Y%m%d-%H%M")
checkpoint_dir=$(realpath "../checkpoints/s312-newlambda")"/${datestr}"
num_iters=100
num_episodes_per_iter=250

let epsilon_final_1=$num_iters*$num_episodes_per_iter/2
let epsilon_final_2=$num_iters*$num_episodes_per_iter
let num_iters_A=$num_iters/8
let num_iters_B=$num_iters/4
let num_iters_C=3*$num_iters/8
let num_iters_D=$num_iters/2
let num_iters_E=5*$num_iters/8
let num_iters_F=3*$num_iters/4
let num_iters_G=7*$num_iters/8
mkdir -p $checkpoint_dir

for i in $(seq $num_iters); do
    arguments=()
    arguments+=( "--algorithm" "ddqn" )
    arguments+=( "--ddqn_new" )
    if (( $i < $num_iters_D )); then
        arguments+=( "--model" "Building_5ZoneAirCooled_SingleSetpoint_SingleAgent" )
    else
      arguments+=( "--model" "Building_5ZoneAirCooled_SingleSetpoint" )
      arguments+=( "--single_setpoint_agent_count" "all" )
      #if (( $i < $num_iters_G )); then
        arguments+=( "--shared_network_per_agent_class" )
      #fi
    fi
    arguments+=( "--fewer_q_values" )
    arguments+=( "--ts_per_hour" 1 )
    arguments+=( "--ts_until_regulation" 0 )
    if   (( $i < $num_iters_A )); then
        arguments+=( "--lr" 0.1 )
    elif (( $i < $num_iters_B )); then
        arguments+=( "--lr" 0.065 )
    elif (( $i < $num_iters_C )); then
        arguments+=( "--lr" 0.04 )
    elif (( $i < $num_iters_D )); then
        arguments+=( "--lr" 0.02 )
    elif (( $i < $num_iters_E )); then
        arguments+=( "--lr" 0.03 )
    elif (( $i < $num_iters_F )); then
        arguments+=( "--lr" 0.02 )
    elif (( $i < $num_iters_G )); then
        arguments+=( "--lr" 0.012 )
    else
        arguments+=( "--lr" 0.005 )
    fi
    arguments+=( "--discount_factor" 0.9 )
    arguments+=( "--next_occ_horizont" 2 )
    arguments+=( "--batch_size" 256 )
    arguments+=( "--episodes_count" $num_episodes_per_iter )
    arguments+=( "--stp_reward_step_offset" 1.0 )
    arguments+=( "--reward_offset" 0.3 )
    arguments+=( "--lambda_rwd_energy" 0.008 )
    arguments+=( "--lambda_rwd_mstpc"  0.12 )
    arguments+=( "--clip_econs_at" 150.0 )
    arguments+=( "--soften_instead_of_clipping" )
    arguments+=( "--energy_cons_in_kWh" )
    arguments+=( "--network_storage_frequency" $num_episodes_per_iter )
    arguments+=( "--target_network_update_freq" 2 )
    arguments+=( "--epsilon" 0.05 )
    arguments+=( "--epsilon_final_step" $epsilon_final_1 )
    arguments+=( "--epsilon_decay_mode" "linear" )
    arguments+=( "--agent_network" "2HiddenLayer,Trapezium" )
    arguments+=( "--agent_init_fn" "xavier_normal" )
    arguments+=( "--agent_init_gain" 0.7 )
    arguments+=( "--agent_w_l2" 0.000001 )
    arguments+=( "--use_layer_normalization" )
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

