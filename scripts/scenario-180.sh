#!/bin/bash
set -o pipefail

cd $(dirname $0)

datestr=$(date +"%Y%m%d-%H%M")
checkpoint_dir=$(realpath "../checkpoints/s180")"/${datestr}"
num_iters=101
num_episodes_per_iter=50

let epsilon_final=$num_iters*$num_episodes_per_iter
let num_iters_half=$num_iters/2
let num_iters_quart=$num_iters/4
let num_iters_threequart=3*$num_iters/4
mkdir -p $checkpoint_dir

for i in $(seq $num_iters); do
    arguments=()
    arguments+=( "--algorithm" "ddqn" )
    arguments+=( "--model" "Building_5ZoneAirCooled_SingleSetpoint" )
    #if (( $i < $num_iters_threequart )); then
    #    arguments+=( "--shared_network_per_agent_class" )
    #fi
    arguments+=( "--single_setpoint_agent_count" "one_but3not5" )
    arguments+=( "--fewer_q_values" )
    arguments+=( "--ts_per_hour" 1 )
    arguments+=( "--ts_until_regulation" 0 )
    if   (( $i < $num_iters_quart )); then
        arguments+=( "--lr" 0.05 )
    	arguments+=( "--batch_size" 512 )
    elif (( $i < $num_iters_half )); then
        arguments+=( "--lr" 0.035 )
    	arguments+=( "--batch_size" 512 )
    elif (( $i < $num_iters_threequart )); then
        arguments+=( "--lr" 0.02 )
    	arguments+=( "--batch_size" 512 )
    else
        arguments+=( "--lr" 0.03 )
    	arguments+=( "--batch_size" 1024 )
    fi
    arguments+=( "--discount_factor" 0.9 )
    arguments+=( "--next_occ_horizont" 2 )
    arguments+=( "--episodes_count" $num_episodes_per_iter )
    arguments+=( "--stp_reward_function" "linear" )
    arguments+=( "--stp_reward_step_offset" 1.0 )
    arguments+=( "--log_rwd_energy" )
    arguments+=( "--energy_cons_in_kWh" )
    arguments+=( "--lambda_rwd_energy" 0.1 )
    arguments+=( "--lambda_rwd_mstpc"  0.2 )
    arguments+=( "--network_storage_frequency" $num_episodes_per_iter )
    arguments+=( "--target_network_update_freq" 3 )
    arguments+=( "--epsilon" 0.05 )
    arguments+=( "--epsilon_final_step" $epsilon_final )
    arguments+=( "--agent_network" "2HiddenLayer,Trapezium,SiLU" )
    arguments+=( "--agent_init_fn" "xavier_normal" )
    arguments+=( "--agent_init_gain" 0.7 )
    arguments+=( "--agent_w_l2" 0.000001 )
    arguments+=( "--checkpoint_dir" $checkpoint_dir )
    arguments+=( "--idf_file" $(realpath 5ZoneAirCooled_HigherWinterSetpoint.idf) )
    arguments+=( "--epw_file" "../../COBS/cobs/data/weathers/8.epw" )
    arguments+=( "--episode_start_month" 1 )
    arguments+=( "--episode_length" 182 )
    arguments+=( "--continue_training" )
    if (( $i >= $num_iters - 1 )); then arguments+=( "--output_Q_vals_iep" ); fi

    python ../code/TrainingController.py "${arguments[@]}" | sed -n -e '/^Ep.*/p' -e '/^Lo.*/p'
    if ! (( $? == 0 )); then
	echo "Error during scenario run."
	exit 1
    fi
done

