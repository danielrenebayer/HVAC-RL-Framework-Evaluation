#!/bin/bash
set -o pipefail

cd $(dirname $0)

datestr=$(ls -1 "../checkpoints/s151" | sort --reverse | head -n 1)
checkpoint_dir=$(realpath "../checkpoints/s151")"/${datestr}"

for i in $(seq 1); do
    arguments=()
    arguments+=( "--algorithm" "ddqn" )
    arguments+=( "--model" "Building_5ZoneAirCooled_SingleSetpoint" )
    arguments+=( "--shared_network_per_agent_class" )
    arguments+=( "--ts_per_hour" 1 )
    arguments+=( "--ts_until_regulation" 0 )
    arguments+=( "--lr" 0.011 )
    arguments+=( "--reward_function" "rulebased_agent_output" )
    arguments+=( "--reward_scale" 0.3 ) # to fit in the range [-1,0]
    arguments+=( "--discount_factor" 0.85 )
    arguments+=( "--next_occ_horizont" 2 )
    arguments+=( "--batch_size" 256 )
    arguments+=( "--episodes_count" 1 )
    arguments+=( "--reward_offset" 0.0 )
    arguments+=( "--stp_reward_function" "linear" )
    arguments+=( "--stp_reward_step_offset" 1.0 )
    arguments+=( "--lambda_rwd_energy" 0.0000025 )
    arguments+=( "--lambda_rwd_mstpc"  0.15 )
    arguments+=( "--network_storage_frequency" 20 )
    arguments+=( "--target_network_update_freq" 6 )
    arguments+=( "--epsilon" 0.05 )
    arguments+=( "--epsilon_final_step" 1 )
    arguments+=( "--agent_network" "2HiddenLayer,Trapezium,SiLU" )
    arguments+=( "--agent_init_fn" "xavier_normal" )
    arguments+=( "--agent_init_gain" 0.7 )
    arguments+=( "--agent_w_l2" 0.000001 )
    arguments+=( "--checkpoint_dir" $checkpoint_dir )
    arguments+=( "--idf_file" $(realpath 5ZoneAirCooled_HigherWinterSetpoint.idf) )
    arguments+=( "--epw_file" "../../COBS/cobs/data/weathers/8.epw" )
    arguments+=( "--episode_start_month" 1 )
    arguments+=( "--continue_training" )
    #if (( $i >= $num_iters - 1 )); then arguments+=( "--output_Q_vals_iep" ); fi

    python ../code/TrainingAdjustRewardDistribution.py "${arguments[@]}"
    if ! (( $? == 0 )); then
	echo "Error during scenario run."
	exit 1
    fi
done

