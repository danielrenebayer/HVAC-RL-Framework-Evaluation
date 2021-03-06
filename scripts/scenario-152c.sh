#!/bin/bash
set -o pipefail

cd $(dirname $0)

scenario_id=152

datestr=$(ls -1 "../checkpoints/s${scenario_id}" | grep -e "[0-9]\$" | sort --reverse | head -n 1)
checkpoint_dir=$(realpath "../checkpoints/s${scenario_id}")"/${datestr}_ch_rwd_fn"
num_iters=140
num_episodes_per_iter=88

let epsilon_final=$num_iters*$num_episodes_per_iter
let num_iters_half=$num_iters/2
mkdir -p $checkpoint_dir

# move networks to the new dir
reward_scale=$(cat "../checkpoints/s${scenario_id}/$datestr/RewardDistributionEval/best-reward-scale.txt")
last_latest_episode=$(cat "../checkpoints/s${scenario_id}/$datestr/RewardDistributionEval/name-of-used-model.txt")
cp "${last_latest_episode}_model_actor.pickle" "${checkpoint_dir}/episode_0_agent_0_model_actor.pickle"
cp "${last_latest_episode}_model_target.pickle" "${checkpoint_dir}/episode_0_agent_0_model_target.pickle"

for i in $(seq $num_iters); do
    arguments=()
    arguments+=( "--algorithm" "ddqn" )
    arguments+=( "--model" "Building_5ZoneAirCooled_SingleSetpoint" )
    arguments+=( "--shared_network_per_agent_class" )
    arguments+=( "--ts_per_hour" 1 )
    arguments+=( "--ts_until_regulation" 0 )
    if (( $i < $num_iters_half )); then
        arguments+=( "--lr" 0.01 )
    else
        arguments+=( "--lr" 0.005 )
    fi
    # finer training start
    arguments+=( "--reward_function" "sum_energy_mstpc" )
    arguments+=( "--reward_scale" $reward_scale )
    arguments+=( "--load_models_from_path" ${checkpoint_dir} )
    arguments+=( "--load_models_episode" 0 )
    # finer training end
    arguments+=( "--discount_factor" 0.85 )
    arguments+=( "--next_occ_horizont" 2 )
    arguments+=( "--batch_size" 256 )
    arguments+=( "--episodes_count" $num_episodes_per_iter )
    arguments+=( "--reward_offset" 0.0 )
    arguments+=( "--stp_reward_function" "linear" )
    arguments+=( "--stp_reward_step_offset" 1.0 )
    arguments+=( "--lambda_rwd_energy" 0.000005 )
    arguments+=( "--lambda_rwd_mstpc"  0.1 )
    arguments+=( "--network_storage_frequency" $num_episodes_per_iter )
    arguments+=( "--target_network_update_freq" 6 )
    arguments+=( "--epsilon_initial" 0.35 )
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
    arguments+=( "--continue_training" )
    if (( $i >= $num_iters - 1 )); then arguments+=( "--output_Q_vals_iep" ); fi

    python ../code/TrainingController.py "${arguments[@]}" | sed -n -e '/^Ep.*/p' -e '/^Lo.*/p'
    if ! (( $? == 0 )); then
	echo "Error during scenario run."
	exit 1
    fi
done

