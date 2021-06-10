#!/bin/bash

#
# Baseline 004
#
# Fairbanks
# January
#

cd $(dirname $0)

checkpoint_dir=$(realpath "../checkpoints/b004")

if [ -d $checkpoint_dir ]; then
	rm -r $checkpoint_dir
fi

mkdir -p $checkpoint_dir

python ../code/TrainingController.py \
	--algorithm "baseline_rule-based" \
	--ts_per_hour 1 \
	--ts_until_regulation 0 \
	--lambda_rwd_energy 0.000005 \
	--lambda_rwd_mstpc  0.1 \
	--stp_reward_step_offset 1.0 \
	--checkpoint_dir $checkpoint_dir \
	--idf_file $(realpath 5ZoneAirCooled_HigherWinterSetpoint.idf) \
	--epw_file ../../COBS/cobs/data/weathers/8.epw \
	--episode_start_month 1 \
	--rulebased_setpoint_unoccu_mean 23.0 \
	--rulebased_setpoint_unoccu_delta 1.0 \
	--rulebased_setpoint_occu_mean 24.0 \
	--rulebased_setpoint_occu_delta 1.0 


