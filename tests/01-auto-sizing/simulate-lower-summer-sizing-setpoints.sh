#!/bin/bash

cd $(dirname $0)

checkpoint_dir=$(realpath checkpoints_lower_summer_stp)

mkdir -p $checkpoint_dir

python ../../code/TrainingController.py \
	--algorithm "rule-based" \
	--lambda_rwd_energy 0.001 \
	--lambda_rwd_mstpc 1.0 \
	--checkpoint_dir $checkpoint_dir \
	--idf_file $(realpath 5ZoneAirCooled_LowerSummerSetpoint.idf) \
	--epw_file ../../../COBS/cobs/data/weathers/1A.epw \
	--episode_start_month 7 \
	--rulebased_setpoint_unoccu_mean 15.0 \
	--rulebased_setpoint_unoccu_delta 7.0 \
	--rulebased_setpoint_occu_mean 13.0 \
	--rulebased_setpoint_occu_delta 1.0 


