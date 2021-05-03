#!/bin/bash

#
# Baseline 012
#
# Chicago O'Hare
# July
#

cd $(dirname $0)

checkpoint_dir=$(realpath "../checkpoints/b012")

mkdir -p $checkpoint_dir

python ../code/TrainingController.py \
	--algorithm "baseline_rule-based" \
	--lambda_rwd_energy 0.001 \
	--lambda_rwd_mstpc 1.0 \
	--checkpoint_dir $checkpoint_dir \
	--idf_file $(realpath 5ZoneAirCooled_HigherWinterSetpoint.idf) \
	--rulebased_setpoint_unoccu_mean 26.0 \
	--rulebased_setpoint_unoccu_delta 4.0 \
	--rulebased_setpoint_occu_mean 17.5 \
	--rulebased_setpoint_occu_delta 1.0 


