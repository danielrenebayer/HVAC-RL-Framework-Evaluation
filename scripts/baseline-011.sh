#!/bin/bash

#
# Baseline 011
#
# Chicago O'Hare
# July
#

cd $(dirname $0)

checkpoint_dir=$(realpath "../checkpoints/b011")

mkdir -p $checkpoint_dir

python ../code/TrainingController.py \
	--algorithm "baseline_rule-based" \
	--ts_until_regulation 0 \
	--lambda_rwd_energy 0.001 \
	--lambda_rwd_mstpc 1.0 \
	--stp_reward_step_offset 1.0 \
	--idf_file $(realpath 5ZoneAirCooled_HigherWinterSetpoint.idf) \
	--checkpoint_dir $checkpoint_dir


