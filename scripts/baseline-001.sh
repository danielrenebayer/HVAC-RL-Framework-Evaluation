#!/bin/bash

#
# Baseline 001
#
# Fairbanks
# January
#

cd $(dirname $0)

checkpoint_dir=$(realpath "../checkpoints/b001")

mkdir -p $checkpoint_dir

python ../code/TrainingController.py \
	--algorithm "baseline_rule-based" \
	--lambda_rwd_energy 0.001 \
	--lambda_rwd_mstpc 1.0 \
	--checkpoint_dir $checkpoint_dir \
	--idf_file $(realpath 5ZoneAirCooled_HigherWinterSetpoint.idf) \
	--epw_file ../../COBS/cobs/data/weathers/8.epw \
	--episode_start_month 1


