#!/bin/bash

#
# Baseline 001
#
# Fairbanks
# January
#

cd $(dirname $0)

checkpoint_dir=$(realpath "../checkpoints/b002")

mkdir -p $checkpoint_dir

python ../code/TrainingController.py \
	--algorithm "rule-based" \
	--lambda_rwd_energy 0.001 \
	--lambda_rwd_mstpc 1.0 \
	--checkpoint_dir $checkpoint_dir \
	--epw_file ../../COBS/cobs/data/weathers/8.epw \
	--episode_start_month 1 \
	--rulebased_setpoint_unoccu_mean 23.0 \
	--rulebased_setpoint_unoccu_delta 7.0 \
	--rulebased_setpoint_occu_mean 23.5 \
	--rulebased_setpoint_occu_delta 1.0 


