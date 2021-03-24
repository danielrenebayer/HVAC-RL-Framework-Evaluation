#!/bin/bash

#
# Baseline 001
#
# like scenario 5
#
# Fairbanks
# January
#

cd $(dirname $0)

checkpoint_dir=$(realpath "../checkpoints/b001")

mkdir -p $checkpoint_dir

python ../code/TrainingController.py \
	--use_rule_based_agent \
	--lambda_rwd_energy 0.001 \
	--lambda_rwd_mstpc 0.13 \
	--checkpoint_dir $checkpoint_dir \
	--epw_file ../../COBS/cobs/data/weathers/8.epw \
	--episode_start_month 1


