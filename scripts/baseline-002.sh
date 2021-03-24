#!/bin/bash

#
# Baseline 002
#
# like scenario 5
#
# Chicago O'Hare
# July
#

cd $(dirname $0)

checkpoint_dir=$(realpath "../checkpoints/b002")

mkdir -p $checkpoint_dir

python ../code/TrainingController.py \
	--use_rule_based_agent \
	--lambda_rwd_energy 0.001 \
	--lambda_rwd_mstpc 0.13 \
	--checkpoint_dir $checkpoint_dir


