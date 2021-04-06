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
	--algorithm "rule-based" \
	--lambda_rwd_energy 0.001 \
	--lambda_rwd_mstpc 1.0 \
	--checkpoint_dir $checkpoint_dir


