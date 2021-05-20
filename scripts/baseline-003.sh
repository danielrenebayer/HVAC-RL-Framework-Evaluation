#!/bin/bash

#
# Baseline 003
#
# Fairbanks
# January
#
# Modified building
#

cd $(dirname $0)

checkpoint_dir=$(realpath "../checkpoints/b003")

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
	--checkpoint_dir $checkpoint_dir \
	--idf_file $(realpath 5ZoneAirCooled_LowerSummerHigherWinterSetpoint.idf) \
	--epw_file ../../COBS/cobs/data/weathers/8.epw \
	--episode_start_month 1


