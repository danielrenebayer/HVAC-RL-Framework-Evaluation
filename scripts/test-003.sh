#!/bin/bash

#
# Scenario 003
#
# multible agents
#
# Chicago O'Hare
# July
#

cd $(dirname $0)

checkpoint_dir=$(realpath "../checkpoints/t003")

filestorage="../checkpoints/t003/"
episode=119

mkdir -p $checkpoint_dir

python ../code/TrainingController.py \
	--model 5ZoneAirCooled \
	--lr 0.0005 \
	--use_cuda \
	--lambda_rwd_energy 0.001 \
	--lambda_rwd_mstpc 0.13 \
	--critic_hidden_activation LeakyReLU \
	--critic_hidden_size 30 \
	--episodes_count 1 \
	--network_storage_frequency 1 \
	--ou_theta 0.3 \
	--ou_sigma 0.15 \
	--checkpoint_dir $checkpoint_dir \
	--add_ou_in_eval_epoch \
	--load_models_from_path $filestorage \
	--load_models_episode $episode


