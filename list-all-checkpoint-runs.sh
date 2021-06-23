#!/bin/bash

for fl in $(ls -1 checkpoints); do
	for dt in $(ls checkpoints/$fl); do
		if [ ${dt:0:1} == "2" ]; then
			echo $dt $fl/$dt
		fi
	done
done | sort

