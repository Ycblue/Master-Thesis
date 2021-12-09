#!/bin/bash
cd /home/students/lan/ma_yuchia_lan/code/x2ct_architecture
args=()
for i in "$@"; do
	args+=" $i"
done

/work/scratch/lan/conda/envs/pytorch/bin/python /home/students/lan/ma_yuchia_lan/code/x2ct_architecture/gan.py$args;
# /work/scratch/lan/conda/envs/pytorch_thesis/bin/python /home/students/lan/Schreibtisch/Code/x2ct_architecture_ff/train.py$args;


