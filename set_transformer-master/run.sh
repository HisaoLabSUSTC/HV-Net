#!/bin/sh

export GPU_ID=$1

echo GPU_ID:$GPU_ID

export CUDA_VISIBLE_DEVICES=$GPU_ID
# Activate the relevant virtual environment:
python HVnet.py $2 $3