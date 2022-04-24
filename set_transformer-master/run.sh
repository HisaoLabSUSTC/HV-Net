#!/bin/sh

export GPU_ID=$1

echo GPU_ID:$GPU_ID

export CUDA_VISIBLE_DEVICES=$GPU_ID
# Activate the relevant virtual environment:
python HVnet.py $2 $3

# bash run.sh 0 train_data_M3_1K_5.mat model_M3_1K_5.pth
# bash run.sh 1 train_data_M3_10K_5.mat model_M3_10K_5.pth
# bash run.sh 2 train_data_M3_100K_5.mat model_M3_100K_5.pth

# bash run.sh 3 train_data_M10_1K_5.mat model_M10_1K_5.pth
# bash run.sh 4 train_data_M10_10K_5.mat model_M10_10K_5.pth
# bash run.sh 5 train_data_M10_100K_5.mat model_M10_100K_5.pth

# bash run.sh 0 train_data_M5_1K_5.mat model_M5_1K_5.pth
# bash run.sh 1 train_data_M5_10K_5.mat model_M5_10K_5.pth
# bash run.sh 2 train_data_M5_100K_5.mat model_M5_100K_5.pth

# bash run.sh 3 train_data_M8_1K_5.mat model_M8_1K_5.pth
# bash run.sh 4 train_data_M8_10K_5.mat model_M8_10K_5.pth
# bash run.sh 5 train_data_M8_100K_5.mat model_M8_100K_5.pth
