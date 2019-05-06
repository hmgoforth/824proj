#!/bin/sh

GPU=0
SOURCE="posewarp/data/3_small.jpg"
TARGET="posewarp/data/4_cropped.jpg"

# uses 2 different scripts since cannot fit OpenPose and PoseWarp network on single GPU
CUDA_VISIBLE_DEVICES=$GPU python3 extract_joints.py --target $TARGET --source $SOURCE
CUDA_VISIBLE_DEVICES=$GPU python3 run_posewarp.py --target $TARGET --source $SOURCE