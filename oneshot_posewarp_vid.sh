#!/bin/sh

GPU=0
SOURCE="posewarp/data/hunter_256.jpeg"
TARGET="posewarp/data/floss_frames"

# uses 2 different scripts since cannot fit OpenPose and PoseWarp network on single GPU
CUDA_VISIBLE_DEVICES=$GPU python3 extract_joints_vid.py --target $TARGET --source $SOURCE
CUDA_VISIBLE_DEVICES=$GPU python3 run_posewarp_vid.py --target $TARGET --source $SOURCE