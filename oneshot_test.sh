#!/bin/sh

GPU=0
SOURCE="posewarp/data/3_small.jpg"
TARGET="posewarp/data/4_cropped.jpg"
CUDA_VISIBLE_DEVICES=$GPU python3 extract_joints.py --target $TARGET --source $SOURCE
CUDA_VISIBLE_DEVICES=$GPU python3 run_posewarp.py --target $TARGET --source $SOURCE