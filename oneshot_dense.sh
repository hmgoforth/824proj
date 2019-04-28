#!/bin/sh

# run while not in a virtual environment

GPU=0
SOURCE=$HOME/824proj/posewarp/data/3_small.jpg # source texture, should be 256x256 with person centered
TARGET=$HOME/824proj/posewarp/data/4_cropped.jpg # target poses, can be single image or directory, should be 256x256 with person centered
IUV_OUTPUT=$HOME/824proj/inpainting/oneshot_data/IUV # where to put DensePose IUV
INPAINTING_NET=$HOME/824proj/inpainting/models/26_4_38000.pth # inpainting network

# get densepose output for both target and source
conda activate caffe2
cd inpainting/densepose
CUDA_VISIBLE_DEVICES=$GPU python3 oneshot_getiuv.py --target $TARGET --source $SOURCE --outdir $IUV_OUTPUT
conda deactivate
cd ..

# transfer pose
source $HOME/posewarp/env/bin/activate
CUDA_VISIBLE_DEVICES=$GPU python3 oneshot_transfer.py --target $TARGET --source $SOURCE --iuv $IUV_OUTPUT --net $INPAINTING_NET
deactivate