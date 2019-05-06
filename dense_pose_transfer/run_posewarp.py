import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

import sys

sys.path.append('./posewarp/code')
import param
import data_generation
import posewarp_wrapper

from pdb import set_trace as st

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="posewarp/data/test/frames/golf/1.png")
    parser.add_argument("--source", default="posewarp/data/train/frames/tennis/1.jpg")
    parser.add_argument("--target-joints", default="joints/target_joints.npy")
    parser.add_argument("--source-joints", default="joints/source_joints.npy")
    parser.add_argument("--model", default="posewarp/models/vgg+gan_5000.h5")
    return parser.parse_args()

def disp_data(d,gray=False):

    plt.figure()

    if gray:
        plt.imshow(d[0,:,:,0:1],cmap='gray')
    else:
        # reverse channels
        d = d[0,:,:,::-1]
        d = d - np.amin(d,axis=(0,1))
        d = d / np.amax(d,axis=(0,1))
        plt.imshow(d)

def run_posewarp(pw, source_image, target_image, source_joints, target_joints):
    # in: source image, source joints, target joints
    # out: source in target pose
    params = param.get_general_params()
    x, y = data_generation.format_network_input(params, source_image, target_image, source_joints, target_joints)
    out = pw.gen(x)
    return out, x, y

def main():
    print('pid: ' + str(os.getpid()))
    args = parse_args()
    pw = posewarp_wrapper.wrapper(model_path=args.model)

    source_image = cv2.imread(args.source)
    target_image = cv2.imread(args.target)
    source_joints = np.load(args.source_joints)
    target_joints = np.load(args.target_joints)

    output, x, y = run_posewarp(pw, source_image, target_image, source_joints, target_joints)

    disp_data(x[0])
    plt.title('source')
    disp_data(np.expand_dims(y,0))
    plt.title('target gt')

    disp_data(output)
    plt.title('target gan')

    plt.show()

if __name__=="__main__":
    main()