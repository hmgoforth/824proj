import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('./openpose/build/python')
from openpose import pyopenpose as op

sys.path.append('./posewarp/code')
import param
import data_generation
import posewarp_wrapper

from pdb import set_trace as st

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="posewarp/data/test/frames/golf/1.png")
    parser.add_argument("--source", default="posewarp/data/train/frames/tennis/1.jpg")
    parser.add_argument("--gpu", default="0")
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

def joints_from_image(image):
    # in: image
    # out: 14x2 MPII-style joints
    # prepare openpose
    
    op_params = dict()
    op_params["model_folder"] = "./openpose/models/"
    opWrapper = op.WrapperPython()
    opWrapper.configure(op_params)
    opWrapper.start()
    datum = op.Datum()

    datum.cvInputData = image
    opWrapper.emplaceAndPop([datum])
    kp = datum.poseKeypoints[0] # get keypoints of first (and only, hopefully) detected person

    # posewarp format:
    # head (0), neck (1), r-shoulder (2), r-elbow (3), r-wrist (4), l-shoulder (5),
    # l-elbow (6), l-wrist (7), r-hip (8), r-knee (9), r-ankle (10), l-hip (11), 
    # l-knee (12), l-ankle (13)

    joints = np.zeros((14,2))

    # openpose format includes MidHip, but posewarp format does not
    joints[0:8,0:2] = kp[0:8,0:2]
    joints[8:14,0:2] = kp[9:15,0:2]

    # plt.imshow(image)
    # plt.scatter(joints[:,0],joints[:,1])
    # plt.show()

    return joints

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
    pw = posewarp_wrapper.wrapper(model_path=args.model,gpu_id=args.gpu)

    source_image = cv2.imread(args.source)
    target_image = cv2.imread(args.target)

    st()

    source_image = cv2.resize(source_image, (0,0), fx=640/source_image.shape[1], fy=640/source_image.shape[1])
    target_image = cv2.resize(target_image, (0,0), fx=640/target_image.shape[1], fy=640/target_image.shape[1])

    st()
    source_joints = joints_from_image(source_image)
    st()
    target_joints = joints_from_image(target_image)
    st()

    output, x, y = run_posewarp(pw, source_image, target_image, source_joints, target_joints)

    disp_data(x[0])
    plt.title('source')
    disp_data(y)
    plt.title('target gt')

    disp_data(output)
    plt.title('target gan')

    plt.show()

    # read both images
    # read joint information from both using openpose
        # bounding box around person and joint positions

    # create x=list[image_src, pose_src, pose_tgt, mask_src, trans]

    # are these created in posewarp code, from image and joint locations, using existing functions?
        # pose_src: batch x 128 x 128 x 14 (gaussian bumps [0,1])
        # pose_tgt: batch x 128 x 128 x 14 (gaussian bumps [0,1])
        # mask_src: batch x 256 x 256 x 11 (?)
        # trans: batch x 2 x 3 x 11 (affine joint transformations, src to tgt)

    # instantiate posewarp wrapper
    # out = pw(x)
    # print output

if __name__=="__main__":
    main()