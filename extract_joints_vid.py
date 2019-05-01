import os
import cv2
import argparse
import numpy as np
import glob
from pdb import set_trace as st

import sys
sys.path.append('./openpose/build/python')

from openpose import pyopenpose as op

op_params = dict()
op_params["model_folder"] = "./openpose/models/"
opWrapper = op.WrapperPython()
opWrapper.configure(op_params)
opWrapper.start()
datum = op.Datum()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", help="directory of video frames to transfer",type=str)
    parser.add_argument("--target-ext", help="extension of target images",type=str)
    parser.add_argument("--source", help="source image",type=str)
    parser.add_argument("--outdir", help="where to save source joints, target joints",type=str)
    return parser.parse_args()

def joints_from_image(image):
    # in: image
    # out: 14x2 MPII-style joints
    # prepare openpose

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

def main():
    print('pid: ' + str(os.getpid()))
    args = parse_args()

    source_image = cv2.imread(args.source)
    source_joints = joints_from_image(source_image)
    np.save(os.path.join(args.outdir, 'source_joints.npy'),source_joints)

    target_images = glob.glob(os.path.join(args.target, '*' + args.target_ext))
    target_images.sort()

    target_outdir = os.path.join(args.outdir, 'targets')

    if not os.path.exists(target_outdir):
        os.mkdir(target_outdir)


    frame_count = 0
    for image in target_images:
        target_im = cv2.imread(image)
        target_joints = joints_from_image(target_im)
        np.save(os.path.join(target_outdir, 'target_joints_{:08d}.npy'.format(frame_count)),target_joints)

        frame_count += 1

        print('frame ' + image)

if __name__=="__main__":
    main()