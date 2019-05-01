import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys

sys.path.append('./posewarp/code')
import param
import data_generation
import posewarp_wrapper

from pdb import set_trace as st

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", help="target image directory")
    parser.add_argument("--source", help="source image")
    parser.add_argument("--target-ext", help="extension of target images")
    parser.add_argument("--target-joints", help="directory containing target joints")
    parser.add_argument("--source-joints", help="file containing source joints")
    parser.add_argument("--model", default="posewarp/models/vgg+gan_5000.h5")
    parser.add_argument("--outdir", help="where to save output frames")
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

def convert_to_cv2_rgb(d):
    d = d[0,:,:,:]
    d = d - np.amin(d,axis=(0,1))
    d = d / np.amax(d,axis=(0,1))
    d = d * 255
    d = d.astype(int)
    return d
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
    source_joints = np.load(args.source_joints)

    target_images = glob.glob(os.path.join(args.target, '*' + args.target_ext))
    target_images.sort()

    target_joints_files = glob.glob(os.path.join(args.target_joints, '*.npy'))
    target_joints_files.sort()

    for idx in range(len(target_images)):

        target_image = cv2.imread(target_images[idx])
        target_joints = np.load(target_joints_files[idx])

        output, x, y = run_posewarp(pw, source_image, target_image, source_joints, target_joints)

        out_rgb = convert_to_cv2_rgb(output)

        cv2.imwrite(os.path.join(args.outdir, '{:08d}.png'.format(idx)), out_rgb)

        print('frame: {:d}'.format(idx))

if __name__=="__main__":
    main()