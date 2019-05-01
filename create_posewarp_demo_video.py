'''

Create video with 3 horizontal panels: source image, target video, generated output

'''

import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
from pdb import set_trace as st
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source-image',
        help='source image path',
        type=str
    )
    parser.add_argument(
        '--target-image-dir',
        help='path to target image directory',
        type=str
    )
    parser.add_argument(
        '--target-ext',
        help='extension of target image',
        type=str
    )
    parser.add_argument(
        '--final-output-dir',
        help='path to final output',
        type=str
    )
    parser.add_argument(
        '--final-output-ext',
        help='extension to final output',
        type=str
    )
    parser.add_argument(
        '--vid-frames-output',
        help='where to write video frames',
        type=str
    )
    return parser.parse_args()

def main(args):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(9,3))

    source_im = cv2.imread(args.source_image)
    source_im = source_im[:,:,::-1]

    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')

    ax[0].imshow(source_im)

    ax[0].set_title('Source')

    ax[1].set_title('Target')
    ax[2].set_title('Output')

    target_images = glob.glob(os.path.join(args.target_image_dir, '*' + args.target_ext))
    target_images.sort()

    output_images = glob.glob(os.path.join(args.final_output_dir, '*' + args.final_output_ext))
    output_images.sort()

    for idx in range(len(target_images)):
        target_im = cv2.imread(target_images[idx])
        target_im = target_im[:,:,::-1]
        output_im = cv2.imread(output_images[idx])
        output_im = output_im[:,:,::-1]

        ax[1].imshow(target_im)
        ax[2].imshow(output_im)

        plt.draw()

        plt.savefig(os.path.join(args.vid_frames_output, '{:08d}.png'.format(idx)), dpi=225)

        print('frame {:d}'.format(idx))
        # plt.waitforbuttonpress()

if __name__ == '__main__':
    args = parse_args()
    main(args)