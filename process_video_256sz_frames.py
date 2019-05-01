'''

Give a video and output directory. Will process video, center crop a square from each frames and resize to 256, and save

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

def parse_args():
    parser = argparse.ArgumentParser(description='preprocess IUV for deepfashion')
    parser.add_argument(
        '--video',
        help='video to process',
        type=str
    )
    parser.add_argument(
        '--outdir',
        help='location to save processed output',
        type=str
    )
    return parser.parse_args()

def main(args):
    vidcap = cv2.VideoCapture(args.video)

    success, image = vidcap.read()
    framecount = 0

    while success:
        # center crop square and resize to 256x256
        min_dim = min(image.shape[0],image.shape[1])
        rescale_factor = 256.0 / min_dim

        im_resize = cv2.resize(image, (0,0), fx=rescale_factor, fy=rescale_factor)

        crop_col_start = (im_resize.shape[1] // 2 - 1) - 127
        crop_col_end = crop_col_start + 256
        crop_row_start = (im_resize.shape[0] // 2 - 1) - 127
        crop_row_end = crop_row_start + 256

        im_crop = im_resize[crop_row_start:crop_row_end, crop_col_start:crop_col_end, :]

        assert(im_crop.shape[0] == 256, 'im_crop height not 256')
        assert(im_crop.shape[1] == 256, 'im_crop width not 256')

        image_name = '{:08d}.png'.format(framecount)
        image_path = os.path.join(args.outdir, image_name)
        cv2.imwrite(image_path, im_crop)

        framecount += 1

        success, image = vidcap.read()

        print("frame {:d}".format(framecount))

if __name__ == '__main__':
    args = parse_args()
    main(args)