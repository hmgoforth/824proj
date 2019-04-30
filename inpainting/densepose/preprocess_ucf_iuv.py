'''
Process UCF videos, save 256x256 frames and IUV
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
from pdb import set_trace as st
import numpy as np
from tqdm import tqdm

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

def parse_args():
    parser = argparse.ArgumentParser(description='preprocess IUV for deepfashion')
    parser.add_argument(
        '--ucf',
        help='location of ucf dataset',
        default='/dataset/UCF101/UCF-101',
        type=str
    )
    parser.add_argument(
        '--outdir',
        help='location of to save preprocessed output',
        default='/dataset/UCF101/preprocessed',
        type=str
    )
    parser.add_argument(
        '--to-process',
        help='location of files containing which videos to process',
        default='../../UCF-dataset',
        type=str
    )
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default='configs/DensePose_ResNet101_FPN_s1x-e2e.yaml',
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default='https://dl.fbaipublicfiles.com/densepose/DensePose_ResNet101_FPN_s1x-e2e.pkl',
        type=str
    )
    return parser.parse_args()

def extract_iuv(im, model, infer_engine):
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    with c2_utils.NamedCudaScope(0):
        cls_boxes, cls_segms, cls_keyps, cls_bodys = infer_engine.im_detect_all(
            model, im, None
        )

    iuv_out = vis_utils.vis_one_image(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            None,
            None,
            cls_boxes,
            cls_segms,
            cls_keyps,
            cls_bodys,
            dataset=dummy_coco_dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=0.7,
            kp_thresh=2
        )

    return iuv_out

def text_file_to_list(fn):
    text = open(fn, 'r')
    return [line.strip() for line in text.readlines()]

def main(args):
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    model = infer_engine.initialize_model_from_cfg(args.weights)

    UCF_categs = glob.glob(os.path.join(args.to_process, '*.txt'))

    for categ_file in UCF_categs:
        print('categ_file: ' + categ_file)

        vid_group_list = text_file_to_list(categ_file)
        ucf_categ = os.path.basename(categ_file).split('.')[0]

        categ_folder = os.path.join(args.outdir, ucf_categ)

        if not os.path.exists(categ_folder):
            os.mkdir(categ_folder)
        else:
            continue

        for vid_group in vid_group_list:
            print('\tvid_group: ' + vid_group)
            group_videos = glob.glob(os.path.join(args.ucf, ucf_categ + '/' + vid_group +'*.avi'))
            group_video_folder = os.path.join(categ_folder, vid_group)

            if not os.path.exists(group_video_folder):
                os.mkdir(group_video_folder)

            group_frame_count = 0

            for vid in group_videos:
                print('\t\tvid: ' + vid)
                vidcap = cv2.VideoCapture(vid)
                success, image = vidcap.read()
                this_vid_frame_count = 0

                while success:
                    # center crop square and resize to 256x256
                    min_dim = min(image.shape[0],image.shape[1])
                    rescale_factor = 256 / min_dim
                    im_resize = cv2.resize(image, (0,0), fx=rescale_factor, fy=rescale_factor)

                    crop_col_start = (im_resize.shape[1] // 2 - 1) - 127
                    crop_col_end = crop_col_start + 256
                    crop_row_start = (im_resize.shape[0] // 2 - 1) - 127
                    crop_row_end = crop_row_start + 256

                    im_crop = im_resize[crop_row_start:crop_row_end, crop_col_start:crop_col_end, :]

                    assert(im_crop.shape[0] == 256, 'im_crop height not 256')
                    assert(im_crop.shape[1] == 256, 'im_crop width not 256')

                    iuv = extract_iuv(im_crop, model, infer_engine)

                    if iuv is not None:
                        image_basename = '{:08d}'.format(group_frame_count)
                        image_path = os.path.join(group_video_folder, image_basename + '.png')
                        cv2.imwrite(image_path, im_crop)

                        # iuv_path = os.path.join(group_video_folder, image_basename + "_IUV.npy")
                        # np.save(iuv_path, iuv)

                        iuv_img_path = os.path.join(group_video_folder, image_basename + "_IUV.png")
                        cv2.imwrite(iuv_img_path, iuv)
                    else:
                        print("Skipped frame {:d} of ".format(this_vid_frame_count) + vid)

                    # cv2.imshow('frame'.format(count),image)
                    # cv2.waitKey(1)
                    # success,image = vidcap.read()
                    # count += 1
                    success, image = vidcap.read()
                    this_vid_frame_count += 1
                    group_frame_count += 1


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)