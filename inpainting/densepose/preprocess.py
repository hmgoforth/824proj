'''
Generate DensePose IUV for all training images in f
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
        '--deepfashion',
        help='location of deepfashion',
        default='/dataset/deepfashion',
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

def extract_iuv(image_path, model, infer_engine):
    im = cv2.imread(image_path)

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

def main(args):
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    model = infer_engine.initialize_model_from_cfg(args.weights)

    filenum = 0

    for root, dirs, files in os.walk(args.deepfashion):
        for file in files:
            if file.endswith(".jpg"):
                image_path = os.path.join(root, file)
                iuv = extract_iuv(image_path, model, infer_engine)

                if iuv is not None:
                    image_name = os.path.splitext(file)[0]
                    iuv_path = os.path.join(root, image_name + "_IUV.npy")
                    np.save(iuv_path, iuv)

                    iuv_img_path = os.path.join(root, image_name + "_IUV.png")
                    cv2.imwrite(iuv_img_path, iuv)
                else:
                    print("Could not extract IUV for: " + image_path)

                print('filenum: {:d}'.format(filenum))
                filenum = filenum + 1

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)