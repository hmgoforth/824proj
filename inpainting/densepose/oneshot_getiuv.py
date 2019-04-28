'''
Given source image and target image/dir, run DensePose to get IUV for source and target
Save IUV out
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
        '--source',
        help='location of source image',
        type=str
    )
    parser.add_argument(
        '--target',
        help='location of target image/dir',
        type=str
    )
    parser.add_argument(
        '--target-ext',
        help='extension of target images if directory was provided',
        default='.jpg',
        type=str
    )
    parser.add_argument(
        '--outdir',
        help='where to save output IUV',
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

def extract_and_save_iuv(image_path, outdir, model, infer_engine):
	# image_path: /path/to/the/image.jpg
	iuv = extract_iuv(image_path, model, infer_engine)
	filename = os.path.splitext(os.path.basename(image_path))[0]
	iuv_path = os.path.join(outdir, filename + '_IUV.npy')
	iuv_image_path = os.path.join(outdir, filename + '_IUV.png')
	np.save(iuv_path, iuv)
	cv2.imwrite(iuv_image_path, iuv)

def main(args):
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    model = infer_engine.initialize_model_from_cfg(args.weights)

    # extract source
    if os.path.isfile(args.source):
    	print('extracting source ...')
    	extract_and_save_iuv(args.source, args.outdir, model, infer_engine)
    else:
    	sys.exit("source not a file")

    # extract target(s)
    if os.path.isfile(args.target):
    	print('extracting target ...')
    	if os.path.isdir(args.target):
    		for idx, file in enumerate(os.listdir(args.target)):
    			if file.endswith(args.target_ext):
    				target_outdir = os.path.join(args.outdir, os.path.basename(args.target))
    				extract_and_save_iuv(file, target_outdir, model, infer_engine)
    			print('\t target file ... {:d}'.format(idx))
    	else:
    		extract_and_save_iuv(args.target, args.outdir, model, infer_engine)
    else:
    	sys.exit("target not a file")

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)