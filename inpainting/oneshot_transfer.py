'''
Given input source image, target image/directory, and source/target IUV, output image of source
in target's pose for all target images
'''

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from skimage import io
import argparse
import matplotlib.pyplot as plt
import pickle
import cv2
import glob
from pdb import set_trace as st

import utils
import network

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
        default='.jpg'
        type=str
    )
    parser.add_argument(
        '--iuv',
        help='where the IUV is for source/target',
        type=str
    )
    parser.add_argument(
        '--net',
        help='network file',
        type=str
    )
    return parser.parse_args()

def main(args):
	# load source image, source IUV, target IUV
    source_img = torch.from_numpy(cv2.imread(args.source)).unsqueeze(0)
    source_path_base = os.path.splitext(os.path.basename(image_path))[0]
    source_iuv_path = os.path.join(args.iuv, source_path_base + '_IUV.npy')
    source_iuv = torch.from_numpy(np.load(source_iuv_path)).unsqueeze(0)

    if os.path.isdir(args.target):
    	target_im = glob.glob(args.target + '/*' + args.target_ext)
    	target_im = target_im.sort()
    	target_iuv = torch.zeros(len(target_im), 3, 256, 256)

		for idx, img in enumerate(target_im):
			target_path_base = os.path.splitext(os.path.basename(img))[0]
			target_iuv_path = os.path.join(args.iuv, target_path_base + '_IUV.npy')
			target_iuv[idx, :, :, :] = torch.from_numpy(np.load(target_iuv_path))
	else:
		target_path_base = os.path.splitext(os.path.basename(args.target))[0]
		target_iuv_path = os.path.join(args.iuv, target_path_base + '_IUV.npy')
		target_iuv = torch.from_numpy(np.load(target_iuv_path)).unsqueeze(0)

    # load inpainting network
    net = network.InpaintingAutoencoder()
    net.load_state_dict(torch.load(args.net))
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        net = net.cuda()

    net.eval()

    # extract texture maps of source image
    source_texture = utils.texture_from_images_and_iuv(source_img, source_iuv)

    # network inference
    source_inpainted = net(im_texture)

    # map inpainted texture maps to target IUV
    transfer_result = utils.images_from_texture_and_iuv(source_inpainted, target_iuv)

    # save output
    


if __name__ == '__main__':
    args = parse_args()
    main(args)