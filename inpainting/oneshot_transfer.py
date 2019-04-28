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
import os

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
        default='.jpg',
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
    source_img = torch.from_numpy(cv2.imread(args.source)).permute(2, 0, 1).unsqueeze(0).float()
    source_path_base = os.path.splitext(os.path.basename(args.source))[0]
    source_iuv_path = os.path.join(args.iuv, source_path_base + '_IUV.npy')
    source_iuv = torch.from_numpy(np.load(source_iuv_path)).permute(2, 0, 1).unsqueeze(0).float()

    if os.path.isdir(args.target):
        target_im = glob.glob(args.target + '/*' + args.target_ext)
        target_im = target_im.sort()
        target_iuv = torch.zeros(len(target_im), 3, 256, 256)

        for idx, img in enumerate(target_im):
            target_path_base = os.path.splitext(os.path.basename(img))[0]
            target_iuv_path = os.path.join(args.iuv, target_path_base + '_IUV.npy')
            target_iuv[idx, :, :, :] = torch.from_numpy(np.load(target_iuv_path)).permute(2, 0, 1).float()
    else:
        target_path_base = os.path.splitext(os.path.basename(args.target))[0]
        target_iuv_path = os.path.join(args.iuv, target_path_base + '_IUV.npy')
        target_iuv = torch.from_numpy(np.load(target_iuv_path)).permute(2,0,1).unsqueeze(0).float()

    # load inpainting network
    net = network.InpaintingAutoencoder()
    net.load_state_dict(torch.load(args.net))
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        net = net.cuda()

    net.eval()

    # extract texture maps of source image
    source_texture = utils.texture_from_images_and_iuv(source_img, source_iuv)


    if use_cuda:
        source_texture = source_texture.cuda()

    # network inference
    source_texture = source_texture / 255
    # utils.plot_texture_map(source_texture[0])

    source_inpainted = net(source_texture)
    source_inpainted = source_inpainted.squeeze(0).detach()

    # utils.plot_texture_map(source_inpainted)

    if use_cuda:
        source_inpainted = source_inpainted.cpu()

    # map inpainted texture maps to target IUV
    transfer_result = utils.images_from_texture_and_iuv(source_inpainted, target_iuv)

    # save output
    if os.path.isdir(args.target):
        target_im = target_im.sort()
        output_dir = args.target + '_transfer_output'
        os.makedirs(output_dir)

        for idx, img in enumerate(target_im):
            target_path_base = os.path.splitext(os.path.basename(img))[0]
            target_transfer_path = os.path.join(output_dir, target_path_base + '_transfer.png')

            cv2.imwrite(target_transfer_path, transfer_result[idx, :, :, :].permute(1,2,0).numpy())
    else:
        target_path_base = os.path.splitext(args.target)[0]
        target_transfer_path = target_path_base + '_transfer.png'

        cv2.imwrite(target_transfer_path, torch.flip(transfer_result[0],dims=[0]).permute(1,2,0).numpy() * 255)

if __name__ == '__main__':
    args = parse_args()
    main(args)