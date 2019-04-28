'''
run after create_filelist
Using the filelist, process each image and saves a texture map (24 x 3 x 256 x 256) for the image
These texture maps are loaded in the dataset
'''

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from skimage import io
import argparse
import matplotlib.pyplot as plt
import pickle

from pdb import set_trace as st

import utils

def parse_args():
    parser = argparse.ArgumentParser(description='preprocess IUV for deepfashion')
    parser.add_argument(
        '--filedict',
        help='location of deepfashion filedict',
        default='deepfashion_filelist.txt',
        type=str
    )
    return parser.parse_args()

def main(args):
    with open(args.filedict, "rb") as fp:
            filedict = pickle.load(fp)

    filelist = filedict['filelist']
    filecount = 0

    for item in filelist:
    	item_path = item['path']
    	im, iuv = utils.read_image_and_iuv(item_path)
    	im = im.unsqueeze(0).float()
    	iuv = iuv.unsqueeze(0)

    	texture = utils.texture_from_images_and_iuv(im, iuv)
    	texture = texture.squeeze(0).numpy()
    	# texture = texture.swapaxes(1,3).swapaxes(1,2)
   
    	# plt.imshow(texture[1,:,:,:].astype(int))
    	# plt.show()
    	# st()
    	
    	np.save(item_path + '_texture.npy', texture)

    	print('{:d}/{:d}'.format(filecount + 1, len(filelist)))
    	filecount = filecount + 1


if __name__ == '__main__':
    args = parse_args()
    main(args)