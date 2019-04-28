'''
run after preprocess_texturemaps (should have saved preprocess_texturemaps into h5py oh well)
Using the filelist, read each npy texture map and save into h5py file
'''

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from skimage import io
import argparse
import matplotlib.pyplot as plt
import h5py
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
    parser.add_argument(
        '--pathtoind-dict',
        help='where to save pathtoind_dict',
        default='deepfasion_pathtoind.txt',
        type=str
    )
    parser.add_argument(
        '--hdf5-file',
        help='where to save hdf5 file',
        default='deepfashion_textures.hdf5',
        type=str
    )
    return parser.parse_args()

def main(args):
    with open(args.filedict, "rb") as fp:
            filedict = pickle.load(fp)

    filelist = filedict['filelist']
    filecount = 0

    # create path->filelist index
    path_to_filelist_ind = {}
    for idx, item in enumerate(filelist):
    	path_to_filelist_ind[item['path']] = idx

    with open(args.pathtoind_dict, "wb") as fp:
        pickle.dump(path_to_filelist_ind, fp)

    # create h5py file
    f = h5py.File(args.hdf5_file, "w")
    dataset_len = len(filelist)
    f.create_dataset('texture_maps',shape=(dataset_len, 24, 3, 256, 256))

    for idx, item in enumerate(filelist):
        print("{:d}/{:d}".format(idx+1, dataset_len))
        item_path = item['path']
        texture_path = item_path + '_texture.npy'
        f['texture_maps'][idx, :, :, :, :] = np.load(texture_path)

    f.close()

if __name__ == '__main__':
    args = parse_args()
    main(args)