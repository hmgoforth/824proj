import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from skimage import io
import argparse
import matplotlib.pyplot as plt
import pickle
import time
import h5py

import utils

from pdb import set_trace as st

class DeepfashionInpaintingDataset(Dataset):
    ''''
    Dataset for Inpainting, using DeepFashion images
    '''
    def __init__(self, filedict_path, pathtoind_path, texture_maps_path, max_multiview=None):
        with open(filedict_path, "rb") as fp:
            filedict = pickle.load(fp)

        with open(pathtoind_path, "rb") as fp:
            self.pathtoind = pickle.load(fp)

        f = h5py.File(texture_maps_path, 'r')
        self.texture_maps = f['texture_maps']

        self.filelist = filedict['filelist']

        # max number of views to sample per example
        # note that there may not be this many views available
        if max_multiview is None:
            self.max_multiview = filedict['max_multiview']
        else:
            self.max_multiview = max_multiview

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        item = self.filelist[idx]
        item_path = item['path']

        # start = time.time()
        # load texture map for this idx
        im_texture = torch.from_numpy(self.texture_maps[idx, :, :, :, :]) / 255

        # load texture maps for multiple views
        mv_texture = torch.zeros(self.max_multiview, 24, 3, im_texture.shape[2], im_texture.shape[3])

        for i, view_path in enumerate(item['views']):
            view_idx = self.pathtoind[view_path]
            mv_texture[i, :, :, :, :] = torch.from_numpy(self.texture_maps[view_idx, :, :, :, :]) / 255

        # end = time.time()
        # num_reads = num_views + 1
        # print('elapsed: {:.3f}'.format((end-start)/num_reads))
        # print(item['views'])
        # print(item_path)

        num_views = len(item['views'])
        
        ret_dict = {'im_texture': im_texture,
                    'mv_texture': mv_texture,
                    'num_views': num_views}

        return ret_dict

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
    ds = DeepfashionInpaintingDataset(args.filedict, args.pathtoind_dict, args.hdf5_file)
    for idx in range(len(ds)):
        print(idx)
        data=ds[idx]
        utils.plot_texture_map(data['im_texture'])

if __name__ == '__main__':
    args = parse_args()
    main(args)