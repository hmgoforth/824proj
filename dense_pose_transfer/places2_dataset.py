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
import glob
import os

import utils
import pdb
#from pdb import set_trace as st

class Places2DatasetUFCMasks(Dataset):
    ''''
    Dataset for Dense Pose Transfer using UCF dataset
    '''
    def __init__(self, ufc_filelist_path, places2_filelist_path, places2_basepath, closest_frame_distance=100):
        with open(ufc_filelist_path, "rb") as fp:
            self.ufc_filelist = pickle.load(fp)
        with open(places2_filelist_path, 'r') as f:
            self.places2_filelist = f.read().splitlines()
        self.places2_basepath = places2_basepath

        self.closest_frame_distance = closest_frame_distance


    def __len__(self):
        return len(self.places2_filelist)

    def __getitem__(self, idx):

        item_basepath = self.ufc_filelist[np.random.randint(len(self.ufc_filelist))]
        #pdb.set_trace()
        #im = torch.from_numpy(io.imread(item_basepath + '.png')).permute(2,0,1).float() / 255.
        im_iuv = torch.flip(torch.from_numpy(io.imread(item_basepath + '_IUV.png')).permute(2,0,1).float(),dims=[0])
        places2_impath = self.places2_basepath + self.places2_filelist[idx]

        im = torch.from_numpy(io.imread(places2_impath)).permute(2,0,1).float() / 255

        other_group_im = glob.glob(os.path.join(os.path.dirname(item_basepath), '*_IUV.png'))
        target_basepath = '_'.join(other_group_im[np.random.randint(len(other_group_im))].split('_')[:-1])

        item_frame_num = int(os.path.basename(item_basepath))
        target_frame_num = int(os.path.basename(target_basepath))

        # keep searching if random frame is too close
        while abs(item_frame_num - target_frame_num) < self.closest_frame_distance:
            target_basepath = '_'.join(other_group_im[np.random.randint(len(other_group_im))].split('_')[:-1])
            target_frame_num = int(os.path.basename(target_basepath))

        target_im = torch.from_numpy(io.imread(target_basepath + '.png')).permute(2,0,1).float() / 255
        target_iuv = torch.flip(torch.from_numpy(io.imread(target_basepath + '_IUV.png')).permute(2,0,1).float(),dims=[0])

        #print('item: ' + item_basepath)
        #print('target: ' + target_basepath)
        #pdb.set_trace()
        ret = {'im':im,
               'im_iuv':im_iuv,
               'target_iuv':target_iuv
              }

        return ret

def parse_args():
    parser = argparse.ArgumentParser(description='preprocess IUV for deepfashion')
    parser.add_argument(
        '--ufcfilelist',
        help='location of ucf filedict',
        default='ucf_filelist.txt',
        type=str
    )
    parser.add_argument(
        '--places2filelist',
        help='location of places2 filedict',
        default='../places365_standard/train.txt',
        type=str
    )
    parser.add_argument(
        '--places2_basepath',
        help='location of places2 filedict',
        default='../places365_standard/',
        type=str
    )
    return parser.parse_args()

def main(args):

    ds = Places2DatasetUFCMasks(args.ufcfilelist, args.places2filelist, args.places2_basepath)
    for idx in range(len(ds)):
        print(idx)
        data=ds[idx]
        # im_texture=utils.texture_from_images_and_iuv(data['im'].unsqueeze(0),data['im_iuv'].unsqueeze(0))
        # utils.plot_texture_map(im_texture.squeeze(0))

if __name__ == '__main__':
    args = parse_args()
    main(args)