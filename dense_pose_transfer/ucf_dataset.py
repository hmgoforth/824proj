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

if __name__ == '__main__':
    import sys
    sys.path.append('../inpainting')

import utils

from pdb import set_trace as st

class UCFDensePoseTransferDataset(Dataset):
    ''''
    Dataset for Dense Pose Transfer using UCF dataset
    '''
    def __init__(self, filelist_path, closest_frame_distance=100):
        with open(filelist_path, "rb") as fp:
            self.filelist = pickle.load(fp)

        self.closest_frame_distance = closest_frame_distance

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        item_basepath = self.filelist[idx]

        im = torch.from_numpy(io.imread(item_basepath + '.png')).permute(2,0,1).float() / 255
        im_iuv = torch.flip(torch.from_numpy(io.imread(item_basepath + '_IUV.png')).permute(2,0,1).float(),dims=[0])

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

        print('item: ' + item_basepath)
        print('target: ' + target_basepath)

        ret = {'im':im,
               'im_iuv':im_iuv,
               'target_im':target_im,
               'target_iuv':target_iuv}

        return ret

def parse_args():
    parser = argparse.ArgumentParser(description='preprocess IUV for deepfashion')
    parser.add_argument(
        '--filelist',
        help='location of ucf filedict',
        default='ucf_filelist.txt',
        type=str
    )
    return parser.parse_args()

def main(args):

    ds = UCFDensePoseTransferDataset(args.filelist)
    for idx in range(len(ds)):
        print(idx)
        data=ds[idx]
        # im_texture=utils.texture_from_images_and_iuv(data['im'].unsqueeze(0),data['im_iuv'].unsqueeze(0))
        # utils.plot_texture_map(im_texture.squeeze(0))

if __name__ == '__main__':
    args = parse_args()
    main(args)