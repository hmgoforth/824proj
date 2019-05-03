import utils
import torch
from torch.utils.data import Dataset, dataloader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
from PIL import Image
import numpy as np
from skimage import io
import argparse
import matplotlib.pyplot as plt
import pickle as pkl
import time
import h5py
import os
import pdb
import glob
from datetime import datetime

DIR=os.getcwd()
DATETIME=datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class DFDenseData(Dataset):
    ''''
    Dataset for Predictive Model using DeepFashion images
    '''
    def __init__(self, filedict="deepfashion_filelist.txt"):
        self.filedict = filedict
        self.transforms = transforms.Compose([
            transforms.Resize(256, 256),
            transforms.ToTensor(),
            ])
         
        self.image_dir = {}
        if self.filedict:
            with open(self.filedict, "rb") as fp:
                loaded_filedict = pkl.load(fp)
            self.loaded_filedict = loaded_filedict
            self.filelist = loaded_filedict['filelist']
            self.max_multiview = loaded_filedict['max_multiview']
        else:
            print("No filedict added!!")
 
           
    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        item = self.filelist[int(idx)]
        item_path = item['path']
        im, iuv = utils.read_image_and_iuv(item_path)
        train_images = torch.zeros(self.max_multiview, 3, 256, 256)
        train_iuvs = torch.zeros(self.max_multiview, 3, 256, 256)
        target_iuvs = torch.zeros(self.max_multiview, 3, 256, 256)
        target_images = torch.zeros(self.max_multiview, 3, 256, 256)
        for view_path in enumerate(item['views']):
            target_im, target_iuv = utils.read_image_and_iuv(view_path[1])
            target_iuvs[view_path[0], :, :, :] = target_iuv
            target_images[view_path[0], :, :, :] = target_im
            train_iuvs[view_path[0], :, :, :] = iuv
            train_images[view_path[0], :, :, :] = im
             
        return {'images':train_images, 'iuvs':train_iuvs,  'target_iuvs':target_iuvs, 'target_images':target_images, 'views':len(item['views'])}
