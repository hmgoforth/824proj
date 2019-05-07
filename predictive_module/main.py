from train_predict import train
import network
from predict_data import DFDenseData
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

import argparse
from datetime import datetime


import pdb

def parse_args():
    parser = argparse.ArgumentParser(description='train predictive module')
    parser.add_argument(
        '--model-save',
        help='where to save predictive model',
        default='models/',
        type=str
    )
    parser.add_argument(
        '--filedict',
        help='deepfashion files',
        default='deepfashion_filelist.txt',
        type=str
    )
    parser.add_argument(
        '--pathtoind',
        help='path to ind dictionary filepath',
        default='deepfashion_pathtoind.txt',
        type=str
    )
    parser.add_argument(
        '--textures',
        help='hdf5 textures filepath',
        default='deepfashion_textures.hdf5',
        type=str
    )
    parser.add_argument(
        '--batch',
        default=100,
        type=int
    )
    parser.add_argument(
        '--num-epochs',
        default=40,
        type=int
    )
    parser.add_argument(
        '--lr',
        default=2e-4,
        type=int,
    )
    parser.add_argument(
        '--log-freq',
        default=2,
        type=int
    )
    parser.add_argument(
        '--image-log-freq',
        default=50,
        type=int
    )
    parser.add_argument(
        '--model-save-freq',
        default=1000,
        type=int
    )
    parser.add_argument(
        '--loaded_images_h5py',
        default=None,
        type=str
    )
    parser.add_argument(
        '--loaded_images_dir',
        default=None,
        type=str
    )
    parser.add_argument(
        '--load_images',
        default=False,
        type=bool
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    model_save_path = args.model_save + str(datetime.today().day) + '_' + str(datetime.today().month)

    predict_model = network.PredictiveModel()
    use_cuda = torch.cuda.is_available()
    
    if use_cuda:
        predict_model = predict_model.cuda()

    optimizer = optim.Adam(predict_model.parameters(), lr=4e-3, betas=(0.5, 0.999))
    validation_split = 0.2
    shuffle_dataset = True
    random_seed = 42

    dataset = DFDenseData(args.filedict)
    print(len(dataset))

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_dataloader = DataLoader(dataset, batch_size=args.batch, sampler=train_sampler)
    test_dataloader = DataLoader(dataset, batch_size=args.batch, sampler=test_sampler)
    train(args, predict_model, optimizer, train_dataloader, model_save_path) 
        

