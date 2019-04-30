from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils

import argparse
from datetime import datetime

import network
import dataset
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
        defualt=40,
        type=int
    )
    parser.add_argument(
        '--lr'
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
    return parser.parse_args()

def main(args):
    model_save_path = args.model_save + str(datetime.today().day) + '_' + str(datetime.today().month)
    net = network.UNet
