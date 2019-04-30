from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils

import argparse
from datetime import datetime

import network
import dataset
import pdb

def train(args, model, optimizer, dataset):
    for epoch in range(args.num_epochs):
        num_batches = len(dataset)
        for batch_id, batch_data in enumerate(dataset):
            model.train()
            current_step = epoch * num_batches + batch_id
           
            dense_input = batch_data[].cuda()
            dense_target = batch_data[].cuda()
            
            predictive_output = model(im_texture)

            loss =  
