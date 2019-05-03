from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils

import argparse
from datetime import datetime as date

import network
from predict_data import DFDenseData
import loss_function

from pdb import set_trace as st

def parse_args():
    parser = argparse.ArgumentParser(description='train inpainting autoencoder')
    parser.add_argument(
        '--model-save',
        help='location to save models',
        default='models/',
        type=str
    )
    parser.add_argument(
        '--filedict',
        help='location of deepfashion filedict',
        default='deepfashion_filelist.txt',
        type=str
    )
    parser.add_argument(
        '--pathtoind',
        help='where to find pathtoind_dict',
        default='deepfasion_pathtoind.txt',
        type=str
    )
    parser.add_argument(
        '--textures',
        help='location of hdf5 training textures file',
        default='deepfashion_textures.hdf5',
        type=str
    )
    parser.add_argument(
        '--batch',
        default=5,
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
        type=int
    )
    parser.add_argument(
        '--log-freq',
        default=1,
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
    model_save_path = args.model_save + str(date.today().day) + '_' + str(date.today().month)
    net = network.PredictiveModel()
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        net = net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.5,0.999))
    train_dataset = DFDenseData(args.filedict)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)

    tboard = SummaryWriter()

    for epoch in range(args.num_epochs):
        num_batches = len(train_dataloader)

        for batch_idx, batch_data in enumerate(train_dataloader):

            net.train()
            current_step = epoch * num_batches + batch_idx
            pdb.set_trace()
            images = batch_data['images'].cuda() if use_cuda else batch_data['images']
            iuvs = batch_data['iuvs'].cuda() if use_cuda else batch_data['iuvs']
            test_iuvs = batch_data['test_iuvs'].cuda() if use_cuda else batch_data['test_iuvs']
            test_images = batch_data['test_images'].cuda() if use_cuda else batch_data['test_iuvs']
            net_input = torch.cat((images, iuvs, test_iuvs), 1)
            predicted_images = net(net_input)

            loss = loss_function.pred_loss(predicted_images, test_images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if current_step % args.log_freq == 0:
                print("Epoch: {}, Batch {}/{} has loss {}".format(epoch, batch_idx, num_batches, loss))
                tboard.add_scalar('train/loss', loss, current_step)

            if current_step % args.image_log_freq == 0:
                partgrid = torchvision.utils.make_grid(predicted_images[0,:,:,:], nrow=6, padding=0)
                tboard.add_image('train/inpainted_{:d}'.format(current_step), partgrid, current_step)

            if current_step % args.model_save_freq == 0:
                torch.save(net.state_dict(), model_save_path + '_' + str(current_step) + '.pth')

if __name__ == '__main__':
    args = parse_args()
    main(args)
