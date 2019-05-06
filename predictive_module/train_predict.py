from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils
import loss_function
import argparse
from datetime import datetime

import network
import pdb

def unNormalize(image_batch, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    for image in image_batch:
        for im_tens, m, s in zip(image, mean, std):
            im_tens.mul_(s).add_(m)
    return image_batch

def train(args, net, optimizer, train_dataloader, model_save_path):

    use_cuda = torch.cuda.is_available()
    
    tboard = SummaryWriter()
    for epoch in range(args.num_epochs):
        num_batches = len(train_dataloader)

        for batch_idx, batch_data in enumerate(train_dataloader):
            net.train()
            current_step = epoch * num_batches + batch_idx
            images = batch_data['images'].cuda() if use_cuda else batch_data['images']
            iuvs = batch_data['iuvs'].cuda() if use_cuda else batch_data['iuvs']
            target_iuvs = batch_data['target_iuvs'].cuda() if use_cuda else batch_data['target_iuvs']
            target_images = batch_data['target_images'].cuda() if use_cuda else batch_data['target_iuvs']
            index = 0
            #unpad_target_images = []
            #unpad_target_iuvs = []
            #unpad_images = []
            #unpad_iuvs = []
            #step_idx = 0
            #num_real_batches = iuvs.size()[0]
            #for view in enumerate(views):
            #    view = int(view[1].detach().cpu().numpy())
            #    for i in range(view):
            #        index = step_idx+i
            #        if index == num_real_batches:
            #            pdb.set_trace()
            #        unpad_target_images.append(target_images[step_idx+i, :, :, :])
            #        unpad_target_iuvs.append(target_iuvs[step_idx+i, :, :, :])
            #        unpad_images.append(images[step_idx+i, :, :, :])
            #        unpad_iuvs.append(iuvs[step_idx+i, :, :, :])
            
            #use_target_images = torch.stack(unpad_target_images)
            #use_target_iuvs = torch.stack(unpad_target_iuvs)
            #use_images = torch.stack(unpad_images)
            #use_iuvs = torch.stack(unpad_iuvs)
                #train_image = torch.stack([images[index,:,:,:], images[index,:,:,:], images[index,:,:,:]) 
            #    net_input = torch.cat((images, iuvs, target_iuv), 1)
            #    predicted_images = net(net_input)

            #    loss = loss_function.pred_loss(predicted_images, target_image)
            net_input = torch.cat((images, iuvs, target_iuvs), 1)
            predicted_images = net(net_input)
            loss = loss_function.pred_loss(predicted_images, target_images)
           
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if current_step % args.log_freq == 0:
                print("Epoch: {}, Batch {}/{} has loss {}".format(epoch, batch_idx, num_batches, loss))
                tboard.add_scalar('train/loss', loss, current_step)

            if current_step % args.image_log_freq == 0:
                #predicted_image = unNormalize(predicted_images[2,:,:,:])
                #target_image = unNormalize(target_images[2, :,:,:])
                predicted_image = unNormalize(predicted_images[2,:,:,:])
                target_image = unNormalize(target_images[2,:,:,:])
                target_iuv = unNormalize(target_iuvs[2,:,:,:])
                image = unNormalize(images[2,:,:,:])
                iuv = unNormalize(iuvs[2,:,:,:])
                 
                partgrid = torchvision.utils.make_grid(torch.stack((image, iuv, target_image, target_iuv, predicted_image)), nrow=2, padding=0)
                tboard.add_image('train/predicted_{:d}'.format(current_step), partgrid, current_step)

            if current_step % args.model_save_freq == 0:
                torch.save(net.state_dict(), model_save_path + '_' + str(current_step) + '.pth')



