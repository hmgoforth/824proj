import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import pdb
import argparse
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from scipy import ndimage
import torchvision.models as M
import torch.nn.functional as F
from ucf_dataset import *

from networks import *
import utils
from tensorboardX import SummaryWriter
import os

class ExperimentRunner(object):
    """
    Main Class for running experiments
    This class creates the GAN, as well as the network for VGG Loss (VGG Net should be frozen)
    This class also creates the datasets
    """
    def __init__(self, train_dataset_path, test_dataset_path, train_batch_size, test_batch_size, model_save_dir, num_epochs=100, num_data_loader_workers=10):
        # GAN Network + VGG Loss Network
        self.gan = DensePoseGAN()
        self.vgg_loss_network = VGG19FeatureNet() #Frozen weights, pretrained

        # Network hyperparameters
        self.gan_lr = 1e-4
        self.disc_lr = 1e-4
        self.disc_lambda = 0.1
        self.optimizer = torch.optim.SGD([ {'params': self.gan.generator.parameters(), 'lr': self.gan_lr},
                                           {'params': self.gan.discriminator.parameters(), 'lr': self.disc_lr}
                                         ], momentum=0.9)
        # Network losses
        self.BCECriterion = nn.BCEWithLogitsLoss().cuda()
        self.VGGLoss = VGGLoss().cuda()

        # Train settings + log settings
        self.num_epochs = num_epochs
        self.start_disc_iters = 5
        self.log_freq = 10  # Steps
        self.test_freq = 1000  # Steps
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        # Create datasets
        self.train_dataset = UCFDensePoseTransferDataset(train_dataset_path)
        self.test_dataset = UCFDensePoseTransferDataset(test_dataset_path)

        self.train_dataset_loader = DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=num_data_loader_workers)
        self.test_dataset_loader = DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False, num_workers=num_data_loader_workers)

        # Use the GPU if it's available.
        self.cuda = torch.cuda.is_available()

        if self.cuda:
            self.gan.cuda()
            self.vgg_loss_network.cuda()

        # Tensorboard logger
        self.txwriter = SummaryWriter()
        self.model_save_dir = model_save_dir
        self.save_freq = 5000

    def _optimizeGAN(self, pred_img, gt_img, y_pred, y_gt):
        """
        VGGLoss + GAN loss
        """
        self.optimizer.zero_grad()
        loss = self.disc_lambda * self.BCECriterion(y_pred, y_gt) + self.VGGLoss(pred_img, gt_img)
        loss.backward()
        self.optimizer.step()
        return loss
    
    def _optimizeDiscriminator(self, y_pred, y_gt):
        """
        Discriminator Loss
        """
        self.optimizer.zero_grad()
        loss = self.BCECriterion(y_pred, y_gt)
        loss.backward()
        self.optimizer.step()
        return loss
    
    def _adjust_learning_rate(self, epoch):
        """
        TODO
        """
        raise NotImplementedError()
    def _clip_weights(self):
        """
        TODO
        """
        raise NotImplementedError()
    
    def test(self):
        """
        TODO: Kevin: How do we verify model performance?
        """
        pred = []
        targ = []
        correct = 0.0
        number_samples = 0.0
        
        for batch_id, batch_data in enumerate(self.test_dataset_loader):
            # Set to eval()
            self.gan.eval()
            self.vgg_loss_network().eval()

            # Print progress
            if batch_id % 1000 == 0:
                print("Test batch: {}/{}".format(batch_id, len(self.test_dataset_loader)))
            
            # Get data from dataset
            src_img = batch_data['im'].cuda(async=True)
            target_img = batch_data['target_im'].cuda(async=True)
            src_iuv = batch_data['im_iuv'].cuda(async=True)
            target_iuv = batch_data['target_iuv'].cuda(async=True)

            # ============
            # Run predictive GAN on source image
            predicted_answer, classification_src = self.gan(src_img, source_iuv, target_iuv, use_gt=False)

            """ 
            TODO : What do we want to plot?
            """
        return

    def train(self):
        """
        Main training loop
        Helpful URL: https://github.com/balakg/posewarp-cvpr2018/blob/master/code/posewarp_gan_train.py
        """
        for epoch in range(self.num_epochs):
            num_batches = len(self.train_dataset_loader)
            # Initialize running averages
            disc_losses = AverageMeter()
            train_disc_accuracies = AverageMeter()
            tot_losses = AverageMeter()
            train_accuracies = AverageMeter()

            for batch_id, batch_data in enumerate(self.train_dataset_loader):
                self.gan.train()  # Set the model to train mode
                self.vgg_loss_network.eval()
                current_step = epoch * num_batches + batch_id

                # Get data from dataset
                src_img = batch_data['im'].cuda(async=True)
                target_img = batch_data['target_im'].cuda(async=True)
                src_iuv = batch_data['im_iuv'].cuda(async=True)
                target_iuv = batch_data['target_iuv'].cuda(async=True)
                
                # ============
                # Run predictive GAN on source image
                predicted_answer, classification_src = self.gan(src_img, src_iuv, target_iuv, use_gt=False)
                # Run predictive GAN on target image
                _ , classification_tgt = self.gan(target_img, src_iuv, target_iuv, use_gt=True)
                # Create discriminator groundtruth
                # For src, we create zeros
                # For tgt, we create ones
                disc_gt_src = torch.zeros(classification_src.shape[0], 1, dtype=torch.float32)
                disc_gt_tgt = torch.ones(classification_src.shape[0], 1, dtype=torch.float32)
                disc_gt = torch.cat((disc_gt_src, disc_gt_tgt), dim=0).cuda(async=True)

                classification_all = torch.cat((classification_src, classification_tgt) , dim=0)
                # Train Discriminator network
                disc_loss = self._optimizeDiscriminator(classification_all, disc_gt)
                disc_losses.update(disc_loss.data[0], disc_gt.shape[0])
                disc_acc = 100.0 * torch.mean( ( torch.round(F.softmax(classification_all, dim=1)) == disc_gt ).float() )

                train_disc_accuracies.update(disc_acc.data[0], disc_gt.shape[0])

                print("Epoch: {}, Batch {}/{} has Discriminator loss {}, and acc {}".format(epoch, batch_id, num_batches, disc_losses.avg, train_disc_accuracies.avg))
                # Start training GAN first for several iterations
                if current_step < self.start_disc_iters:
                    print("Discriminator training only: {}/{}\n").format(current_step,self.start_disc_iters)
                    continue
               
                # ============
                # Optimize the GAN
                # Note that now we use disc_gt_tgt which are 1's
                tot_loss = self._optimizeGAN(predicted_answer, target_img, classification_src, disc_gt_tgt)
                tot_losses.update(tot_loss.data[0], disc_gt_tgt.shape[0])

                acc = 100.0 * torch.mean( ( torch.round(F.softmax(classification_src, dim=1)) == disc_gt_tgt ).float() )

                tot_losses.update(tot_loss.data[0], disc_gt_tgt.shape[0])
                train_accuracies.update(acc.data[0], disc_gt_tgt.shape[0])

                # Not adjusting learning rate currently
                # if epoch % 100 == 99:
                #     self._adjust_learning_rate(epoch)
                # # Not Clipping Weights
                # self._clip_weights()

                if current_step % self.log_freq == 0:
                    print("Epoch: {}, Batch {}/{} has loss {}, and acc {}".format(epoch, batch_id, num_batches, tot_losses.avg, train_accuracies.avg))
                    # TODO: you probably want to plot something here
                    self.txwriter.add_scalar('train/loss', tot_losses.avg, current_step)
                    self.txwriter.add_scalar('train/acc', train_accuracies.avg, current_step)
                """
                TODO : Test accuracies
                if current_step % self.test_freq == 0:#self._test_freq-1:
                    self._model.eval()
                    val_accuracy = self.validate()
                    print("Epoch: {} has val accuracy {}".format(epoch, val_accuracy))
                    self.txwriter.add_scalar('test/acc', val_accuracy, current_step)
                """
                """
                Save Model periodically
                """
                if (current_step % self.save_freq == 0) and current_step > 0:
                    save_name = os.path.join(
                        self.model_save_dir, 'model_iter_{}.h5'.format(current_step))
                    self.gan.save_net(save_name, gan)
                    print('Saved model to {}'.format(save_name))
            return
                   
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    # Feel free to add more args, or change/remove these.
    parser = argparse.ArgumentParser(description='Run Densepose Transfer Network.')
    parser.add_argument('--train_dataset_path', type=str, default='./ucf_train_list.txt')
    parser.add_argument('--test_dataset_path', type=str, default='./ucf_test_list.txt')
    parser.add_argument('--train_batch_size', type=int, default=5)
    parser.add_argument('--test_batch_size', type=int, default=5)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_data_loader_workers', type=int, default=10)
    parser.add_argument('--model_save_dir', type=str, default='./models')
    args = parser.parse_args()

    # Create experiment runner object
    # Loads data, creates models
    experiment_runner = ExperimentRunner( train_dataset_path=args.train_dataset_path,
                                          test_dataset_path=args.test_dataset_path, 
                                          train_batch_size=args.train_batch_size,
                                          test_batch_size=args.test_batch_size, 
                                          model_save_dir=args.model_save_dir,
                                          num_epochs=args.num_epochs, 
                                          num_data_loader_workers=args.num_data_loader_workers)
    # Train Models
    experiment_runner.train()