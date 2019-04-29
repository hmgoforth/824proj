import argparse
import torch
from torchvision import transforms

import opt
from customDataset import *
from evaluation import evaluate
from evaluation import evaluateCustom
from net import PConvUNet
from util.io import load_ckpt

parser = argparse.ArgumentParser()
# training options
parser.add_argument('--imPath', type=str, default='../posewarp/data/hunter_256.jpeg')
parser.add_argument('--maskPath', type=str, default='../posewarp/data/hunter_256_IUV.npy')
#parser.add_argument('--maskPath', type=str, default='/home/cloudlet/vlr_824/project/backgroundInpainting/mask/003328.jpg')
parser.add_argument('--modelPath', type=str, default='./pretrainedModel.pth')
parser.add_argument('--image_size', type=int, default=256)
args = parser.parse_args()

device = torch.device('cuda')

size = (args.image_size, args.image_size)
img_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor(),
     transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
mask_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor()])

dataset_val = customData(args.imPath, args.maskPath, img_transform, mask_transform, 'val')

model = PConvUNet()
load_ckpt(args.modelPath, [('model', model)])
model.to(device)


model.eval()
evaluateCustom(model, dataset_val, device, 'result.jpg')