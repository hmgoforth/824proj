import random
import torch
from PIL import Image
import torchvision.transforms as transforms
from glob import glob
import pdb
import numpy as np
from scipy import ndimage

class customData(torch.utils.data.Dataset):
    def __init__(self, imgPath, maskPath, img_transform, mask_transform,
                 split='train'):
        super(customData, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform  
        self.paths = [imgPath]
        self.mask_paths = [maskPath]
        self.N_mask = len(self.mask_paths)

    def __getitem__(self, index):
        gt_img = Image.open(self.paths[index])
        gt_img = self.img_transform(gt_img.convert('RGB'))
        
        
        mask = np.load(self.mask_paths[index])
        mask = mask[:,:,0] > 1.
        mask = ndimage.binary_dilation(mask, iterations=5).astype(np.float32)
        mask = mask == 0.
        mask = mask.astype(np.float32)
        mask = np.array([mask, mask, mask], dtype=np.float32)
        #pdb.set_trace()
        mask = transforms.ToPILImage()(torch.from_numpy(mask))
        #mask = Image.fromarray(mask.astype(np.float32) * 255)#.convert('1')
        #pdb.set_trace()
        mask = self.mask_transform(mask.convert('RGB'))
        
        masked_im = gt_img * mask
        # finalMask = torch.zeros_like(mask, dtype=torch.float32)
        # background = mask == 0.
        # finalMask = finalMask.masked_fill_(background, 1.0)
        # mask = finalMask

        # mask = Image.open(self.mask_paths[index])
        # 
        return masked_im, mask, gt_img

    def __len__(self):
        return len(self.paths)
