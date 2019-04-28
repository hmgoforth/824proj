import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage import io
import numpy as np
import torch

from pdb import set_trace as st

def read_image_and_iuv(path):
    # read image and IUV from base path
    image_path = path + '.jpg'
    im = torch.from_numpy(io.imread(image_path)).permute(2,0,1)
    iuv_path = path + '_IUV.npy'
    iuv = torch.from_numpy(np.load(iuv_path)).permute(2,0,1)
    return im, iuv

def images_from_texture_and_iuv(texture_map, iuv):
	# return images corresponding to iuv, which are filled in using texture maps
	# texture: torch, 24 x (R, G, B) x H x W
	# IUV: torch, views x (I, U, V) x H x W
	# output images: views x (R, G, B) x H x W

	views = iuv.shape[0]
	images = torch.zeros(views, 3, 256, 256).view(-1)

	part_ind = (iuv[:, 0:1, :, :].repeat(1, 3, 1, 1).contiguous().view(-1)).long() - 1
	row_ind = (iuv[:, 1:2, :, :].repeat(1, 3, 1, 1).contiguous().view(-1)).long()
	col_ind = (iuv[:, 2:3, :, :].repeat(1, 3, 1, 1).contiguous().view(-1)).long()

	channel_ind = torch.arange(3)
	channel_ind = channel_ind.unsqueeze(1).repeat(1, 256 * 256)
	channel_ind = channel_ind.unsqueeze(0).repeat(views, 1, 1).view(-1)

	valid_part = part_ind >= 0

	images[valid_part] = texture_map[part_ind[valid_part], channel_ind[valid_part], row_ind[valid_part], col_ind[valid_part]]
	images = images.view(views, 3, 256, 256)

	return images

def texture_from_images_and_iuv(images, iuv, size_texture=256):
	# return texture maps from images and IUV
	# images: torch, B x (R, G, B) x H x W
	# IUV: torch, B x (I, U, V) x H x W
	# output texture map: B x 24 x 3 x size_texture x size_texture

	B = images.shape[0]
	texture_map = torch.zeros(B, 24, 3, size_texture, size_texture)

	intensities = images.unsqueeze(1).repeat(1, 24, 1, 1, 1).contiguous().view(-1)

	iuv = iuv.unsqueeze(1).repeat(1, 24, 1, 1, 1)

	part_ind = (iuv[:,:,0:1,:,:].repeat(1, 1, 3, 1, 1).contiguous().view(-1)).long() - 1
	row_ind = (iuv[:,:,1:2,:,:].repeat(1, 1, 3, 1, 1).contiguous().view(-1)).long()
	col_ind = (iuv[:,:,2:3,:,:].repeat(1, 1, 3, 1, 1).contiguous().view(-1)).long()
	
	channel_ind = torch.arange(3)
	channel_ind = channel_ind.unsqueeze(1).repeat(1, size_texture * size_texture)
	channel_ind = channel_ind.unsqueeze(0).repeat(B * 24, 1, 1).view(-1)

	view_ind = torch.arange(B).unsqueeze(1).repeat(1, 24 * 3 * size_texture * size_texture).view(-1)	

	valid_part = part_ind >= 0

	intensities_valid = intensities[valid_part]

	texture_map[view_ind[valid_part], part_ind[valid_part], channel_ind[valid_part], row_ind[valid_part], col_ind[valid_part]] = intensities[valid_part]

	return texture_map

def plot_texture_map(texture):
	# input: torch float 24 x 3 x size_texture x size_texture [0, 1]
	# out: plot part textures in 4 x 6 window

	plt.figure(figsize = (6*1.7,4*1.7))
	gs1 = gridspec.GridSpec(4, 6)
	gs1.update(wspace=0.0, hspace=0.0) # set the spacing between axes. 

	for i in range(24):
	    ax1 = plt.subplot(gs1[i])
	    plt.axis('on')
	    ax1.set_xticklabels([])
	    ax1.set_yticklabels([])
	    ax1.imshow(texture[i,:,:,:].permute(1,2,0))

	plt.show()