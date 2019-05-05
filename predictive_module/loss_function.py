import torch
import numpy as np
import pdb

def pred_loss(predicted_image, target_image):
	# input: im_texture, B x 24 x 3 x 256 x 256
	#		 mv_texture, B x views x 24 x 3 x 256 x 256
	# output: loss value
        valid_mask = target_image > 0
        masked_prediction = predicted_image * valid_mask.float()
	predicted_loss = masked_prediction - target_image
	loss = torch.mean(torch.abs(predicted_loss))

	return loss
