import torch

from pdb import set_trace as st

def mv_loss(inpainted_texture, mv_texture):
	# input: im_texture, B x 24 x 3 x 256 x 256
	#		 mv_texture, B x views x 24 x 3 x 256 x 256
	# output: loss value

	inpainted_texture = inpainted_texture.unsqueeze(1).repeat(1, mv_texture.shape[1], 1, 1, 1, 1)
	valid_mask = mv_texture > 0
	inpainted_texture = inpainted_texture * valid_mask.float()


	inpainted_texture = inpainted_texture - mv_texture
	loss = torch.sum(torch.abs(inpainted_texture))

	return loss
