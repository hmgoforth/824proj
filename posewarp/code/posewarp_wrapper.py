import tensorflow as tf
import numpy as np
import networks
import param
import os

from pdb import set_trace as st
import matplotlib.pyplot as plt

class wrapper:
    def __init__(self, model_path='../models/vgg+gan_5000.h5'):
        params = param.get_general_params()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        self.generator = networks.network_posewarp(params)
        self.generator.load_weights(model_path)

    def gen(self, x):
        # x: list[image_src, pose_src, pose_tgt, mask_src, trans]
        # see data_generation.warp_example_generator() output for more info
        # image_src: batch x 256 x 256 x 3 (rgb images)
        # pose_src: batch x 128 x 128 x 14 (gaussian bumps [0,1])
        # pose_tgt: batch x 128 x 128 x 14 (gaussian bumps [0,1])
        # mask_src: batch x 256 x 256 x 11 (?)
        # trans: batch x 2 x 3 x 11 (affine joint transformations, src to tgt)
        # returns: batch x 256 x 256 x 3 (rgb gan output)

        # disp_data(x[3],gray=True)
        # plt.title('mask')
        # plt.show()
        # plot pose_src: plt.imshow(np.sum(x[1][0,:,:,:],axis=2);plt.show()

        if len(x[0].shape) == 3: # need to add batch dim
            x[0] = np.expand_dims(x[0], 0)
            x[1] = np.expand_dims(x[1], 0)
            x[2] = np.expand_dims(x[2], 0)
            x[3] = np.expand_dims(x[3], 0)
            x[4] = np.expand_dims(x[4], 0)

        return self.generator.predict(x)