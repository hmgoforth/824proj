import tensorflow as tf
import os
import numpy as np
import sys
import data_generation
import param
import posewarp_wrapper

from pdb import set_trace as st
import matplotlib.pyplot as plt

def disp_data(d,gray=False):

    plt.figure()

    if gray:
        plt.imshow(d[0,:,:,0:1],cmap='gray')
    else:
        # reverse channels
        d = d[0,:,:,::-1]
        d = d - np.amin(d,axis=(0,1))
        d = d / np.amax(d,axis=(0,1))
        plt.imshow(d)

def test(gpu_id):
    params = param.get_general_params()

    test_feed = data_generation.create_feed(params, params['data_dir'], 'train')

    pw = posewarp_wrapper.wrapper()

    n_iters = 10000

    for step in range(n_iters):

        x, y = next(test_feed)

        disp_data(x[0])
        plt.title('source')
        disp_data(y)
        plt.title('target gt')

        gen = pw.gen(x)

        disp_data(gen)
        plt.title('target gan')

        plt.show()

if __name__ == "__main__":
    print('pid: ' + str(os.getpid()))
    if len(sys.argv) != 2:
        print("Need gpu id as command line argument.")
    else:
        test(sys.argv[1])
