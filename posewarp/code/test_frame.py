import argparse
from pdb import set_trace as st
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import imageio

def main():
	args = read_args()

	source_img = imageio.imread(args.source)
	target_img = imageio.imread(args.target)

	plt.figure()
	plt.imshow(source_img)
	plt.figure()
	plt.imshow(target_img)
	plt.show()

	result = generate_result(source_img, target_img)
	imageio.imwrite(args.result)

def generate_result(source_img, target_img):
	# make 


def read_args():
	parser = argparse.ArgumentParser()
	
	parser.add_argument('--source',default='../data/test/frames/golf/4.png',help='source image')
	parser.add_argument('--target',default='../data/test/frames/golf/8.png',help='target image')
	parser.add_argument('--model',default='../models/vgg+gan_5000.h5',help='network')
	parser.add_argument('--result',default='result.png',help='where to save result image')

	args = parser.parse_args()

	return args

if __name__ == "__main__":
	main()