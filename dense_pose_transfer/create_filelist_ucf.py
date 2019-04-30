'''
Create and save list of all file paths in the UCF dataset
To be used in the UCF dataset loader
'''

import argparse
import os
from pdb import set_trace as st
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description='preprocess IUV for deepfashion')
    parser.add_argument(
        '--ucf',
        help='location of ucf dataset',
        default='../UCF-dataset/preprocessed',
        type=str
    )
    parser.add_argument(
        '--outfile',
        help='location of UCF filelist outfile',
        default='ucf_filelist.txt',
        type=str
    )
    return parser.parse_args()

def main(args):
    filelist = []

    for root, dirs, files in os.walk(args.ucf):
        for file in files:
            if file.endswith("IUV.png"): # only look for valid files
                item_name = file.split('_')[0]
                item_path = os.path.join(root, item_name)

                filelist.append(item_path)

    with open(args.outfile, "wb") as fp:
        pickle.dump(filelist, fp)

if __name__ == '__main__':
    args = parse_args()
    main(args)