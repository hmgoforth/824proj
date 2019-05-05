'''
Create dictionary containing:

1. Max number of multiple views found in dataset

2. List of all deep fashion images to help with loading data in dataset
Only load examples for which IUV was able to be computed by DensePose
Save list as pickle

    Format:
    [
    {image path, [other view, other view, other view]}, # for first item in dataset
    {image path, [other view, ...]}, # for second item in dataset
     ...
    ]
'''

import argparse
import os
import pdb
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description='preprocess IUV for deepfashion')
    parser.add_argument(
        '--deepfashion',
        help='location of deepfashion',
        default='/dataset/deepfashion',
        type=str
    )
    parser.add_argument(
        '--outfile',
        help='location of outfile',
        default='deepfashion_filelist.txt',
        type=str
    )
    return parser.parse_args()

def main(args):
    filelist = []
    filecount = 0
    max_multiview = 0

    for root, dirs, files in os.walk(args.deepfashion):
        for file in files:
            if file.endswith("IUV.png"): # only look for valid files
                item_name = '_'.join(os.path.splitext(file)[0].split('_')[:-1])
                item_path = os.path.join(root, item_name)

                filelist.append({})
                filelist[filecount]['path'] = item_path

                filelist[filecount]['views'] = []

                view_id = item_name.split('_')[0] # unique ID for person + outfit

                for other_view in files:
                    if other_view != file: # not current file
                        if other_view.endswith("IUV.png"):
                            this_view_id = os.path.splitext(other_view)[0].split('_')[0]

                            if this_view_id == view_id: # different view of same person and outfit
                                other_view_name = '_'.join(os.path.splitext(other_view)[0].split('_')[:-1])
                                other_view_path = os.path.join(root, other_view_name)
                                filelist[filecount]['views'].append(other_view_path)

                                if len(filelist[filecount]['views']) > max_multiview:
                                    # print('new_max: {:d}'.format(filecount))
                                    max_multiview = len(filelist[filecount]['views'])

                filecount = filecount + 1

    filedict = {'max_multiview': max_multiview,
                'filelist': filelist}

    with open(args.outfile, "wb") as fp:
        pickle.dump(filedict, fp)

if __name__ == '__main__':
    args = parse_args()
    main(args)
