import numpy as np
import pandas as pd
import argparse

import os
import sys
cwd = os.getcwd()
parent_dir = os.path.abspath(os.path.join(cwd, os.pardir))
sys.path.insert(0, os.path.join(parent_dir, 'pysta2'))

import stcl


if __name__ == '__main__':
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default='20201209', help="dataset")
    parser.add_argument("-t", "--tap", type=int, default=12, help="number of taps")
    # clustering param
    # parser.add_argument("-d", "--dim", type=int, default=2, help="dimension used for clustering")

    # read arguments from the command line
    args = parser.parse_args()

    print('dataset is {}.'.format(args.dataset))
    print("number of tap is {}.".format(args.tap))
    # print("dimension for clustering is is {}.".format(args.dim))

    # load experimental info
    print('loading experimental info')
    data_path = 'data'
    info = pd.read_csv(os.path.join(data_path, args.dataset + '_info.csv'))

    # load data
    print("loading data...")
    # load stim and spike data

    data = np.load(os.path.join(data_path, args.dataset + '.npz'))

    stim = data['stim']
    if stim.shape[0] > stim.shape[1]: # 0st dim should be spatial, 1st dim should be time
        stim = stim.T

    spike_counts = data['spike_counts']

    save_folder_name = os.path.join('results', args.dataset)
    if not os.path.exists(save_folder_name):
        os.makedirs(save_folder_name)

    print("running STCL...")
    stcl.run(stim, spike_counts, info['channel'].astype('str').to_list(), tap=args.tap, results_path=save_folder_name)

