# this code runs only with py2
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import pandas as pd
import numpy.random as npr                                                          
import pmf
from scipy import sparse, stats
from utils import exp_to_imp
import argparse

import time
import random
# randseed = int(time.time()*1000000%100000000)
randseed = 26499506
print("random seed: ", randseed)
random.seed(randseed)
np.random.seed(randseed)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-ddir', '--datadir', type=str, \
        default='NA')
    parser.add_argument('-odir', '--outdatadir', \
        type=str, default='NA')
    args = parser.parse_args()


    print("setting params")

    DATA_DIR = args.datadir
    OUT_DATA_DIR = args.outdatadir

    if not os.path.exists(OUT_DATA_DIR):
        os.makedirs(OUT_DATA_DIR)
        
    unique_uid = list()
    with open(os.path.join(DATA_DIR, 'unique_uid.txt'), 'r') as f:
        for line in f:
            unique_uid.append(line.strip())
        
    unique_sid = list()
    with open(os.path.join(DATA_DIR, 'unique_sid.txt'), 'r') as f:
        for line in f:
            unique_sid.append(line.strip())

    n_users = len(unique_uid)
    n_items = len(unique_sid)

    def load_data(csv_file, shape=(n_users, n_items)):
        tp = pd.read_csv(csv_file)
        rows, cols, vals = np.array(tp['uid']), np.array(tp['sid']), np.array(tp['rating']) 
        data = sparse.csr_matrix((vals, (rows, cols)), dtype=np.float32, shape=shape)
        return data


    train_data = load_data(os.path.join(DATA_DIR, 'train.csv'))
    vad_data = load_data(os.path.join(DATA_DIR, 'validation.csv'))
    test_data = load_data(os.path.join(DATA_DIR, 'test_full.csv'))

    train_data_imp = exp_to_imp(train_data, 0.5)
    vad_data_imp = exp_to_imp(vad_data, 0.5)

    dims = np.array([1,2,5,10,20,50,100])
     
    train_data_coo = train_data_imp.tocoo()
    row_tr, col_tr = train_data_coo.row, train_data_coo.col

    vad_data_coo = vad_data_imp.tocoo()
    row_vd, col_vd = vad_data_coo.row, vad_data_coo.col

    for i, dim in enumerate(dims):
        print("dim", dim)
        pf = pmf.PoissonMF(n_components=dim, random_state=98765, verbose=True, a=0.3, b=0.3, c=0.3, d=0.3)
        pf.fit(train_data, row_tr, col_tr, dict(X_new=vad_data.data, rows_new=row_vd, cols_new=col_vd))
        U, V = pf.Eb, pf.Et.T
        np.savetxt(OUT_DATA_DIR + '/cause_pmf_k'+str(dim)+'_U.csv', U)
        np.savetxt(OUT_DATA_DIR + '/cause_pmf_k'+str(dim)+'_V.csv', V)


