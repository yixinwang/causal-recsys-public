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

    unique_uid = list()
    DATA_DIR = args.datadir
    OUT_DATA_DIR = args.outdatadir

    if not os.path.exists(OUT_DATA_DIR):
        os.makedirs(OUT_DATA_DIR)

    with open(os.path.join(DATA_DIR, 'unique_uid.txt'), 'r') as f:
        for line in f:
            unique_uid.append(line.strip())
        
    unique_sid = list()
    with open(os.path.join(DATA_DIR, 'unique_sid.txt'), 'r') as f:
        for line in f:
            unique_sid.append(line.strip())

    n_users = len(unique_uid)
    n_items = len(unique_sid)

    def load_train_data(csv_file):
        tp = pd.read_csv(csv_file)
        n_users = tp['uid'].max() + 1

        ratings, rows, cols = tp['rating'], tp['uid'], tp['sid']
        data = sparse.csr_matrix((ratings,
                                 (rows, cols)), dtype='float64',
                                 shape=(n_users, n_items))
        return data

    def load_tr_te_data(csv_file_tr, csv_file_te):
        tp_tr = pd.read_csv(csv_file_tr)
        tp_te = pd.read_csv(csv_file_te)

        start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
        end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

        ratings_tr, rows_tr, cols_tr = tp_tr['rating'], tp_tr['uid'] - start_idx, tp_tr['sid']
        ratings_te, rows_te, cols_te = tp_te['rating'], tp_te['uid'] - start_idx, tp_te['sid']

        data_tr = sparse.csr_matrix((ratings_tr,
                                 (rows_tr, cols_tr)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
        data_te = sparse.csr_matrix((ratings_te,
                                 (rows_te, cols_te)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
        return data_tr, data_te

    train_data = load_train_data(os.path.join(DATA_DIR, 'train.csv')).tocsr()
    vad_data_tr, vad_data_te = load_tr_te_data(os.path.join(DATA_DIR, 'validation_tr.csv'), 
                                          os.path.join(DATA_DIR, 'validation_te.csv'))
    test_data_tr, test_data_te = load_tr_te_data(os.path.join(DATA_DIR, 'test_tr.csv'), 
                                              os.path.join(DATA_DIR, 'test_te.csv'))

    train_data = exp_to_imp(train_data, 0.5)
    vad_data_tr, vad_data_te = exp_to_imp(vad_data_tr, 0.5), exp_to_imp(vad_data_tr, 0.5)
    test_data_tr, test_data_te = exp_to_imp(test_data_tr, 0.5), exp_to_imp(test_data_tr, 0.5)

    dims = np.array([1,2,5,10,20,50,100])
     
    train_data_coo = train_data.tocoo()
    vad_data_tr_coo = vad_data_tr.tocoo()
    test_data_tr_coo = test_data_tr.tocoo()


    for i, dim in enumerate(dims):
        print("dim", dim)
        pf = pmf.PoissonMF(n_components=dim, random_state=98765, verbose=True, a=0.3, b=0.3, c=0.3, d=0.3)
        pf.fit(train_data_coo, train_data_coo.row, train_data_coo.col)
        U, V = pf.Eb.copy(), pf.Et.T
        np.savetxt(OUT_DATA_DIR + '/cause_pmf_k'+str(dim)+'_trainU.csv', U)
        np.savetxt(OUT_DATA_DIR + '/cause_pmf_k'+str(dim)+'_V.csv', V)
        U = pf.transform(vad_data_tr_coo, vad_data_tr_coo.row, vad_data_tr_coo.col).copy()
        np.savetxt(OUT_DATA_DIR + '/cause_pmf_k'+str(dim)+'_vadU.csv', U)
        U = pf.transform(test_data_tr_coo, test_data_tr_coo.row, test_data_tr_coo.col).copy()
        np.savetxt(OUT_DATA_DIR + '/cause_pmf_k'+str(dim)+'_testU.csv', U)




