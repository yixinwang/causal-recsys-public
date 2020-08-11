import argparse
import numpy as np
import pandas as pd
import os
import sys

from rec_eval import normalized_dcg_nonbinary, recall_at_k, \
map_at_k, prec_at_k, ric_rank, mean_perc_rank

from rec_eval_xpred import log_cond_poisson_prob_metrics, \
normalized_dcg_nonbinary_xpred, recall_at_k_xpred, map_at_k_xpred,\
prec_at_k_xpred, ric_rank_xpred, mean_perc_rank_xpred, \
normalized_dcg_at_k_nonbinary_xpred, log_cond_normal_prob_metrics, mse, mae, avg_col_mse, avg_col_mae


def binarize_rating(data, cutoff=3, eps=1e-6):
    data.data[data.data <= cutoff] = eps   
    data.data[data.data > cutoff] = 1
    return data


def exp_to_imp(data, cutoff=1e-10):
    data_imp = data.copy()
    data_imp.data[data_imp.data < cutoff] = 0
    data_imp.data[data_imp.data >= cutoff] = 1
    data_imp.data = data_imp.data.astype('int32')
    data_imp.eliminate_zeros()
    return data_imp

def binarize_spmat(spmat):
    spmat_binary = spmat.copy()
    spmat_binary.data = np.ones_like(spmat_binary.data)
    return spmat_binary

def subsample_negatives(data, full_data=None, random_state=0, verbose=False):
    
    n_users, n_items = data.shape
    
    if full_data is None:
        full_data = data

    rows_neg, cols_neg = [], []

    np.random.seed(random_state)

    for u in xrange(n_users):
        p = np.ones(n_items, dtype='float32')
        p[full_data[u].nonzero()[1]] = 0
        p /= p.sum()

        neg_items = np.random.choice(n_items, size=data[u].nnz, replace=False, p=p)

        rows_neg.append([u] * data[u].nnz)
        cols_neg.append(neg_items)

        if verbose and u % 5000 == 0:
            print("%d users sampled" % u)
            sys.stdout.flush()

    rows_neg = np.hstack(rows_neg)
    cols_neg = np.hstack(cols_neg)

    return rows_neg, cols_neg


def next_batch(x_train, M):
    idx_batch = np.random.choice(x_train.shape[0],M)
    return x_train[idx_batch,:], idx_batch

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ddir', '--datadir', type=str, \
        default='Webscope_R3')
    parser.add_argument('-cdir', '--causedir', type=str, \
        default='R3_out')
    parser.add_argument('-odir', '--outdatadir', \
        type=str, default='R3_edout')
    parser.add_argument('-odim', '--outdim', \
        type=int, default=5)
    parser.add_argument('-cdim', '--caudim', \
        type=int, default=5)
    parser.add_argument('-th', '--thold', \
        type=float, default=3) # only strictly larger than thold is taken as relevant
    parser.add_argument('-M', '--M', \
        type=int, default=100) #batch size
    parser.add_argument('-nitr', '--n_iter', \
        type=int, default=10)
    parser.add_argument('-pU', '--priorU', \
        type=int, default=1) # it is the inverse of real prior sd for U
    parser.add_argument('-pV', '--priorV', \
        type=int, default=1) # it is the inverse of real prior sd for V
    parser.add_argument('-alpha', '--alpha', \
        type=int, default=40) # alpha is the alpha for wmf
    parser.add_argument('-binary', '--binary', \
        type=int, default=0)

    return parser


def set_params(args):
    DATA_DIR = args.datadir
    CAUSEFIT_DIR = args.causedir
    OUT_DATA_DIR = args.outdatadir
    outdim = args.outdim
    caudim = args.caudim
    thold = args.thold # only count ratings > thold as relevant in recall and map
    M = args.M  #batch size
    n_iter = args.n_iter
    binary = args.binary
    pri_U = 1./args.priorU
    pri_V = 1./args.priorV
    alpha = args.alpha # alpha 0 is vanilla gmf

    if not os.path.exists(OUT_DATA_DIR):
        os.makedirs(OUT_DATA_DIR)

    return DATA_DIR, CAUSEFIT_DIR, OUT_DATA_DIR, \
        outdim, caudim, thold, M, n_iter, binary, \
        pri_U, pri_V, alpha

def load_prefit_pfcau(CAUSEFIT_DIR, dim):
    U = np.loadtxt(CAUSEFIT_DIR + '/cause_pmf_k'+str(dim)+'_U.csv')
    V = np.loadtxt(CAUSEFIT_DIR + '/cause_pmf_k'+str(dim)+'_V.csv')
    U = (np.atleast_2d(U.T).T)
    V = (np.atleast_2d(V.T).T)
    reconstr_cau = U.dot(V.T)
    return U, V, reconstr_cau

def create_metric_holders(outdims, ks):
    train_ndcg, vad_ndcg, test_ndcg = \
        np.zeros(len(outdims)), np.zeros(len(outdims)), np.zeros(len(outdims))
    train_poisson_pos_plp, vad_poisson_pos_plp, test_poisson_pos_plp = \
        np.zeros(len(outdims)), np.zeros(len(outdims)), np.zeros(len(outdims))
    train_poisson_neg_plp, vad_poisson_neg_plp, test_poisson_neg_plp = \
        np.zeros(len(outdims)), np.zeros(len(outdims)), np.zeros(len(outdims))
    train_poisson_all_plp, vad_poisson_all_plp, test_poisson_all_plp = \
        np.zeros(len(outdims)), np.zeros(len(outdims)), np.zeros(len(outdims))
    train_normal_pos_plp, vad_normal_pos_plp, test_normal_pos_plp = \
        np.zeros(len(outdims)), np.zeros(len(outdims)), np.zeros(len(outdims))
    train_normal_neg_plp, vad_normal_neg_plp, test_normal_neg_plp = \
        np.zeros(len(outdims)), np.zeros(len(outdims)), np.zeros(len(outdims))
    train_normal_all_plp, vad_normal_all_plp, test_normal_all_plp = \
        np.zeros(len(outdims)), np.zeros(len(outdims)), np.zeros(len(outdims))
    train_mse_pos, vad_mse_pos, test_mse_pos = \
        np.zeros(len(outdims)), np.zeros(len(outdims)), np.zeros(len(outdims))
    train_mse_neg, vad_mse_neg, test_mse_neg = \
        np.zeros(len(outdims)), np.zeros(len(outdims)), np.zeros(len(outdims))
    train_mse_all, vad_mse_all, test_mse_all = \
        np.zeros(len(outdims)), np.zeros(len(outdims)), np.zeros(len(outdims))
    train_mae_pos, vad_mae_pos, test_mae_pos = \
        np.zeros(len(outdims)), np.zeros(len(outdims)), np.zeros(len(outdims))
    train_mae_neg, vad_mae_neg, test_mae_neg = \
        np.zeros(len(outdims)), np.zeros(len(outdims)), np.zeros(len(outdims))
    train_mae_all, vad_mae_all, test_mae_all = \
        np.zeros(len(outdims)), np.zeros(len(outdims)), np.zeros(len(outdims))
    train_avg_col_mse_pos, vad_avg_col_mse_pos, test_avg_col_mse_pos = \
        np.zeros(len(outdims)), np.zeros(len(outdims)), np.zeros(len(outdims))
    train_avg_col_mse_neg, vad_avg_col_mse_neg, test_avg_col_mse_neg = \
        np.zeros(len(outdims)), np.zeros(len(outdims)), np.zeros(len(outdims))
    train_avg_col_mse_all, vad_avg_col_mse_all, test_avg_col_mse_all = \
        np.zeros(len(outdims)), np.zeros(len(outdims)), np.zeros(len(outdims))
    train_avg_col_mae_pos, vad_avg_col_mae_pos, test_avg_col_mae_pos = \
        np.zeros(len(outdims)), np.zeros(len(outdims)), np.zeros(len(outdims))
    train_avg_col_mae_neg, vad_avg_col_mae_neg, test_avg_col_mae_neg = \
        np.zeros(len(outdims)), np.zeros(len(outdims)), np.zeros(len(outdims))
    train_avg_col_mae_all, vad_avg_col_mae_all, test_avg_col_mae_all = \
        np.zeros(len(outdims)), np.zeros(len(outdims)), np.zeros(len(outdims))
    test_mae, test_mse = \
        np.zeros(len(outdims)), np.zeros(len(outdims))
    train_ndcgk, vad_ndcgk, test_ndcgk = \
        np.zeros((len(outdims), len(ks))), \
        np.zeros((len(outdims), len(ks))), \
        np.zeros((len(outdims), len(ks)))
    train_recallk, vad_recallk, test_recallk = \
        np.zeros((len(outdims), len(ks))), \
        np.zeros((len(outdims), len(ks))), \
        np.zeros((len(outdims), len(ks)))
    train_mapk, vad_mapk, test_mapk = \
        np.zeros((len(outdims), len(ks))), \
        np.zeros((len(outdims), len(ks))), \
        np.zeros((len(outdims), len(ks)))
    train_preck, vad_preck, test_preck = \
        np.zeros((len(outdims), len(ks))), \
        np.zeros((len(outdims), len(ks))), \
        np.zeros((len(outdims), len(ks)))
    train_ric_rank, vad_ric_rank, test_ric_rank = \
        np.zeros(len(outdims)), np.zeros(len(outdims)), np.zeros(len(outdims))
    train_mean_perc_rank, vad_mean_perc_rank, test_mean_perc_rank = \
        np.zeros(len(outdims)), np.zeros(len(outdims)), np.zeros(len(outdims))

    return train_ndcg, vad_ndcg, test_ndcg, \
        train_mse_pos, vad_mse_pos, test_mse_pos, \
        train_mse_neg, vad_mse_neg, test_mse_neg, \
        train_mse_all, vad_mse_all, test_mse_all, \
        train_mae_pos, vad_mae_pos, test_mae_pos, \
        train_mae_neg, vad_mae_neg, test_mae_neg, \
        train_mae_all, vad_mae_all, test_mae_all, \
        train_avg_col_mse_pos, vad_avg_col_mse_pos, test_avg_col_mse_pos, \
        train_avg_col_mse_neg, vad_avg_col_mse_neg, test_avg_col_mse_neg, \
        train_avg_col_mse_all, vad_avg_col_mse_all, test_avg_col_mse_all, \
        train_avg_col_mae_pos, vad_avg_col_mae_pos, test_avg_col_mae_pos, \
        train_avg_col_mae_neg, vad_avg_col_mae_neg, test_avg_col_mae_neg, \
        train_avg_col_mae_all, vad_avg_col_mae_all, test_avg_col_mae_all, \
        train_poisson_pos_plp, vad_poisson_pos_plp, test_poisson_pos_plp, \
        train_poisson_neg_plp, vad_poisson_neg_plp, test_poisson_neg_plp, \
        train_poisson_all_plp, vad_poisson_all_plp, test_poisson_all_plp, \
        train_normal_pos_plp, vad_normal_pos_plp, test_normal_pos_plp, \
        train_normal_neg_plp, vad_normal_neg_plp, test_normal_neg_plp, \
        train_normal_all_plp, vad_normal_all_plp, test_normal_all_plp, \
        train_ndcgk, vad_ndcgk, test_ndcgk, \
        train_recallk, vad_recallk, test_recallk, \
        train_mapk, vad_mapk, test_mapk, \
        train_preck, vad_preck, test_preck, \
        train_ric_rank, vad_ric_rank, test_ric_rank, \
        train_mean_perc_rank, vad_mean_perc_rank, test_mean_perc_rank, \
        test_mae, test_mse


def wg_eval_acc_metrics_update_i(all_metric_holders, i, pred, train_data, \
    vad_data, test_data, ks, thold):
        # pred must be sparse matrix
    train_ndcg, vad_ndcg, test_ndcg, \
    train_mse_pos, vad_mse_pos, test_mse_pos, \
    train_mse_neg, vad_mse_neg, test_mse_neg, \
    train_mse_all, vad_mse_all, test_mse_all, \
    train_mae_pos, vad_mae_pos, test_mae_pos, \
    train_mae_neg, vad_mae_neg, test_mae_neg, \
    train_mae_all, vad_mae_all, test_mae_all, \
    train_avg_col_mse_pos, vad_avg_col_mse_pos, test_avg_col_mse_pos, \
    train_avg_col_mse_neg, vad_avg_col_mse_neg, test_avg_col_mse_neg, \
    train_avg_col_mse_all, vad_avg_col_mse_all, test_avg_col_mse_all, \
    train_avg_col_mae_pos, vad_avg_col_mae_pos, test_avg_col_mae_pos, \
    train_avg_col_mae_neg, vad_avg_col_mae_neg, test_avg_col_mae_neg, \
    train_avg_col_mae_all, vad_avg_col_mae_all, test_avg_col_mae_all, \
    train_poisson_pos_plp, vad_poisson_pos_plp, test_poisson_pos_plp, \
    train_poisson_neg_plp, vad_poisson_neg_plp, test_poisson_neg_plp, \
    train_poisson_all_plp, vad_poisson_all_plp, test_poisson_all_plp, \
    train_normal_pos_plp, vad_normal_pos_plp, test_normal_pos_plp, \
    train_normal_neg_plp, vad_normal_neg_plp, test_normal_neg_plp, \
    train_normal_all_plp, vad_normal_all_plp, test_normal_all_plp, \
    train_ndcgk, vad_ndcgk, test_ndcgk, \
    train_recallk, vad_recallk, test_recallk, \
    train_mapk, vad_mapk, test_mapk, \
    train_preck, vad_preck, test_preck, \
    train_ric_rank, vad_ric_rank, test_ric_rank, \
    train_mean_perc_rank, vad_mean_perc_rank, test_mean_perc_rank, \
    test_mae, test_mse = all_metric_holders

    train_mse_pos[i], train_mse_neg[i], train_mse_all[i] = \
        mse(train_data.todense(), pred)

    vad_mse_pos[i], vad_mse_neg[i], vad_mse_all[i] = \
        mse(vad_data.todense(), pred)

    test_mse_pos[i], test_mse_neg[i], test_mse_all[i] = \
        mse(test_data.todense(), pred)

    train_mae_pos[i], train_mae_neg[i], train_mae_all[i] = \
        mae(train_data.todense(), pred)

    vad_mae_pos[i], vad_mae_neg[i], vad_mae_all[i] = \
        mae(vad_data.todense(), pred)

    test_mae_pos[i], test_mae_neg[i], test_mae_all[i] = \
        mae(test_data.todense(), pred)

    train_avg_col_mse_pos[i], train_avg_col_mse_neg[i], train_avg_col_mse_all[i] = \
        avg_col_mse(train_data.todense(), pred)

    vad_avg_col_mse_pos[i], vad_avg_col_mse_neg[i], vad_avg_col_mse_all[i] = \
        avg_col_mse(vad_data.todense(), pred)

    test_avg_col_mse_pos[i], test_avg_col_mse_neg[i], test_avg_col_mse_all[i] = \
        avg_col_mse(test_data.todense(), pred)     

    train_avg_col_mae_pos[i], train_avg_col_mae_neg[i], train_avg_col_mae_all[i] = \
        avg_col_mae(train_data.todense(), pred)

    vad_avg_col_mae_pos[i], vad_avg_col_mae_neg[i], vad_avg_col_mae_all[i] = \
        avg_col_mae(vad_data.todense(), pred)

    test_avg_col_mae_pos[i], test_avg_col_mae_neg[i], test_avg_col_mae_all[i] = \
        avg_col_mae(test_data.todense(), pred)


    print("train mse pos neg all", train_mse_pos[i], train_mse_neg[i], train_mse_all[i])
    print("vad mse pos neg all", vad_mse_pos[i], vad_mse_neg[i], vad_mse_all[i])
    print("test mse pos neg all", test_mse_pos[i], test_mse_neg[i], test_mse_all[i])

    print("train mae pos neg all", train_mae_pos[i], train_mae_neg[i], train_mae_all[i])
    print("vad mae pos neg all", vad_mae_pos[i], vad_mae_neg[i], vad_mae_all[i])
    print("test mae pos neg all", test_mae_pos[i], test_mae_neg[i], test_mae_all[i])

    print("train avg_col_mse pos neg all", train_avg_col_mse_pos[i], train_avg_col_mse_neg[i], train_avg_col_mse_all[i])
    print("vad avg_col_mse pos neg all", vad_avg_col_mse_pos[i], vad_avg_col_mse_neg[i], vad_avg_col_mse_all[i])
    print("test avg_col_mse pos neg all", test_avg_col_mse_pos[i], test_avg_col_mse_neg[i], test_avg_col_mse_all[i])

    print("train avg_col_mae pos neg all", train_avg_col_mae_pos[i], train_avg_col_mae_neg[i], train_avg_col_mae_all[i])
    print("vad avg_col_mae pos neg all", vad_avg_col_mae_pos[i], vad_avg_col_mae_neg[i], vad_avg_col_mae_all[i])
    print("test avg_col_mae pos neg all", test_avg_col_mae_pos[i], test_avg_col_mae_neg[i], test_avg_col_mae_all[i])

    train_poisson_pos_plp[i], train_poisson_neg_plp[i], train_poisson_all_plp[i] = \
        log_cond_poisson_prob_metrics(train_data.todense(), pred)

    vad_poisson_pos_plp[i], vad_poisson_neg_plp[i], vad_poisson_all_plp[i] = \
        log_cond_poisson_prob_metrics(vad_data.todense(), pred)

    test_poisson_pos_plp[i], test_poisson_neg_plp[i], test_poisson_all_plp[i] = \
        log_cond_poisson_prob_metrics(test_data.todense(), pred)

    train_normal_pos_plp[i], train_normal_neg_plp[i], train_normal_all_plp[i] = \
        log_cond_normal_prob_metrics(train_data.todense(), pred)

    vad_normal_pos_plp[i], vad_normal_neg_plp[i], vad_normal_all_plp[i] = \
        log_cond_normal_prob_metrics(vad_data.todense(), pred)

    test_normal_pos_plp[i], test_normal_neg_plp[i], test_normal_all_plp[i] = \
        log_cond_normal_prob_metrics(test_data.todense(), pred)

    print("train poisson pos neg all", train_poisson_pos_plp[i], train_poisson_neg_plp[i], train_poisson_all_plp[i])
    print("vad poisson pos neg all", vad_poisson_pos_plp[i], vad_poisson_neg_plp[i], vad_poisson_all_plp[i])
    print("test poisson pos neg all", test_poisson_pos_plp[i], test_poisson_neg_plp[i], test_poisson_all_plp[i])

    print("train normal pos neg all", train_normal_pos_plp[i], train_normal_neg_plp[i], train_normal_all_plp[i])
    print("vad normal pos neg all", vad_normal_pos_plp[i], vad_normal_neg_plp[i], vad_normal_all_plp[i])
    print("test normal pos neg all", test_normal_pos_plp[i], test_normal_neg_plp[i], test_normal_all_plp[i])

    
    
    train_ndcg[i], vad_ndcg[i], test_ndcg[i] = \
        normalized_dcg_nonbinary_xpred(vad_data, train_data, pred), \
        normalized_dcg_nonbinary_xpred(train_data, vad_data, pred), \
        normalized_dcg_nonbinary_xpred(train_data, test_data, pred, vad_data = vad_data)
    
    train_ndcgk[i], vad_ndcgk[i], test_ndcgk[i] =\
        [normalized_dcg_at_k_nonbinary_xpred(vad_data, train_data, pred, k=k) for k in ks], \
        [normalized_dcg_at_k_nonbinary_xpred(train_data, vad_data, pred, k=k) for k in ks], \
        [normalized_dcg_at_k_nonbinary_xpred(train_data, test_data, pred, k=k, vad_data = vad_data) for k in ks]


    train_recallk[i], vad_recallk[i], test_recallk[i] = \
        [recall_at_k_xpred(vad_data, train_data, pred, k=k, thold=thold) for k in ks], \
        [recall_at_k_xpred(train_data, vad_data, pred, k=k, thold=thold) for k in ks], \
        [recall_at_k_xpred(train_data, test_data, pred, vad_data = vad_data, k=k, thold=thold) for k in ks]

    train_mapk[i], vad_mapk[i], test_mapk[i] = \
        [map_at_k_xpred(vad_data, train_data, pred, k=k, thold=thold) for k in ks], \
        [map_at_k_xpred(train_data, vad_data, pred, k=k, thold=thold) for k in ks], \
        [map_at_k_xpred(train_data, test_data, pred, vad_data = vad_data, k=k, thold=thold) for k in ks]

    train_preck[i], vad_preck[i], test_preck[i] = \
        [prec_at_k_xpred(vad_data, train_data, pred, k=k, thold=thold) for k in ks], \
        [prec_at_k_xpred(train_data, vad_data, pred, k=k, thold=thold) for k in ks], \
        [prec_at_k_xpred(train_data, test_data, pred, vad_data = vad_data, k=k, thold=thold) for k in ks]

    train_ric_rank[i], vad_ric_rank[i], test_ric_rank[i] = \
        ric_rank_xpred(vad_data, train_data, pred, thold=thold), \
        ric_rank_xpred(train_data, vad_data, pred, thold=thold), \
        ric_rank_xpred(train_data, test_data, pred, vad_data = vad_data, thold=thold)
    
    train_mean_perc_rank[i], vad_mean_perc_rank[i], test_mean_perc_rank[i] = \
        mean_perc_rank_xpred(vad_data, train_data, pred), \
        mean_perc_rank_xpred(train_data, vad_data, pred), \
        mean_perc_rank_xpred(train_data, test_data, pred, vad_data = vad_data)
    
    print("ndcg train vad test", train_ndcg[i], vad_ndcg[i], test_ndcg[i])
    
    print("recall train vad test", train_recallk[i], vad_recallk[i], test_recallk[i])
    
    print("map train vad test", train_mapk[i], vad_mapk[i], test_mapk[i])            

    print("prec train vad test", train_preck[i], vad_preck[i], test_preck[i])

    print("ric_rank train vad test", train_ric_rank[i], vad_ric_rank[i], test_ric_rank[i])
    
    print("mean_perc_rank train vad test", train_mean_perc_rank[i], vad_mean_perc_rank[i], test_mean_perc_rank[i])
        
    test_mse[i] = np.mean(np.square(test_data[test_data.nonzero()] - pred[test_data.nonzero()]))
    test_mae[i] = np.mean(np.abs(test_data[test_data.nonzero()] - pred[test_data.nonzero()]))
  
    print('mse', test_mse[i])
    print('mae', test_mae[i])


    return train_ndcg, vad_ndcg, test_ndcg, \
        train_mse_pos, vad_mse_pos, test_mse_pos, \
        train_mse_neg, vad_mse_neg, test_mse_neg, \
        train_mse_all, vad_mse_all, test_mse_all, \
        train_mae_pos, vad_mae_pos, test_mae_pos, \
        train_mae_neg, vad_mae_neg, test_mae_neg, \
        train_mae_all, vad_mae_all, test_mae_all, \
        train_avg_col_mse_pos, vad_avg_col_mse_pos, test_avg_col_mse_pos, \
        train_avg_col_mse_neg, vad_avg_col_mse_neg, test_avg_col_mse_neg, \
        train_avg_col_mse_all, vad_avg_col_mse_all, test_avg_col_mse_all, \
        train_avg_col_mae_pos, vad_avg_col_mae_pos, test_avg_col_mae_pos, \
        train_avg_col_mae_neg, vad_avg_col_mae_neg, test_avg_col_mae_neg, \
        train_avg_col_mae_all, vad_avg_col_mae_all, test_avg_col_mae_all, \
        train_poisson_pos_plp, vad_poisson_pos_plp, test_poisson_pos_plp, \
        train_poisson_neg_plp, vad_poisson_neg_plp, test_poisson_neg_plp, \
        train_poisson_all_plp, vad_poisson_all_plp, test_poisson_all_plp, \
        train_normal_pos_plp, vad_normal_pos_plp, test_normal_pos_plp, \
        train_normal_neg_plp, vad_normal_neg_plp, test_normal_neg_plp, \
        train_normal_all_plp, vad_normal_all_plp, test_normal_all_plp, \
        train_ndcgk, vad_ndcgk, test_ndcgk, \
        train_recallk, vad_recallk, test_recallk, \
        train_mapk, vad_mapk, test_mapk, \
        train_preck, vad_preck, test_preck, \
        train_ric_rank, vad_ric_rank, test_ric_rank, \
        train_mean_perc_rank, vad_mean_perc_rank, test_mean_perc_rank, \
        test_mae, test_mse


def sg_eval_acc_metrics_update_i(all_metric_holders, i, \
    pred_train, pred_vad, pred_test, \
    train_data, \
    vad_data_tr, vad_data_te, \
    test_data_tr, test_data_te, \
    ks, thold):
        # pred must be sparse matrix

    train_ndcg, vad_ndcg, test_ndcg, \
    train_mse_pos, vad_mse_pos, test_mse_pos, \
    train_mse_neg, vad_mse_neg, test_mse_neg, \
    train_mse_all, vad_mse_all, test_mse_all, \
    train_mae_pos, vad_mae_pos, test_mae_pos, \
    train_mae_neg, vad_mae_neg, test_mae_neg, \
    train_mae_all, vad_mae_all, test_mae_all, \
    train_avg_col_mse_pos, vad_avg_col_mse_pos, test_avg_col_mse_pos, \
    train_avg_col_mse_neg, vad_avg_col_mse_neg, test_avg_col_mse_neg, \
    train_avg_col_mse_all, vad_avg_col_mse_all, test_avg_col_mse_all, \
    train_avg_col_mae_pos, vad_avg_col_mae_pos, test_avg_col_mae_pos, \
    train_avg_col_mae_neg, vad_avg_col_mae_neg, test_avg_col_mae_neg, \
    train_avg_col_mae_all, vad_avg_col_mae_all, test_avg_col_mae_all, \
    train_poisson_pos_plp, vad_poisson_pos_plp, test_poisson_pos_plp, \
    train_poisson_neg_plp, vad_poisson_neg_plp, test_poisson_neg_plp, \
    train_poisson_all_plp, vad_poisson_all_plp, test_poisson_all_plp, \
    train_normal_pos_plp, vad_normal_pos_plp, test_normal_pos_plp, \
    train_normal_neg_plp, vad_normal_neg_plp, test_normal_neg_plp, \
    train_normal_all_plp, vad_normal_all_plp, test_normal_all_plp, \
    train_ndcgk, vad_ndcgk, test_ndcgk, \
    train_recallk, vad_recallk, test_recallk, \
    train_mapk, vad_mapk, test_mapk, \
    train_preck, vad_preck, test_preck, \
    train_ric_rank, vad_ric_rank, test_ric_rank, \
    train_mean_perc_rank, vad_mean_perc_rank, test_mean_perc_rank, \
    test_mae, test_mse = all_metric_holders


    train_mse_pos[i], train_mse_neg[i], train_mse_all[i] = \
        mse(train_data.todense(), pred_train)

    vad_mse_pos[i], vad_mse_neg[i], vad_mse_all[i] = \
        mse(vad_data_te.todense(), pred_vad)

    test_mse_pos[i], test_mse_neg[i], test_mse_all[i] = \
        mse(test_data_te.todense(), pred_test)

    train_mae_pos[i], train_mae_neg[i], train_mae_all[i] = \
        mae(train_data.todense(), pred_train)

    vad_mae_pos[i], vad_mae_neg[i], vad_mae_all[i] = \
        mae(vad_data_te.todense(), pred_vad)

    test_mae_pos[i], test_mae_neg[i], test_mae_all[i] = \
        mae(test_data_te.todense(), pred_test)

    train_avg_col_mse_pos[i], train_avg_col_mse_neg[i], train_avg_col_mse_all[i] = \
        avg_col_mse(train_data.todense(), pred_train)

    vad_avg_col_mse_pos[i], vad_avg_col_mse_neg[i], vad_avg_col_mse_all[i] = \
        avg_col_mse(vad_data_te.todense(), pred_vad)

    test_avg_col_mse_pos[i], test_avg_col_mse_neg[i], test_avg_col_mse_all[i] = \
        avg_col_mse(test_data_te.todense(), pred_test)     

    train_avg_col_mae_pos[i], train_avg_col_mae_neg[i], train_avg_col_mae_all[i] = \
        avg_col_mae(train_data.todense(), pred_train)

    vad_avg_col_mae_pos[i], vad_avg_col_mae_neg[i], vad_avg_col_mae_all[i] = \
        avg_col_mae(vad_data_te.todense(), pred_vad)

    test_avg_col_mae_pos[i], test_avg_col_mae_neg[i], test_avg_col_mae_all[i] = \
        avg_col_mae(test_data_te.todense(), pred_test)


    print("train mse pos neg all", train_mse_pos[i], train_mse_neg[i], train_mse_all[i])
    print("vad mse pos neg all", vad_mse_pos[i], vad_mse_neg[i], vad_mse_all[i])
    print("test mse pos neg all", test_mse_pos[i], test_mse_neg[i], test_mse_all[i])

    print("train mae pos neg all", train_mae_pos[i], train_mae_neg[i], train_mae_all[i])
    print("vad mae pos neg all", vad_mae_pos[i], vad_mae_neg[i], vad_mae_all[i])
    print("test mae pos neg all", test_mae_pos[i], test_mae_neg[i], test_mae_all[i])

    print("train avg_col_mse pos neg all", train_avg_col_mse_pos[i], train_avg_col_mse_neg[i], train_avg_col_mse_all[i])
    print("vad avg_col_mse pos neg all", vad_avg_col_mse_pos[i], vad_avg_col_mse_neg[i], vad_avg_col_mse_all[i])
    print("test avg_col_mse pos neg all", test_avg_col_mse_pos[i], test_avg_col_mse_neg[i], test_avg_col_mse_all[i])

    print("train avg_col_mae pos neg all", train_avg_col_mae_pos[i], train_avg_col_mae_neg[i], train_avg_col_mae_all[i])
    print("vad avg_col_mae pos neg all", vad_avg_col_mae_pos[i], vad_avg_col_mae_neg[i], vad_avg_col_mae_all[i])
    print("test avg_col_mae pos neg all", test_avg_col_mae_pos[i], test_avg_col_mae_neg[i], test_avg_col_mae_all[i])

    train_poisson_pos_plp[i], train_poisson_neg_plp[i], train_poisson_all_plp[i] = \
        log_cond_poisson_prob_metrics(train_data.todense(), pred_train)

    vad_poisson_pos_plp[i], vad_poisson_neg_plp[i], vad_poisson_all_plp[i] = \
        log_cond_poisson_prob_metrics(vad_data_te.todense(), pred_vad)

    test_poisson_pos_plp[i], test_poisson_neg_plp[i], test_poisson_all_plp[i] = \
        log_cond_poisson_prob_metrics(test_data_te.todense(), pred_test)

    train_normal_pos_plp[i], train_normal_neg_plp[i], train_normal_all_plp[i] = \
        log_cond_normal_prob_metrics(train_data.todense(), pred_train)

    vad_normal_pos_plp[i], vad_normal_neg_plp[i], vad_normal_all_plp[i] = \
        log_cond_normal_prob_metrics(vad_data_te.todense(), pred_vad)

    test_normal_pos_plp[i], test_normal_neg_plp[i], test_normal_all_plp[i] = \
        log_cond_normal_prob_metrics(test_data_te.todense(), pred_test)


    print("train poisson pos neg all", train_poisson_pos_plp[i], train_poisson_neg_plp[i], train_poisson_all_plp[i])
    print("vad poisson pos neg all", vad_poisson_pos_plp[i], vad_poisson_neg_plp[i], vad_poisson_all_plp[i])
    print("test poisson pos neg all", test_poisson_pos_plp[i], test_poisson_neg_plp[i], test_poisson_all_plp[i])

    print("train normal pos neg all", train_normal_pos_plp[i], train_normal_neg_plp[i], train_normal_all_plp[i])
    print("vad normal pos neg all", vad_normal_pos_plp[i], vad_normal_neg_plp[i], vad_normal_all_plp[i])
    print("test normal pos neg all", test_normal_pos_plp[i], test_normal_neg_plp[i], test_normal_all_plp[i])

    
    train_ndcg[i], vad_ndcg[i], test_ndcg[i] = \
        normalized_dcg_nonbinary_xpred(vad_data_tr, train_data, pred_train), \
        normalized_dcg_nonbinary_xpred(vad_data_tr, vad_data_te, pred_vad), \
        normalized_dcg_nonbinary_xpred(test_data_tr, test_data_te, pred_test, vad_data = vad_data_tr)
    
    train_ndcgk[i], vad_ndcgk[i], test_ndcgk[i] =\
        [normalized_dcg_at_k_nonbinary_xpred(vad_data_tr, train_data, pred_train, k=k) for k in ks], \
        [normalized_dcg_at_k_nonbinary_xpred(vad_data_tr, vad_data_te, pred_vad, k=k) for k in ks], \
        [normalized_dcg_at_k_nonbinary_xpred(test_data_tr, test_data_te, pred_test, k=k, vad_data = vad_data_tr) for k in ks]


    train_recallk[i], vad_recallk[i], test_recallk[i] = \
        [recall_at_k_xpred(vad_data_tr, train_data, pred_train, k=k, thold=thold) for k in ks], \
        [recall_at_k_xpred(vad_data_tr, vad_data_te, pred_vad, k=k, thold=thold) for k in ks], \
        [recall_at_k_xpred(test_data_tr, test_data_te, pred_test, vad_data = vad_data_tr, k=k, thold=thold) for k in ks]

    train_mapk[i], vad_mapk[i], test_mapk[i] = \
        [map_at_k_xpred(vad_data_tr, train_data, pred_train, k=k, thold=thold) for k in ks], \
        [map_at_k_xpred(vad_data_tr, vad_data_te, pred_vad, k=k, thold=thold) for k in ks], \
        [map_at_k_xpred(test_data_tr, test_data_te, pred_test, vad_data = vad_data_tr, k=k, thold=thold) for k in ks]

    train_preck[i], vad_preck[i], test_preck[i] = \
        [prec_at_k_xpred(vad_data_tr, train_data, pred_train, k=k, thold=thold) for k in ks], \
        [prec_at_k_xpred(vad_data_tr, vad_data_te, pred_vad, k=k, thold=thold) for k in ks], \
        [prec_at_k_xpred(test_data_tr, test_data_te, pred_test, vad_data = vad_data_tr, k=k, thold=thold) for k in ks]

    train_ric_rank[i], vad_ric_rank[i], test_ric_rank[i] = \
        ric_rank_xpred(vad_data_tr, train_data, pred_train, thold=thold), \
        ric_rank_xpred(vad_data_tr, vad_data_te, pred_vad, thold=thold), \
        ric_rank_xpred(test_data_tr, test_data_te, pred_test, vad_data = vad_data_tr, thold=thold)
    
    train_mean_perc_rank[i], vad_mean_perc_rank[i], test_mean_perc_rank[i] = \
        mean_perc_rank_xpred(vad_data_tr, train_data, pred_train), \
        mean_perc_rank_xpred(vad_data_tr, vad_data_te, pred_vad), \
        mean_perc_rank_xpred(test_data_tr, test_data_te, pred_test, vad_data = vad_data_tr)
    
    print("ndcg train vad test", train_ndcg[i], vad_ndcg[i], test_ndcg[i])
    
    print("recall train vad test", train_recallk[i], vad_recallk[i], test_recallk[i])
    
    print("map train vad test", train_mapk[i], vad_mapk[i], test_mapk[i])            

    print("prec train vad test", train_preck[i], vad_preck[i], test_preck[i])

    print("ric_rank train vad test", train_ric_rank[i], vad_ric_rank[i], test_ric_rank[i])
    
    print("mean_perc_rank train vad test", train_mean_perc_rank[i], vad_mean_perc_rank[i], test_mean_perc_rank[i])
        
    test_mse[i] = np.mean(np.square(test_data_te[test_data_te.nonzero()] - pred_test[test_data_te.nonzero()]))
    test_mae[i] = np.mean(np.abs(test_data_te[test_data_te.nonzero()] - pred_test[test_data_te.nonzero()]))
  
    print('mse', test_mse[i])
    print('mae', test_mae[i])


    return train_ndcg, vad_ndcg, test_ndcg, \
        train_mse_pos, vad_mse_pos, test_mse_pos, \
        train_mse_neg, vad_mse_neg, test_mse_neg, \
        train_mse_all, vad_mse_all, test_mse_all, \
        train_mae_pos, vad_mae_pos, test_mae_pos, \
        train_mae_neg, vad_mae_neg, test_mae_neg, \
        train_mae_all, vad_mae_all, test_mae_all, \
        train_avg_col_mse_pos, vad_avg_col_mse_pos, test_avg_col_mse_pos, \
        train_avg_col_mse_neg, vad_avg_col_mse_neg, test_avg_col_mse_neg, \
        train_avg_col_mse_all, vad_avg_col_mse_all, test_avg_col_mse_all, \
        train_avg_col_mae_pos, vad_avg_col_mae_pos, test_avg_col_mae_pos, \
        train_avg_col_mae_neg, vad_avg_col_mae_neg, test_avg_col_mae_neg, \
        train_avg_col_mae_all, vad_avg_col_mae_all, test_avg_col_mae_all, \
        train_poisson_pos_plp, vad_poisson_pos_plp, test_poisson_pos_plp, \
        train_poisson_neg_plp, vad_poisson_neg_plp, test_poisson_neg_plp, \
        train_poisson_all_plp, vad_poisson_all_plp, test_poisson_all_plp, \
        train_normal_pos_plp, vad_normal_pos_plp, test_normal_pos_plp, \
        train_normal_neg_plp, vad_normal_neg_plp, test_normal_neg_plp, \
        train_normal_all_plp, vad_normal_all_plp, test_normal_all_plp, \
        train_ndcgk, vad_ndcgk, test_ndcgk, \
        train_recallk, vad_recallk, test_recallk, \
        train_mapk, vad_mapk, test_mapk, \
        train_preck, vad_preck, test_preck, \
        train_ric_rank, vad_ric_rank, test_ric_rank, \
        train_mean_perc_rank, vad_mean_perc_rank, test_mean_perc_rank, \
        test_mae, test_mse

def save_eval_metrics(all_metric_holders, model_name, outdims, all_params, ks):
    DATA_DIR, CAUSEFIT_DIR, OUT_DATA_DIR, \
        outdim, caudim, thold, M, n_iter, binary, \
        pri_U, pri_V, alpha = all_params


    train_ndcg, vad_ndcg, test_ndcg, \
    train_mse_pos, vad_mse_pos, test_mse_pos, \
    train_mse_neg, vad_mse_neg, test_mse_neg, \
    train_mse_all, vad_mse_all, test_mse_all, \
    train_mae_pos, vad_mae_pos, test_mae_pos, \
    train_mae_neg, vad_mae_neg, test_mae_neg, \
    train_mae_all, vad_mae_all, test_mae_all, \
    train_avg_col_mse_pos, vad_avg_col_mse_pos, test_avg_col_mse_pos, \
    train_avg_col_mse_neg, vad_avg_col_mse_neg, test_avg_col_mse_neg, \
    train_avg_col_mse_all, vad_avg_col_mse_all, test_avg_col_mse_all, \
    train_avg_col_mae_pos, vad_avg_col_mae_pos, test_avg_col_mae_pos, \
    train_avg_col_mae_neg, vad_avg_col_mae_neg, test_avg_col_mae_neg, \
    train_avg_col_mae_all, vad_avg_col_mae_all, test_avg_col_mae_all, \
    train_poisson_pos_plp, vad_poisson_pos_plp, test_poisson_pos_plp, \
    train_poisson_neg_plp, vad_poisson_neg_plp, test_poisson_neg_plp, \
    train_poisson_all_plp, vad_poisson_all_plp, test_poisson_all_plp, \
    train_normal_pos_plp, vad_normal_pos_plp, test_normal_pos_plp, \
    train_normal_neg_plp, vad_normal_neg_plp, test_normal_neg_plp, \
    train_normal_all_plp, vad_normal_all_plp, test_normal_all_plp, \
    train_ndcgk, vad_ndcgk, test_ndcgk, \
    train_recallk, vad_recallk, test_recallk, \
    train_mapk, vad_mapk, test_mapk, \
    train_preck, vad_preck, test_preck, \
    train_ric_rank, vad_ric_rank, test_ric_rank, \
    train_mean_perc_rank, vad_mean_perc_rank, test_mean_perc_rank, \
    test_mae, test_mse = all_metric_holders

    out_df = pd.DataFrame({"dim": outdims, \
                    "train_ndcg": train_ndcg, \
                    "vad_ndcg": vad_ndcg, \
                    "test_ndcg": test_ndcg, \
                    "train_mse_pos": train_mse_pos, \
                    "vad_mse_pos": vad_mse_pos, \
                    "test_mse_pos": test_mse_pos, \
                    "train_mse_neg": train_mse_neg,\
                    "vad_mse_neg": vad_mse_neg, \
                    "test_mse_neg": test_mse_neg, \
                    "train_mse_all": train_mse_all, \
                    "vad_mse_all": vad_mse_all ,\
                    "test_mse_all": test_mse_all, \
                    "train_mae_pos": train_mae_pos, \
                    "vad_mae_pos": vad_mae_pos, \
                    "test_mae_pos": test_mae_pos, \
                    "train_mae_neg": train_mae_neg, \
                    "vad_mae_neg": vad_mae_neg, \
                    "test_mae_neg": test_mae_neg, \
                    "train_mae_all": train_mae_all, \
                    "vad_mae_all": vad_mae_all, \
                    "test_mae_all": test_mae_all, \
                    "train_avg_col_mse_pos": train_avg_col_mse_pos, \
                    "vad_avg_col_mse_pos": vad_avg_col_mse_pos, \
                    "test_avg_col_mse_pos": test_avg_col_mse_pos, \
                    "train_avg_col_mse_neg": train_avg_col_mse_neg, \
                    "vad_avg_col_mse_neg": vad_avg_col_mse_neg, \
                    "test_avg_col_mse_neg": test_avg_col_mse_neg, \
                    "train_avg_col_mse_all": train_avg_col_mse_all, \
                    "vad_avg_col_mse_all": vad_avg_col_mse_all, \
                    "test_avg_col_mse_all": test_avg_col_mse_all, \
                    "train_avg_col_mae_pos": train_avg_col_mae_pos, \
                    "vad_avg_col_mae_pos": vad_avg_col_mae_pos, \
                    "test_avg_col_mae_pos": test_avg_col_mae_pos, \
                    "train_avg_col_mae_neg": train_avg_col_mae_neg, \
                    "vad_avg_col_mae_neg": vad_avg_col_mae_neg, \
                    "test_avg_col_mae_neg": test_avg_col_mae_neg, \
                    "train_avg_col_mae_all": train_avg_col_mae_all, \
                    "vad_avg_col_mae_all": vad_avg_col_mae_all, \
                    "test_avg_col_mae_all": test_avg_col_mae_all, \
                    "train_poisson_pos_plp": train_poisson_pos_plp, \
                    "vad_poisson_pos_plp": vad_poisson_pos_plp, \
                    "test_poisson_pos_plp": test_poisson_pos_plp, \
                    "train_poisson_neg_plp": train_poisson_neg_plp, \
                    "vad_poisson_neg_plp": vad_poisson_neg_plp, \
                    "test_poisson_neg_plp": test_poisson_neg_plp, \
                    "train_poisson_all_plp": train_poisson_all_plp, \
                    "vad_poisson_all_plp": vad_poisson_all_plp, \
                    "test_poisson_all_plp": test_poisson_all_plp, \
                    "train_normal_pos_plp": train_normal_pos_plp, \
                    "vad_normal_pos_plp": vad_normal_pos_plp, \
                    "test_normal_pos_plp": test_normal_pos_plp, \
                    "train_normal_neg_plp": train_normal_neg_plp, \
                    "vad_normal_neg_plp": vad_normal_neg_plp, \
                    "test_normal_neg_plp": test_normal_neg_plp, \
                    "train_normal_all_plp": train_normal_all_plp, \
                    "vad_normal_all_plp": vad_normal_all_plp, \
                    "test_normal_all_plp": test_normal_all_plp, \
                    "train_ric_rank": train_ric_rank, \
                    "vad_ric_rank": vad_ric_rank, \
                    "test_ric_rank": test_ric_rank, \
                    "train_mean_perc_rank": train_mean_perc_rank, \
                    "vad_mean_perc_rank": vad_mean_perc_rank, \
                    "test_mean_perc_rank": test_mean_perc_rank, \
                    "test_mse": test_mse, \
                    "test_mae": test_mae, \
                    "model": np.repeat(model_name, len(outdims)), \
                    "alpha": np.repeat(alpha, len(outdims)), \
                    "K": np.repeat(outdim, len(outdims)), \
                    "K0": np.repeat(caudim, len(outdims)), \
                    "priorU": np.repeat(pri_U, len(outdims)), \
                    "priorV": np.repeat(pri_V, len(outdims)), \
                    "binary": np.repeat(binary, len(outdims)), \
                    "niter": np.repeat(n_iter, len(outdims)), \
                    "batch": np.repeat(M, len(outdims)), \
                    "thold": np.repeat(int(thold+1), len(outdims))})
# last line to avoid thold = 0.5 will turn display as 1 so filename not messed up

    for i, k in enumerate(ks):
        out_df["train_recall"+str(k)] = train_recallk[:,i]
        out_df["vad_recall"+str(k)] = vad_recallk[:,i]
        out_df["test_recall"+str(k)] = test_recallk[:,i]
        out_df["train_ndcg"+str(k)] = train_ndcgk[:,i]
        out_df["vad_ndcg"+str(k)] = vad_ndcgk[:,i]
        out_df["test_ndcg"+str(k)] = test_ndcgk[:,i]
        out_df["train_prec"+str(k)] = train_preck[:,i]
        out_df["vad_prec"+str(k)] = vad_preck[:,i]
        out_df["test_prec"+str(k)] = test_preck[:,i]
        out_df["train_map"+str(k)] = train_mapk[:,i]
        out_df["vad_map"+str(k)] = vad_mapk[:,i]
        out_df["test_map"+str(k)] = test_mapk[:,i]

    return out_df


def wg_eval_acc_metrics_update_i_nomeanperc(all_metric_holders, i, pred, train_data, \
    vad_data, test_data, ks, thold):
        # pred must be sparse matrix

    train_ndcg, vad_ndcg, test_ndcg, \
    train_mse_pos, vad_mse_pos, test_mse_pos, \
    train_mse_neg, vad_mse_neg, test_mse_neg, \
    train_mse_all, vad_mse_all, test_mse_all, \
    train_mae_pos, vad_mae_pos, test_mae_pos, \
    train_mae_neg, vad_mae_neg, test_mae_neg, \
    train_mae_all, vad_mae_all, test_mae_all, \
    train_avg_col_mse_pos, vad_avg_col_mse_pos, test_avg_col_mse_pos, \
    train_avg_col_mse_neg, vad_avg_col_mse_neg, test_avg_col_mse_neg, \
    train_avg_col_mse_all, vad_avg_col_mse_all, test_avg_col_mse_all, \
    train_avg_col_mae_pos, vad_avg_col_mae_pos, test_avg_col_mae_pos, \
    train_avg_col_mae_neg, vad_avg_col_mae_neg, test_avg_col_mae_neg, \
    train_avg_col_mae_all, vad_avg_col_mae_all, test_avg_col_mae_all, \
    train_poisson_pos_plp, vad_poisson_pos_plp, test_poisson_pos_plp, \
    train_poisson_neg_plp, vad_poisson_neg_plp, test_poisson_neg_plp, \
    train_poisson_all_plp, vad_poisson_all_plp, test_poisson_all_plp, \
    train_normal_pos_plp, vad_normal_pos_plp, test_normal_pos_plp, \
    train_normal_neg_plp, vad_normal_neg_plp, test_normal_neg_plp, \
    train_normal_all_plp, vad_normal_all_plp, test_normal_all_plp, \
    train_ndcgk, vad_ndcgk, test_ndcgk, \
    train_recallk, vad_recallk, test_recallk, \
    train_mapk, vad_mapk, test_mapk, \
    train_preck, vad_preck, test_preck, \
    train_ric_rank, vad_ric_rank, test_ric_rank, \
    train_mean_perc_rank, vad_mean_perc_rank, test_mean_perc_rank, \
    test_mae, test_mse = all_metric_holders


    train_mse_pos[i], train_mse_neg[i], train_mse_all[i] = \
        mse(train_data.todense(), pred)

    vad_mse_pos[i], vad_mse_neg[i], vad_mse_all[i] = \
        mse(vad_data.todense(), pred)

    test_mse_pos[i], test_mse_neg[i], test_mse_all[i] = \
        mse(test_data.todense(), pred)

    train_mae_pos[i], train_mae_neg[i], train_mae_all[i] = \
        mae(train_data.todense(), pred)

    vad_mae_pos[i], vad_mae_neg[i], vad_mae_all[i] = \
        mae(vad_data.todense(), pred)

    test_mae_pos[i], test_mae_neg[i], test_mae_all[i] = \
        mae(test_data.todense(), pred)

    train_avg_col_mse_pos[i], train_avg_col_mse_neg[i], train_avg_col_mse_all[i] = \
        avg_col_mse(train_data.todense(), pred)

    vad_avg_col_mse_pos[i], vad_avg_col_mse_neg[i], vad_avg_col_mse_all[i] = \
        avg_col_mse(vad_data.todense(), pred)

    test_avg_col_mse_pos[i], test_avg_col_mse_neg[i], test_avg_col_mse_all[i] = \
        avg_col_mse(test_data.todense(), pred)     

    train_avg_col_mae_pos[i], train_avg_col_mae_neg[i], train_avg_col_mae_all[i] = \
        avg_col_mae(train_data.todense(), pred)

    vad_avg_col_mae_pos[i], vad_avg_col_mae_neg[i], vad_avg_col_mae_all[i] = \
        avg_col_mae(vad_data.todense(), pred)

    test_avg_col_mae_pos[i], test_avg_col_mae_neg[i], test_avg_col_mae_all[i] = \
        avg_col_mae(test_data.todense(), pred)


    print("train mse pos neg all", train_mse_pos[i], train_mse_neg[i], train_mse_all[i])
    print("vad mse pos neg all", vad_mse_pos[i], vad_mse_neg[i], vad_mse_all[i])
    print("test mse pos neg all", test_mse_pos[i], test_mse_neg[i], test_mse_all[i])

    print("train mae pos neg all", train_mae_pos[i], train_mae_neg[i], train_mae_all[i])
    print("vad mae pos neg all", vad_mae_pos[i], vad_mae_neg[i], vad_mae_all[i])
    print("test mae pos neg all", test_mae_pos[i], test_mae_neg[i], test_mae_all[i])

    print("train avg_col_mse pos neg all", train_avg_col_mse_pos[i], train_avg_col_mse_neg[i], train_avg_col_mse_all[i])
    print("vad avg_col_mse pos neg all", vad_avg_col_mse_pos[i], vad_avg_col_mse_neg[i], vad_avg_col_mse_all[i])
    print("test avg_col_mse pos neg all", test_avg_col_mse_pos[i], test_avg_col_mse_neg[i], test_avg_col_mse_all[i])

    print("train avg_col_mae pos neg all", train_avg_col_mae_pos[i], train_avg_col_mae_neg[i], train_avg_col_mae_all[i])
    print("vad avg_col_mae pos neg all", vad_avg_col_mae_pos[i], vad_avg_col_mae_neg[i], vad_avg_col_mae_all[i])
    print("test avg_col_mae pos neg all", test_avg_col_mae_pos[i], test_avg_col_mae_neg[i], test_avg_col_mae_all[i])


    train_poisson_pos_plp[i], train_poisson_neg_plp[i], train_poisson_all_plp[i] = \
        log_cond_poisson_prob_metrics(train_data.todense(), pred)

    vad_poisson_pos_plp[i], vad_poisson_neg_plp[i], vad_poisson_all_plp[i] = \
        log_cond_poisson_prob_metrics(vad_data.todense(), pred)

    test_poisson_pos_plp[i], test_poisson_neg_plp[i], test_poisson_all_plp[i] = \
        log_cond_poisson_prob_metrics(test_data.todense(), pred)

    train_normal_pos_plp[i], train_normal_neg_plp[i], train_normal_all_plp[i] = \
        log_cond_normal_prob_metrics(train_data.todense(), pred)

    vad_normal_pos_plp[i], vad_normal_neg_plp[i], vad_normal_all_plp[i] = \
        log_cond_normal_prob_metrics(vad_data.todense(), pred)

    test_normal_pos_plp[i], test_normal_neg_plp[i], test_normal_all_plp[i] = \
        log_cond_normal_prob_metrics(test_data.todense(), pred)

    print("train poisson pos neg all", train_poisson_pos_plp[i], train_poisson_neg_plp[i], train_poisson_all_plp[i])
    print("vad poisson pos neg all", vad_poisson_pos_plp[i], vad_poisson_neg_plp[i], vad_poisson_all_plp[i])
    print("test poisson pos neg all", test_poisson_pos_plp[i], test_poisson_neg_plp[i], test_poisson_all_plp[i])

    print("train normal pos neg all", train_normal_pos_plp[i], train_normal_neg_plp[i], train_normal_all_plp[i])
    print("vad normal pos neg all", vad_normal_pos_plp[i], vad_normal_neg_plp[i], vad_normal_all_plp[i])
    print("test normal pos neg all", test_normal_pos_plp[i], test_normal_neg_plp[i], test_normal_all_plp[i])

    
    
    train_ndcg[i], vad_ndcg[i], test_ndcg[i] = \
        normalized_dcg_nonbinary_xpred(vad_data, train_data, pred), \
        normalized_dcg_nonbinary_xpred(train_data, vad_data, pred), \
        normalized_dcg_nonbinary_xpred(train_data, test_data, pred, vad_data = vad_data)
    
    train_ndcgk[i], vad_ndcgk[i], test_ndcgk[i] =\
        [normalized_dcg_at_k_nonbinary_xpred(vad_data, train_data, pred, k=k) for k in ks], \
        [normalized_dcg_at_k_nonbinary_xpred(train_data, vad_data, pred, k=k) for k in ks], \
        [normalized_dcg_at_k_nonbinary_xpred(train_data, test_data, pred, k=k, vad_data = vad_data) for k in ks]


    train_recallk[i], vad_recallk[i], test_recallk[i] = \
        [recall_at_k_xpred(vad_data, train_data, pred, k=k, thold=thold) for k in ks], \
        [recall_at_k_xpred(train_data, vad_data, pred, k=k, thold=thold) for k in ks], \
        [recall_at_k_xpred(train_data, test_data, pred, vad_data = vad_data, k=k, thold=thold) for k in ks]

    train_mapk[i], vad_mapk[i], test_mapk[i] = \
        [map_at_k_xpred(vad_data, train_data, pred, k=k, thold=thold) for k in ks], \
        [map_at_k_xpred(train_data, vad_data, pred, k=k, thold=thold) for k in ks], \
        [map_at_k_xpred(train_data, test_data, pred, vad_data = vad_data, k=k, thold=thold) for k in ks]

    train_preck[i], vad_preck[i], test_preck[i] = \
        [prec_at_k_xpred(vad_data, train_data, pred, k=k, thold=thold) for k in ks], \
        [prec_at_k_xpred(train_data, vad_data, pred, k=k, thold=thold) for k in ks], \
        [prec_at_k_xpred(train_data, test_data, pred, vad_data = vad_data, k=k, thold=thold) for k in ks]

    train_ric_rank[i], vad_ric_rank[i], test_ric_rank[i] = \
        ric_rank_xpred(vad_data, train_data, pred, thold=thold), \
        ric_rank_xpred(train_data, vad_data, pred, thold=thold), \
        ric_rank_xpred(train_data, test_data, pred, vad_data = vad_data, thold=thold)
    

    print("ndcg train vad test", train_ndcg[i], vad_ndcg[i], test_ndcg[i])
    
    print("recall train vad test", train_recallk[i], vad_recallk[i], test_recallk[i])
    
    print("map train vad test", train_mapk[i], vad_mapk[i], test_mapk[i])            

    print("prec train vad test", train_preck[i], vad_preck[i], test_preck[i])

    print("ric_rank train vad test", train_ric_rank[i], vad_ric_rank[i], test_ric_rank[i])
    
    print("mean_perc_rank train vad test", train_mean_perc_rank[i], vad_mean_perc_rank[i], test_mean_perc_rank[i])
        
    test_mse[i] = np.mean(np.square(test_data[test_data.nonzero()] - pred[test_data.nonzero()]))
    test_mae[i] = np.mean(np.abs(test_data[test_data.nonzero()] - pred[test_data.nonzero()]))
  
    print('mse', test_mse[i])
    print('mae', test_mae[i])

    return train_ndcg, vad_ndcg, test_ndcg, \
        train_mse_pos, vad_mse_pos, test_mse_pos, \
        train_mse_neg, vad_mse_neg, test_mse_neg, \
        train_mse_all, vad_mse_all, test_mse_all, \
        train_mae_pos, vad_mae_pos, test_mae_pos, \
        train_mae_neg, vad_mae_neg, test_mae_neg, \
        train_mae_all, vad_mae_all, test_mae_all, \
        train_avg_col_mse_pos, vad_avg_col_mse_pos, test_avg_col_mse_pos, \
        train_avg_col_mse_neg, vad_avg_col_mse_neg, test_avg_col_mse_neg, \
        train_avg_col_mse_all, vad_avg_col_mse_all, test_avg_col_mse_all, \
        train_avg_col_mae_pos, vad_avg_col_mae_pos, test_avg_col_mae_pos, \
        train_avg_col_mae_neg, vad_avg_col_mae_neg, test_avg_col_mae_neg, \
        train_avg_col_mae_all, vad_avg_col_mae_all, test_avg_col_mae_all, \
        train_poisson_pos_plp, vad_poisson_pos_plp, test_poisson_pos_plp, \
        train_poisson_neg_plp, vad_poisson_neg_plp, test_poisson_neg_plp, \
        train_poisson_all_plp, vad_poisson_all_plp, test_poisson_all_plp, \
        train_normal_pos_plp, vad_normal_pos_plp, test_normal_pos_plp, \
        train_normal_neg_plp, vad_normal_neg_plp, test_normal_neg_plp, \
        train_normal_all_plp, vad_normal_all_plp, test_normal_all_plp, \
        train_ndcgk, vad_ndcgk, test_ndcgk, \
        train_recallk, vad_recallk, test_recallk, \
        train_mapk, vad_mapk, test_mapk, \
        train_preck, vad_preck, test_preck, \
        train_ric_rank, vad_ric_rank, test_ric_rank, \
        train_mean_perc_rank, vad_mean_perc_rank, test_mean_perc_rank, \
        test_mae, test_mse

