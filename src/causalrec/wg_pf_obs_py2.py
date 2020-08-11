from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys


# import edward as ed
import numpy as np
import tensorflow as tf
import pandas as pd
import numpy.random as npr
import pmf

# from edward.models import Normal, Gamma, Dirichlet, InverseGamma, \
#     Poisson, PointMass, Empirical, ParamMixture, \
#     MultivariateNormalDiag, Categorical, Laplace,\
#     MultivariateNormalTriL, Bernoulli

from scipy import sparse, stats
from scipy.stats import poisson, norm
import bottleneck as bn

import argparse

from rec_eval import normalized_dcg_nonbinary, recall_at_k, \
map_at_k, prec_at_k, ric_rank, mean_perc_rank

from rec_eval_xpred import log_cond_poisson_prob_metrics,\
normalized_dcg_nonbinary_xpred, recall_at_k_xpred, map_at_k_xpred,\
prec_at_k_xpred, ric_rank_xpred, mean_perc_rank_xpred, \
normalized_dcg_at_k_nonbinary_xpred, log_cond_poisson_prob_metrics

from utils import binarize_rating, exp_to_imp, binarize_spmat


npr.seed(0)
# ed.set_seed(0)
tf.set_random_seed(0)


def next_batch(x_train, M):
  idx_batch = np.random.choice(x_train.shape[0],M)
  return x_train[idx_batch,:], idx_batch

if __name__ == '__main__':

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
        type=int, default=100)
    parser.add_argument('-nitr', '--n_iter', \
        type=int, default=10)
    parser.add_argument('-pU', '--priorU', \
        type=int, default=1) # it is the inverse of priorU
    parser.add_argument('-pV', '--priorV', \
        type=int, default=1) # it is the inverse of priorV
    parser.add_argument('-alpha', '--alpha', \
        type=int, default=40)
    parser.add_argument('-binary', '--binary', \
        type=int, default=0)

    args = parser.parse_args()


    print("setting params")

    DATA_DIR = args.datadir
    CAUSEFIT_DIR = args.causedir
    OUT_DATA_DIR = args.outdatadir

    print("data/cause/out directories", DATA_DIR, CAUSEFIT_DIR, OUT_DATA_DIR)

    outdim = args.outdim
    caudim = args.caudim
    
    thold = args.thold # only count ratings > thold as relevant in recall and map
    M = args.M  #batch size
    n_iter = args.n_iter
    binary = args.binary

    print("relevance thold", thold)

    print("batch size", M, "n_iter", n_iter)

    print("outdim", outdim)
    print("caudim", caudim)

    outdims = np.array([outdim])
    dims = np.array([caudim])

    ks = np.array([1,2,3,4,5,6,7,8,9,10,20,50,100])

    pri_U = 1./args.priorU
    pri_V = 1./args.priorV

    print("prior sd on U", pri_U, "prior sd on V", pri_V)

    print("sanity check of loading the right dataset") 
    unique_uid = list()
    with open(os.path.join(DATA_DIR, 'unique_uid.txt'), 'r') as f:
        for line in f:
            unique_uid.append(line.strip())
        
    unique_sid = list()
    with open(os.path.join(DATA_DIR, 'unique_sid.txt'), 'r') as f:
        for line in f:
            unique_sid.append(line.strip())

    n_items = len(unique_sid)
    n_users = len(unique_uid)

    print(n_users, n_items)

    def load_data(csv_file, shape=(n_users, n_items)):
        tp = pd.read_csv(csv_file)
        rows, cols, vals = np.array(tp['uid']), np.array(tp['sid']), np.array(tp['rating']) 
        data = sparse.csr_matrix((vals, (rows, cols)), dtype=np.float32, shape=shape)
        return data

    train_data = load_data(os.path.join(DATA_DIR, 'train.csv'))
    test_data = load_data(os.path.join(DATA_DIR, 'test_full.csv'))
    vad_data = load_data(os.path.join(DATA_DIR, 'validation.csv'))

    alpha = args.alpha # alpha 0 is vanilla gmf

    print("alpha", alpha)

    cau = exp_to_imp(train_data, 0.5)

    if binary > 0:
        train_data = binarize_rating(train_data)

    train_data_coo = train_data.tocoo()
    row_tr, col_tr = train_data_coo.row, train_data_coo.col

    vad_data_coo = vad_data.tocoo()
    row_vd, col_vd = vad_data_coo.row, vad_data_coo.col

    model_name = 'wg_pmf_obs'

    print("model", model_name)



    for dim in dims:
        U = np.loadtxt(CAUSEFIT_DIR + '/cause_pmf_k'+str(dim)+'_U.csv')
        V = np.loadtxt(CAUSEFIT_DIR + '/cause_pmf_k'+str(dim)+'_V.csv')
        U = (np.atleast_2d(U.T).T)
        V = (np.atleast_2d(V.T).T)
        reconstr_cau = U.dot(V.T)
        

        train_ndcg, vad_ndcg, test_ndcg = \
            np.zeros(len(outdims)), np.zeros(len(outdims)), np.zeros(len(outdims))
        train_pos_plp, vad_pos_plp, test_pos_plp = \
            np.zeros(len(outdims)), np.zeros(len(outdims)), np.zeros(len(outdims))
        train_neg_plp, vad_neg_plp, test_neg_plp = \
            np.zeros(len(outdims)), np.zeros(len(outdims)), np.zeros(len(outdims))
        train_all_plp, vad_all_plp, test_all_plp = \
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
    
        for i, K in enumerate(outdims):
            print("K0", dim, "K", K)

            pf = pmf.PoissonMF(n_components=K, random_state=98765, verbose=True, a=0.3, b=0.3, c=0.3, d=0.3)
            pf.fit(train_data, row_tr, col_tr, dict(X_new=vad_data.data, rows_new=row_vd, cols_new=col_vd))
            U_out, V_out = pf.Eb, pf.Et.T
            
                
            np.savetxt(OUT_DATA_DIR + '/'+ model_name + '_bin'+str(binary)+'_cauk0_'+str(caudim)\
                +'outK'+str(outdim)+"_nitr"+str(n_iter)+"_batch"+str(M)+\
                "_thold"+str(thold)+"_pU"+str(args.priorU)+"_pV"+str(args.priorV)+\
                "_alpha"+str(args.alpha)+"_cas_U.csv", U_out)
            np.savetxt(OUT_DATA_DIR + '/'+ model_name + '_bin'+str(binary)+'_cauk0_'+str(caudim)\
                +'outK'+str(outdim)+"_nitr"+str(n_iter)+"_batch"+str(M)+\
                "_thold"+str(thold)+"_pU"+str(args.priorU)+"_pV"+str(args.priorV)+\
                "_alpha"+str(args.alpha)+"_cas_V.csv", V_out)

            pred = sparse.csr_matrix(U_out.dot(V_out.T))

            train_pos_plp[i], train_neg_plp[i], train_all_plp[i] = \
                log_cond_poisson_prob_metrics(train_data, pred)

            vad_pos_plp[i], vad_neg_plp[i], vad_all_plp[i] = \
                log_cond_poisson_prob_metrics(vad_data, pred)

            test_pos_plp[i], test_neg_plp[i], test_all_plp[i] = \
                log_cond_poisson_prob_metrics(test_data, pred)

            print("train pos neg all", train_pos_plp[i], train_neg_plp[i], train_all_plp[i])
            print("vad pos neg all", vad_pos_plp[i], vad_neg_plp[i], vad_all_plp[i])
            print("test pos neg all", test_pos_plp[i], test_neg_plp[i], test_all_plp[i])

            pred = pred.todense()
            
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

    gmf_obs = pd.DataFrame({"dim": outdims, \
                        "train_ndcg": train_ndcg, \
                        "vad_ndcg": vad_ndcg, \
                        "test_ndcg": test_ndcg, \
                        "train_pos_plp": train_pos_plp, \
                        "vad_pos_plp": vad_pos_plp, \
                        "test_pos_plp": test_pos_plp, \
                        "train_neg_plp": train_neg_plp, \
                        "vad_neg_plp": vad_neg_plp, \
                        "test_neg_plp": test_neg_plp, \
                        "train_all_plp": train_all_plp, \
                        "vad_all_plp": vad_all_plp, \
                        "test_all_plp": test_all_plp, \
                        "train_ric_rank": train_ric_rank, \
                        "vad_ric_rank": vad_ric_rank, \
                        "test_ric_rank": test_ric_rank, \
                        "train_mean_perc_rank": train_mean_perc_rank, \
                        "vad_mean_perc_rank": vad_mean_perc_rank, \
                        "test_mean_perc_rank": test_mean_perc_rank, \
                        "test_mse": test_mse, \
                        "test_mae": test_mae, \
                        "model": np.repeat(model_name, len(outdims)), \
                        "alpha": np.repeat(args.alpha, len(outdims)), \
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
        gmf_obs["train_recall"+str(k)] = train_recallk[:,i]
        gmf_obs["vad_recall"+str(k)] = vad_recallk[:,i]
        gmf_obs["test_recall"+str(k)] = test_recallk[:,i]
        gmf_obs["train_ndcg"+str(k)] = train_ndcgk[:,i]
        gmf_obs["vad_ndcg"+str(k)] = vad_ndcgk[:,i]
        gmf_obs["test_ndcg"+str(k)] = test_ndcgk[:,i]
        gmf_obs["train_prec"+str(k)] = train_preck[:,i]
        gmf_obs["vad_prec"+str(k)] = vad_preck[:,i]
        gmf_obs["test_prec"+str(k)] = test_preck[:,i]
        gmf_obs["train_map"+str(k)] = train_mapk[:,i]
        gmf_obs["vad_map"+str(k)] = vad_mapk[:,i]
        gmf_obs["test_map"+str(k)] = test_mapk[:,i]

        
    gmf_obs.to_csv(OUT_DATA_DIR + '/res_'+ model_name + '_bin'+str(binary)+'_cauk0_'+str(caudim)\
        +'outK'+str(outdim)+"_nitr"+str(n_iter)+"_batch"+str(M)+\
        "_thold"+str(int(thold+1))+"_pU"+str(args.priorU)+"_pV"+str(args.priorV)+\
        "_alpha"+str(args.alpha)+".csv")

