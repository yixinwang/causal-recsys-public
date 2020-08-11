

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import edward as ed
import numpy as np
import tensorflow as tf
import pandas as pd
import numpy.random as npr
import random

import time

from edward.models import Normal, Gamma, Dirichlet, InverseGamma, \
    Poisson, PointMass, Empirical, ParamMixture, \
    MultivariateNormalDiag, Categorical, Laplace,\
    MultivariateNormalTriL, Bernoulli

from scipy import sparse, stats
from scipy.stats import poisson, norm
import bottleneck as bn

import argparse


from utils import binarize_rating, exp_to_imp, binarize_spmat, \
next_batch, create_argparser, set_params, load_prefit_pfcau, \
create_metric_holders, wg_eval_acc_metrics_update_i, \
sg_eval_acc_metrics_update_i, save_eval_metrics

randseed = int(time.time())
print("random seed: ", randseed)
random.seed(randseed)
npr.seed(randseed)
ed.set_seed(randseed)
tf.set_random_seed(randseed)



if __name__ == '__main__':

    parser = create_argparser()
    args = parser.parse_args()

    all_params = set_params(args)
    DATA_DIR, CAUSEFIT_DIR, OUT_DATA_DIR, \
        outdim, caudim, thold, M, n_iter, binary, \
        pri_U, pri_V, alpha = all_params


    print("setting params....")
    print("data/cause/out directories", DATA_DIR, CAUSEFIT_DIR, OUT_DATA_DIR)
    print("relevance thold", thold)
    print("batch size", M, "n_iter", n_iter)
    print("outdim", outdim)
    print("caudim", caudim)
    print("prior sd on U", pri_U, "prior sd on V", pri_V)
    print("alpha", alpha)




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

    if binary > 0:
        train_data = binarize_rating(train_data)
        vad_data_tr, vad_data_te = binarize_rating(vad_data_tr), binarize_rating(vad_data_te)
        test_data_tr, test_data_te = binarize_rating(test_data_tr), binarize_rating(test_data_te)

    
    model_name = 'sg_pf_cau_user_add'

    dat_name = DATA_DIR.split('/')[-1]

    out_filename = model_name+ \
        '_datadir'+str(dat_name) + \
        '_bin'+str(binary)+ \
        '_cauk0_'+str(caudim)+ \
        'outK'+str(outdim)+ \
        "_nitr"+str(n_iter)+ \
        "_batch"+str(M)+ \
        "_thold"+str(int(thold+1))+ \
        "_pU"+str(args.priorU)+ \
        "_pV"+str(args.priorV)+ \
        "_alpha"+str(args.alpha)+ \
        "_randseed"+str(randseed)

    # last line to avoid thold = 0.5 will turn display as 1 \
    # so filename not messed up

    print("#############\nmodel", model_name)
    print("out_filename", out_filename)

    outdims = np.array([outdim])
    dims = np.array([caudim])
    ks = np.array([1,2,3,4,5,6,7,8,9,10,20,50,100])


    all_metric_holders = create_metric_holders(outdims, ks)


    for dim in dims:

        U = np.loadtxt(CAUSEFIT_DIR + '/cause_pmf_k'+str(dim)+'_trainU.csv')
        V = np.loadtxt(CAUSEFIT_DIR + '/cause_pmf_k'+str(dim)+'_V.csv')
        U = (np.atleast_2d(U.T).T)
        V = (np.atleast_2d(V.T).T)
        reconstr_cau_train = U.dot(V.T)
    

        for i, K in enumerate(outdims):
            print("K0", dim, "K", K)

            D = train_data.shape[0]
            N = train_data.shape[1]
            weights = train_data * alpha
            cau = exp_to_imp(train_data)

            tf.reset_default_graph()
            sess = tf.InteractiveSession()

            idx_ph = tf.placeholder(tf.int32, M)
            cau_ph = tf.placeholder(tf.float32, [M, N])
            sd_ph = tf.placeholder(tf.float32, [M, N])
            reconstr_cau_ph = tf.placeholder(tf.float32, [M, N])

            U = Gamma(0.3*tf.ones([M, K]), 0.3*tf.ones([M, K]))
            V = Gamma(0.3*tf.ones([N, K]), 0.3*tf.ones([N, K]))
            gamma = Gamma(tf.ones([M, 1]), tf.ones([M, 1]))
            beta0 = Gamma(0.3*tf.ones([1, 1]), 0.3*tf.ones([1, 1]))
            
            x = Poisson(tf.add(tf.matmul(U, V, transpose_b=True),\
                tf.multiply(tf.matmul(gamma, tf.ones([1, N])), \
                    reconstr_cau_ph)) + beta0)


            qU_variables = [tf.Variable(tf.random_uniform([D, K])), \
                           tf.Variable(tf.random_uniform([D, K]))]

            qU = PointMass(params=tf.nn.softplus(tf.gather(qU_variables[0], idx_ph)))


            qV_variables = [tf.Variable(tf.random_uniform([N, K])), \
                           tf.Variable(tf.random_uniform([N, K]))]

            qV = PointMass(params=tf.nn.softplus(qV_variables[0]))

            qgamma_variables = [tf.Variable(tf.random_uniform([D, 1])), \
                           tf.Variable(tf.nn.softplus(tf.random_uniform([D, 1])))]

            qgamma = PointMass(params=tf.nn.softplus(tf.gather(qgamma_variables[0], idx_ph)))

            
            qbeta0_variables = [tf.Variable(tf.random_uniform([1, 1])), \
                           tf.Variable(tf.nn.softplus(tf.random_uniform([1, 1])))]

            qbeta0 = PointMass(params=tf.nn.softplus(qbeta0_variables[0]))

            x_ph = tf.placeholder(tf.float32, [M, N])

            optimizer = tf.train.RMSPropOptimizer(5e-5)
        
            scale_factor = float(D) / M

            inference_U = ed.MAP({U: qU}, \
                data={x: x_ph, V: qV, gamma: qgamma, beta0: qbeta0})
            inference_V = ed.MAP({V: qV}, \
                data={x: x_ph, U: qU, gamma: qgamma, beta0: qbeta0})
            inference_gamma = ed.MAP({gamma: qgamma}, \
                data={x: x_ph, V: qV, U: qU, beta0: qbeta0})
            inference_beta0 = ed.MAP({beta0: qbeta0}, \
                data={x: x_ph, V: qV, U: qU, gamma: qgamma})

            inference_U.initialize(scale={x: scale_factor, U: scale_factor, gamma: scale_factor},
                                 var_list=qU_variables, optimizer=optimizer)
            inference_V.initialize(scale={x: scale_factor, U: scale_factor, gamma: scale_factor},
                                 var_list=qV_variables, n_iter=n_iter, optimizer=optimizer)
            inference_gamma.initialize(scale={x: scale_factor, U: scale_factor, gamma: scale_factor},
                                 var_list=qgamma_variables, optimizer=optimizer)
            inference_beta0.initialize(scale={x: scale_factor, U: scale_factor, gamma: scale_factor},
                                 var_list=qbeta0_variables, optimizer=optimizer)        

            tf.global_variables_initializer().run()
            
            loss = np.empty(inference_V.n_iter, dtype=np.float32)
            
            for j in range(inference_V.n_iter):
                x_batch, idx_batch = next_batch(train_data, M)
                cau_batch = cau[idx_batch,:]
                weights_batch = weights[idx_batch,:]
                reconstr_cau_batch = reconstr_cau_train[idx_batch,:]

                x_batch = x_batch.todense().astype('int')
                cau_batch = cau_batch.todense()
                weights_batch = weights_batch.todense()
                sd_batch = 1./np.sqrt(1+weights_batch)

                info_dict = inference_V.update(feed_dict={x_ph: x_batch, idx_ph: idx_batch, \
                    reconstr_cau_ph: reconstr_cau_batch, cau_ph: cau_batch, sd_ph: sd_batch})
                inference_beta0.update(feed_dict={x_ph: x_batch, idx_ph: idx_batch, \
                    reconstr_cau_ph: reconstr_cau_batch, cau_ph: cau_batch, sd_ph: sd_batch})
                inference_gamma.update(feed_dict={x_ph: x_batch, idx_ph: idx_batch, \
                    reconstr_cau_ph: reconstr_cau_batch, cau_ph: cau_batch, sd_ph: sd_batch})
                inference_U.update(feed_dict={x_ph: x_batch, idx_ph: idx_batch, \
                    reconstr_cau_ph: reconstr_cau_batch, cau_ph: cau_batch, sd_ph: sd_batch})

                inference_V.print_progress(info_dict)


                loss[j] = info_dict["loss"]
                

            V_out = tf.nn.softplus(qV_variables[0]).eval()
            U_trainout = tf.nn.softplus(qU_variables[0]).eval()
            gamma_trainout = tf.nn.softplus(qgamma_variables[0]).eval()
            beta0_out = tf.nn.softplus(qbeta0_variables[0]).eval()
            
                

            
            # np.savetxt(OUT_DATA_DIR + '/'+ out_filename + "_cas_trainU.csv", U_trainout)
            # np.savetxt(OUT_DATA_DIR + '/'+ out_filename + "_cas_V.csv", V_out)
            # np.savetxt(OUT_DATA_DIR + '/'+ out_filename + "_cas_gamma.csv", gamma_trainout)
            # np.savetxt(OUT_DATA_DIR + '/'+ out_filename + "_cas_beta0.csv", beta0_out)



            # now estimate the user vector on new validation users 
            D = vad_data_tr.shape[0]
            weights = vad_data_tr * alpha
            cau = exp_to_imp(vad_data_tr)

            U = np.loadtxt(CAUSEFIT_DIR + '/cause_pmf_k'+str(dim)+'_vadU.csv')
            V = np.loadtxt(CAUSEFIT_DIR + '/cause_pmf_k'+str(dim)+'_V.csv')
            U = (np.atleast_2d(U.T).T)
            V = (np.atleast_2d(V.T).T)
            reconstr_cau_vad = U.dot(V.T)

            tf.reset_default_graph()
            sess = tf.InteractiveSession()

            idx_ph = tf.placeholder(tf.int32, M)
            cau_ph = tf.placeholder(tf.float32, [M, N])
            sd_ph = tf.placeholder(tf.float32, [M, N])
            reconstr_cau_ph = tf.placeholder(tf.float32, [M, N])

            U = Gamma(0.3*tf.ones([M, K]), 0.3*tf.ones([M, K]))
            V = tf.placeholder(tf.float32, [N, K])
            gamma = Gamma(tf.ones([M, 1]), tf.ones([M, 1]))
            beta0 = tf.placeholder(tf.float32, [1, 1])

            x = Poisson(tf.add(tf.matmul(U, V, transpose_b=True),\
                tf.multiply(tf.matmul(gamma, tf.ones([1, N])), \
                    reconstr_cau_ph)) + beta0)

            qU_variables = [tf.Variable(tf.random_uniform([D, K])), \
                           tf.Variable(tf.random_uniform([D, K]))]

            qU = PointMass(params=tf.nn.softplus(tf.gather(qU_variables[0], idx_ph)))


            qgamma_variables = [tf.Variable(tf.random_uniform([D, 1])), \
                           tf.Variable(tf.nn.softplus(tf.random_uniform([D, 1])))]

            qgamma = PointMass(params=tf.nn.softplus(tf.gather(qgamma_variables[0], idx_ph)))
            
            x_ph = tf.placeholder(tf.float32, [M, N])


            scale_factor = float(D) / M

            inference_U = ed.MAP({U: qU}, \
                data={x: x_ph, gamma: qgamma})
            inference_gamma = ed.MAP({gamma: qgamma}, \
                data={x: x_ph, U: qU})

            inference_U.initialize(scale={x: scale_factor, U: scale_factor, gamma: scale_factor},
                                 var_list=qU_variables, n_iter=n_iter, optimizer=optimizer)
            inference_gamma.initialize(scale={x: scale_factor, U: scale_factor, gamma: scale_factor},
                                 var_list=qgamma_variables, optimizer=optimizer)

            tf.global_variables_initializer().run()

            loss = np.empty(inference_U.n_iter, dtype=np.float32)
            
            for j in range(inference_U.n_iter):
                x_batch, idx_batch = next_batch(vad_data_tr, M)
                cau_batch = cau[idx_batch,:]
                weights_batch = weights[idx_batch,:]
                reconstr_cau_batch = reconstr_cau_vad[idx_batch,:]

                x_batch = x_batch.todense().astype('int')
                cau_batch = cau_batch.todense()
                weights_batch = weights_batch.todense()
                sd_batch = 1./np.sqrt(1+weights_batch)

                info_dict = inference_U.update(feed_dict={x_ph: x_batch, idx_ph: idx_batch, \
                                              reconstr_cau_ph: reconstr_cau_batch, cau_ph: cau_batch, \
                                              sd_ph: sd_batch, V: V_out, beta0: beta0_out})
                inference_gamma.update(feed_dict={x_ph: x_batch, idx_ph: idx_batch, \
                                              reconstr_cau_ph: reconstr_cau_batch, cau_ph: cau_batch, \
                                              sd_ph: sd_batch, V: V_out, beta0: beta0_out})

                inference_U.print_progress(info_dict)

                loss[j] = info_dict["loss"]
                

            U_vadout = tf.nn.softplus(qU_variables[0]).eval()
            gamma_vadout = tf.nn.softplus(qgamma_variables[0]).eval()
            
                

            
            # np.savetxt(OUT_DATA_DIR + '/'+ out_filename + "_cas_vadU.csv", U_vadout)
            # np.savetxt(OUT_DATA_DIR + '/'+ out_filename + "_cas_vadgamma.csv", gamma_vadout)



            # now estimate the user vector on new test users 
            D = test_data_tr.shape[0]
            weights = test_data_tr * alpha
            cau = exp_to_imp(test_data_tr)

            U = np.loadtxt(CAUSEFIT_DIR + '/cause_pmf_k'+str(dim)+'_testU.csv')
            V = np.loadtxt(CAUSEFIT_DIR + '/cause_pmf_k'+str(dim)+'_V.csv')
            U = (np.atleast_2d(U.T).T)
            V = (np.atleast_2d(V.T).T)
            reconstr_cau_test = U.dot(V.T)

            tf.reset_default_graph()
            sess = tf.InteractiveSession()

            idx_ph = tf.placeholder(tf.int32, M)
            cau_ph = tf.placeholder(tf.float32, [M, N])
            sd_ph = tf.placeholder(tf.float32, [M, N])
            reconstr_cau_ph = tf.placeholder(tf.float32, [M, N])

            U = Gamma(0.3*tf.ones([M, K]), 0.3*tf.ones([M, K]))
            V = tf.placeholder(tf.float32, [N, K])
            gamma = Gamma(tf.ones([M, 1]), tf.ones([M, 1]))
            beta0 = tf.placeholder(tf.float32, [1, 1])

            x = Poisson(tf.add(tf.matmul(U, V, transpose_b=True),\
                tf.multiply(tf.matmul(gamma, tf.ones([1, N])), \
                    reconstr_cau_ph)) + beta0)

            qU_variables = [tf.Variable(tf.random_uniform([D, K])), \
                           tf.Variable(tf.random_uniform([D, K]))]

            qU = PointMass(params=tf.nn.softplus(tf.gather(qU_variables[0], idx_ph)))


            qgamma_variables = [tf.Variable(tf.random_uniform([D, 1])), \
                           tf.Variable(tf.nn.softplus(tf.random_uniform([D, 1])))]

            qgamma = PointMass(params=tf.nn.softplus(tf.gather(qgamma_variables[0], idx_ph)))
            
            x_ph = tf.placeholder(tf.float32, [M, N])


            scale_factor = float(D) / M

            inference_U = ed.MAP({U: qU}, \
                data={x: x_ph, gamma: qgamma})
            inference_gamma = ed.MAP({gamma: qgamma}, \
                data={x: x_ph, U: qU})

            inference_U.initialize(scale={x: scale_factor, U: scale_factor, gamma: scale_factor},
                                 var_list=qU_variables, n_iter=n_iter, optimizer=optimizer)
            inference_gamma.initialize(scale={x: scale_factor, U: scale_factor, gamma: scale_factor},
                                 var_list=qgamma_variables, optimizer=optimizer)

            tf.global_variables_initializer().run()

            loss = np.empty(inference_U.n_iter, dtype=np.float32)
            
            for j in range(inference_U.n_iter):
                x_batch, idx_batch = next_batch(test_data_tr, M)
                cau_batch = cau[idx_batch,:]
                weights_batch = weights[idx_batch,:]
                reconstr_cau_batch = reconstr_cau_test[idx_batch,:]

                x_batch = x_batch.todense().astype('int')
                cau_batch = cau_batch.todense()
                weights_batch = weights_batch.todense()
                sd_batch = 1./np.sqrt(1+weights_batch)

                info_dict = inference_U.update(feed_dict={x_ph: x_batch, idx_ph: idx_batch, \
                                              reconstr_cau_ph: reconstr_cau_batch, cau_ph: cau_batch, \
                                              sd_ph: sd_batch, V: V_out, beta0: beta0_out})
                inference_gamma.update(feed_dict={x_ph: x_batch, idx_ph: idx_batch, \
                                              reconstr_cau_ph: reconstr_cau_batch, cau_ph: cau_batch, \
                                              sd_ph: sd_batch, V: V_out, beta0: beta0_out})

                inference_U.print_progress(info_dict)

                loss[j] = info_dict["loss"]
                

            U_testout = tf.nn.softplus(qU_variables[0]).eval()
            gamma_testout = tf.nn.softplus(qgamma_variables[0]).eval()
            
                

            
            # np.savetxt(OUT_DATA_DIR + '/'+ out_filename + "_cas_testU.csv", U_testout)
            # np.savetxt(OUT_DATA_DIR + '/'+ out_filename + "_cas_testgamma.csv", gamma_testout)



            pred_train = sparse.csr_matrix(U_trainout.dot(V_out.T) + \
                                     np.multiply(np.repeat(gamma_trainout, reconstr_cau_train.shape[1], axis=1), \
                                                 reconstr_cau_train) + beta0_out)
            pred_vad = sparse.csr_matrix(U_vadout.dot(V_out.T) + \
                                     np.multiply(np.repeat(gamma_vadout, reconstr_cau_vad.shape[1], axis=1), \
                                                 reconstr_cau_vad) + beta0_out)
            pred_test = sparse.csr_matrix(U_testout.dot(V_out.T) + \
                                     np.multiply(np.repeat(gamma_testout, reconstr_cau_test.shape[1], axis=1), \
                                                 reconstr_cau_test) + beta0_out)

            pred_train = pred_train.todense()
            pred_vad = pred_vad.todense()
            pred_test = pred_test.todense()

            all_metric_holders = sg_eval_acc_metrics_update_i(all_metric_holders, i, \
                pred_train, pred_vad, pred_test, \
                train_data, \
                vad_data_tr, vad_data_te, \
                test_data_tr, test_data_te, \
                ks, thold)

    out_df = save_eval_metrics(all_metric_holders, model_name, outdims, all_params, ks)
        
    out_df.to_csv(OUT_DATA_DIR + '/res_'+ out_filename + ".csv")


