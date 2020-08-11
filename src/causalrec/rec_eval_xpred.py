import bottleneck as bn
import numpy as np

from scipy import sparse

from scipy.stats import norm, poisson


### this file is different from dawen's rec_eval. we only evaluate
#   items with explicit ratings in the heldout dataset. in dawen's
#   rec_eval, he assume items with zeros are present but irrelevant.

"""
All the data should be in the shape of (n_users, n_items)
All the latent factors should in the shape of (n_users/n_items, n_components)

1. train_data refers to the data that was used to train the model
2. heldout_data refers to the data that was used for evaluation (could be test
set or validation set)
3. vad_data refers to the data that should be excluded as validation set, which
should only be used when calculating test scores

"""

def user_idx_generator(n_users, batch_users):
    ''' helper function to generate the user index to loop through the dataset
    '''
    for start in range(0, n_users, batch_users):
        end = min(n_users, start + batch_users)
        yield slice(start, end)


def normalized_dcg_nonbinary_xpred(train_data, heldout_data, x_pred,
                             bias_V=None,
                             batch_users=5000,
                             heldout_rel=None, mu=None, vad_data=None,
                             agg=np.nanmean):
    n_users = train_data.shape[0]
    res = list()
    for user_idx in user_idx_generator(n_users, batch_users):
        res.append(NDCG_nbinary_batch_xpred(train_data, heldout_data, x_pred, bias_V,
                                      user_idx, heldout_rel=heldout_rel,
                                      mu=mu, vad_data=vad_data))
    ndcg = np.hstack(res)
    if callable(agg):
        return agg(ndcg)
    return ndcg

def NDCG_nbinary_batch_xpred(train_data, heldout_data, x_pred, bias_Eb, user_idx,
                       heldout_rel=None, mu=None, vad_data=None):
    '''
    normalized discounted cumulative gain
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users = user_idx.stop - user_idx.start
    n_items = train_data.shape[1]

    # make prediction
    # exclude examples from training and validation (if any)
    item_idx = np.zeros((batch_users, n_items), dtype=bool)
    item_idx[heldout_data[user_idx].nonzero()] = True
    item_idx[train_data[user_idx].nonzero()] = False
    if vad_data is not None:
        item_idx[vad_data[user_idx].nonzero()] = False

    X_pred = x_pred[user_idx].copy()
    X_pred[~item_idx] = -np.inf

    all_rank = np.argsort(np.argsort(-X_pred, axis=1), axis=1)
    # build the discount template
    #tp = np.hstack((1, 1. / np.log2(np.arange(2, n_items + 1))))
    tp = 1. / np.log2(np.arange(2, n_items + 2))
    all_disc = tp[all_rank]

    if heldout_rel is None:
        # if no relevance is provided, just use the data itself as relevance
        heldout_rel = heldout_data
    else:
        assert heldout_data.shape == heldout_rel.shape
        #assert heldout_data.data.size == heldout_rel.data.size
    heldout_batch = heldout_rel[user_idx]
    heldout_batch_coo = heldout_batch.tocoo()
    # this could be done more efficiently once the new scipy pull request on
    # element-wise sparse matrices multiplication is merged
    #disc = sparse.csr_matrix((heldout_batch.data *
    #                          all_disc[heldout_batch_coo.row,
    #                                   heldout_batch_coo.col],
    #                          (heldout_batch_coo.row, heldout_batch_coo.col)),
    #                         shape=all_disc.shape)
    disc = sparse.csr_matrix(((2**heldout_batch.data - 1)*
                              all_disc[heldout_batch_coo.row,
                                       heldout_batch_coo.col],
                              (heldout_batch_coo.row, heldout_batch_coo.col)),
                             shape=all_disc.shape)

    DCG = np.array(disc.sum(axis=1)).ravel()
    #IDCG = -np.array([(tp[:n] * np.sort(-heldout_batch.getrow(i).data)).sum()
    #                  for i, n in enumerate(heldout_batch.getnnz(axis=1))])
    IDCG = np.array([(tp[:n] * (2**(-np.sort(-heldout_batch.getrow(i).data)) - 1)).sum()
                      for i, n in enumerate(heldout_batch.getnnz(axis=1))])
    return DCG / IDCG


def recall_at_k_xpred(train_data, heldout_data, x_pred, bias_V=None, batch_users=5000,
                k=20, mu=None, thold=0, vad_data=None, agg=np.nanmean):
    n_users = train_data.shape[0]
    res = list()
    for user_idx in user_idx_generator(n_users, batch_users):
        res.append(recall_at_k_batch_xpred(train_data, heldout_data, x_pred, 
                                     bias_V, user_idx, k=k, thold=thold, 
                                     mu=mu, vad_data=vad_data))
    mn_recall = np.hstack(res)
    if callable(agg):
        return agg(mn_recall)
    return mn_recall



def recall_at_k_batch_xpred(train_data, heldout_data, x_pred,  bias_Eb, user_idx,
                      k=20, thold=0, normalize=True, mu=None, vad_data=None):
    batch_users = user_idx.stop - user_idx.start
    n_items = train_data.shape[1]

    # make prediction
    # exclude examples from training and validation (if any)
    item_idx = np.zeros((batch_users, n_items), dtype=bool)
    item_idx[heldout_data[user_idx].nonzero()] = True
    item_idx[train_data[user_idx].nonzero()] = False
    if vad_data is not None:
        item_idx[vad_data[user_idx].nonzero()] = False

    X_pred = x_pred[user_idx].copy()
    X_pred[~item_idx] = -np.inf

    idx = np.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_data[user_idx] > thold).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)
    recall = np.squeeze(tmp) / np.minimum(k, X_true_binary.sum(axis=1))

    return recall


def map_at_k_xpred(train_data, heldout_data, x_pred, bias_V=None, batch_users=5000,
             k=100, mu=None, thold=0, vad_data=None, agg=np.nanmean):

    n_users = train_data.shape[0]
    res = list()
    for user_idx in user_idx_generator(n_users, batch_users):
        res.append(MAP_at_k_batch_xpred(train_data, heldout_data, x_pred,
                                bias_V, user_idx, thold=thold,
                                k=k, mu=mu, vad_data=vad_data))
    map = np.hstack(res)
    if callable(agg):
        return agg(map)
    return map



def MAP_at_k_batch_xpred(train_data, heldout_data, x_pred, bias_Eb, user_idx,
                   mu=None, k=100, thold=0, vad_data=None):
    '''
    mean average precision@k
    '''
    batch_users = user_idx.stop - user_idx.start
    n_items = train_data.shape[1]

    # make prediction
    # exclude examples from training and validation (if any)
    item_idx = np.zeros((batch_users, n_items), dtype=bool)
    item_idx[heldout_data[user_idx].nonzero()] = True
    item_idx[train_data[user_idx].nonzero()] = False
    if vad_data is not None:
        item_idx[vad_data[user_idx].nonzero()] = False

    X_pred = x_pred[user_idx].copy()
    X_pred[~item_idx] = -np.inf

    idx_topk_part = np.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]

    aps = np.zeros(batch_users)
    for i, idx in enumerate(range(user_idx.start, user_idx.stop)):
        actual = (heldout_data[idx]>thold).nonzero()[1]
        if len(actual) > 0:
            predicted = idx_topk[i]
            aps[i] = apk(actual, predicted, k=k)
        else:
            aps[i] = np.nan
    return aps


def prec_at_k_xpred(train_data, heldout_data, x_pred, bias_V=None, batch_users=5000,
              k=20, mu=None, vad_data=None, thold=0, agg=np.nanmean):
    n_users = train_data.shape[0]
    res = list()
    for user_idx in user_idx_generator(n_users, batch_users):
        res.append(precision_at_k_batch_xpred(train_data, heldout_data, x_pred, 
                                        bias_V, user_idx, k=k, thold=thold,
                                        mu=mu, vad_data=vad_data))
    mn_prec = np.hstack(res)
    if callable(agg):
        return agg(mn_prec)
    return mn_prec


def precision_at_k_batch_xpred(train_data, heldout_data, x_pred, bias_Eb, user_idx,
                         k=20, normalize=False, thold=0, mu=None, vad_data=None):
    batch_users = user_idx.stop - user_idx.start
    n_items = train_data.shape[1]

    # make prediction
    # exclude examples from training and validation (if any)
    item_idx = np.zeros((batch_users, n_items), dtype=bool)
    item_idx[heldout_data[user_idx].nonzero()] = True
    item_idx[train_data[user_idx].nonzero()] = False
    if vad_data is not None:
        item_idx[vad_data[user_idx].nonzero()] = False

    X_pred = x_pred[user_idx].copy()
    X_pred[~item_idx] = -np.inf

    idx = np.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_data[user_idx] > thold).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)

    if normalize:
        precision = np.squeeze(tmp) / np.minimum(k, X_true_binary.sum(axis=1)).T
    else:
        precision = np.squeeze(tmp) / k
    return precision



def ric_rank_xpred(train_data, heldout_data, x_pred, bias_V=None, batch_users=5000,
             mu=None, vad_data=None, thold=0, agg=np.nanmean):
    n_users = train_data.shape[0]
    res = list()
    for user_idx in user_idx_generator(n_users, batch_users):
        res.append(mean_rrank_batch_xpred(train_data, heldout_data, x_pred, 
                                    bias_V, user_idx, thold=thold,
                                    mu=mu, vad_data=vad_data))
    mrrank = np.hstack(res)
    if callable(agg):
        return agg(mrrank)
    return mrrank


def mean_rrank_batch_xpred(train_data, heldout_data, x_pred, bias_Eb,
                     user_idx, k=5, mu=None, thold=0, vad_data=None):
    '''
    mean reciprocal rank: For each user, make predictions and rank for
    all the items. Then calculate the mean reciprocal rank for items that
    are in the held-out set.
    '''
    batch_users = user_idx.stop - user_idx.start
    n_items = train_data.shape[1]

    # make prediction
    # exclude examples from training and validation (if any)
    item_idx = np.zeros((batch_users, n_items), dtype=bool)
    item_idx[heldout_data[user_idx].nonzero()] = True
    item_idx[train_data[user_idx].nonzero()] = False
    if vad_data is not None:
        item_idx[vad_data[user_idx].nonzero()] = False

    X_pred = x_pred[user_idx].copy()
    X_pred[~item_idx] = -np.inf

    all_rrank = 1. / (np.argsort(np.argsort(-X_pred, axis=1), axis=1) + 1)
    X_true_binary = (heldout_data[user_idx] > thold).toarray()

    heldout_rrank = np.multiply(X_true_binary, all_rrank)
    return np.squeeze(heldout_rrank.sum(axis=1)) / np.squeeze(X_true_binary.sum(axis=1))


def mean_perc_rank_xpred(train_data, heldout_data, x_pred, bias_V=None,
                   batch_users=5000, mu=None, thold=0, vad_data=None):
    n_users = train_data.shape[0]
    mpr = 0
    for user_idx in user_idx_generator(n_users, batch_users):
        mpr += mean_perc_rank_batch_xpred(train_data, heldout_data, x_pred, 
                                    bias_V, user_idx, thold=thold,
                                    mu=mu, vad_data=vad_data)
    mpr /= heldout_data.sum()
    return mpr


def mean_perc_rank_batch_xpred(train_data, heldout_data, x_pred, bias_Eb, user_idx,
                         mu=None, thold=0, vad_data=None):
    '''
    mean percentile rank for a batch of users
    MPR of the full set is the sum of batch MPR's divided by the sum of all the
    feedbacks. (Eq. 8 in Hu et al.)
    This metric not necessarily constrains the data to be binary
    '''
    batch_users = user_idx.stop - user_idx.start
    n_items = train_data.shape[1]

    # make prediction
    # exclude examples from training and validation (if any)
    item_idx = np.zeros((batch_users, n_items), dtype=bool)
    item_idx[heldout_data[user_idx].nonzero()] = True
    item_idx[train_data[user_idx].nonzero()] = False
    if vad_data is not None:
        item_idx[vad_data[user_idx].nonzero()] = False

    X_pred = x_pred[user_idx].copy()
    X_pred[~item_idx] = -np.inf

    all_perc = np.argsort(np.argsort(-X_pred, axis=1), axis=1) / \
        np.isfinite(X_pred).sum(axis=1).astype(np.float32)
    perc_batch = (np.multiply(all_perc[(heldout_data[user_idx]>thold).nonzero()],
                  heldout_data[user_idx].data)).sum()

    return perc_batch

def avg_col_mse(data, x_pred):

    dat = data.copy() 
 
    idx_pos = (dat > 1e-16)
    neg_idx_pos = (dat <= 1e-6) & (dat > 0) # ignore missing data

    mse_pos = np.squeeze(np.array([np.nanmean(np.array(dat[np.squeeze(np.array(idx_pos[:,i])), i] - x_pred[np.squeeze(np.array(idx_pos[:,i])), i])**2) for i in range(dat.shape[1])]))
    mse_neg = np.squeeze(np.array([np.nanmean(np.array(dat[np.squeeze(np.array(neg_idx_pos[:,i])), i] - x_pred[np.squeeze(np.array(neg_idx_pos[:,i])), i])**2) for i in range(dat.shape[1])]))
    mse = np.squeeze(np.array([np.nanmean(np.array(dat[:, i] - x_pred[:, i])**2) for i in range(dat.shape[1])]))

    return np.nanmean(mse_pos), np.nanmean(mse_neg), np.nanmean(mse)


def avg_col_mae(data, x_pred):

    dat = data.copy() 
 
    idx_pos = (dat > 1e-16)
    neg_idx_pos = (dat <= 1e-6) & (dat > 0) # ignore missing data

    mae_pos = np.squeeze(np.array([np.nanmean(abs(np.array(dat[np.squeeze(np.array(idx_pos[:,i])), i] - x_pred[np.squeeze(np.array(idx_pos[:,i])), i]))) for i in range(dat.shape[1])]))
    mae_neg = np.squeeze(np.array([np.nanmean(abs(np.array(dat[np.squeeze(np.array(neg_idx_pos[:,i])), i] - x_pred[np.squeeze(np.array(neg_idx_pos[:,i])), i]))) for i in range(dat.shape[1])]))
    mae = np.squeeze(np.array([np.nanmean(abs(np.array(dat[:, i] - x_pred[:, i]))) for i in range(dat.shape[1])]))

    return np.nanmean(mae_pos), np.nanmean(mae_neg), np.nanmean(mae)


def mse(data, x_pred):
    
    dat = data.copy() 
 
    idx_pos = (dat > 1e-16)
    neg_idx_pos = (dat <= 1e-6) & (dat > 0) # ignore missing data
    
    mse_pos = np.squeeze(np.array(np.array(dat[idx_pos]-x_pred[idx_pos])**2))
    mse_neg = np.squeeze(np.array(np.array(dat[neg_idx_pos] - x_pred[neg_idx_pos])**2))
    
    mse = np.concatenate([mse_pos, mse_neg])
    
    return np.nanmean(mse_pos), np.nanmean(mse_neg), np.nanmean(mse)

def mae(data, x_pred):
    
    dat = data.copy() 
 
    idx_pos = (dat > 1e-16)
    neg_idx_pos = (dat <= 1e-6) & (dat > 0) # ignore missing data
    
    mae_pos = np.squeeze(np.array(abs(np.array(dat[idx_pos]-x_pred[idx_pos]))))
    mae_neg = np.squeeze(np.array(abs(np.array(dat[neg_idx_pos] - x_pred[neg_idx_pos]))))
    
    mae = np.concatenate([mae_pos, mae_neg])
    
    return np.nanmean(mae_pos), np.nanmean(mae_neg), np.nanmean(mae)

def log_cond_poisson_prob_metrics(data, x_pred):
    
    dat = data.copy() 
 
    idx_pos = (dat > 1e-6)
    neg_idx_pos = (dat <= 1e-6) & (dat > 0) # ignore missing data

    x_pred = np.maximum(x_pred, 1e-6 * np.ones_like(x_pred))
    
    log_prob_pos = np.squeeze(np.array(poisson.logpmf(dat[idx_pos].astype(int), x_pred[idx_pos])))
    log_prob_neg = np.squeeze(np.array(poisson.logpmf(dat[neg_idx_pos].astype(int), x_pred[neg_idx_pos])))
    
    log_plp = np.concatenate([log_prob_pos, log_prob_neg])
    
    return np.nanmean(log_prob_pos), np.nanmean(log_prob_neg), np.nanmean(log_plp)


def log_cond_normal_prob_metrics(data, x_pred):
    
    dat = data.copy() 
 
    idx_pos = (dat > 1e-16)
    neg_idx_pos = (dat <= 1e-6) & (dat > 0) # ignore missing data
    
    log_prob_pos = np.squeeze(np.array(norm.logpdf(dat[idx_pos], x_pred[idx_pos])))
    log_prob_neg = np.squeeze(np.array(norm.logpdf(dat[neg_idx_pos], x_pred[neg_idx_pos])))
    
    log_plp = np.concatenate([log_prob_pos, log_prob_neg])
    
    return np.nanmean(log_prob_pos), np.nanmean(log_prob_neg), np.nanmean(log_plp)


def normalized_dcg_at_k_nonbinary_xpred(train_data, heldout_data, x_pred,
                                  bias_V=None, batch_users=5000,
                                  heldout_rel=None, k=100,
                                  mu=None, vad_data=None, agg=np.nanmean):
    n_users = train_data.shape[0]
    res = list()
    for user_idx in user_idx_generator(n_users, batch_users):
        res.append(NDCG_nbinary_at_k_batch_xpred(train_data, heldout_data, x_pred, 
                                           bias_V, user_idx,
                                           heldout_rel=heldout_rel,
                                           k=k, mu=mu, vad_data=vad_data))
    ndcg = np.hstack(res)
    if callable(agg):
        return agg(ndcg)
    return ndcg




def NDCG_nbinary_at_k_batch_xpred(train_data, heldout_data, x_pred, bias_Eb,
                            user_idx, heldout_rel=None, k=500, mu=None,
                            vad_data=None):
    '''
    normalized discounted cumulative gain
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users = user_idx.stop - user_idx.start
    n_items = train_data.shape[1]

    # make prediction
    # exclude examples from training and validation (if any)
    item_idx = np.zeros((batch_users, n_items), dtype=bool)
    item_idx[heldout_data[user_idx].nonzero()] = True
    item_idx[train_data[user_idx].nonzero()] = False
    if vad_data is not None:
        item_idx[vad_data[user_idx].nonzero()] = False

    X_pred = x_pred[user_idx].copy()
    X_pred[~item_idx] = -np.inf

    idx_topk_part = np.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted list
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    #tp = np.hstack((1, 1. / np.log2(np.arange(2, k + 1))))
    tp = 1. / np.log2(np.arange(2, k + 2))

    if heldout_rel is None:
        # if no relevance is provided, just use the data itself as relevance
        heldout_rel = heldout_data
    else:
        assert heldout_data.shape == heldout_rel.shape
        #assert heldout_data.data.size == heldout_rel.data.size
    heldout_batch = heldout_rel[user_idx]
    #DCG = np.array(heldout_batch[np.arange(batch_users)[:, np.newaxis],
    #                             idx_topk].toarray() * tp).sum(axis=1)
    DCG = np.array((2**heldout_batch[np.arange(batch_users)[:, np.newaxis],
                                 idx_topk].toarray() - 1) * tp).sum(axis=1)

    # hack
    # if n = getnnz > k, tp[:n] -> (k,), data[:k] -> (k,)
    # else, tp[:n] -> (n,), data[:k] -> (n,)
    #IDCG = -np.array([(tp[:n] * np.sort(-heldout_batch.getrow(i).data[:k])).sum()
    #                  for i, n in enumerate(heldout_batch.getnnz(axis=1))])
    IDCG = np.array([(tp[:n] * (2**(-np.sort(-heldout_batch.getrow(i).data)[:k]) - 1)).sum()
                      for i, n in enumerate(heldout_batch.getnnz(axis=1))])
    return DCG / IDCG

# fixed a bug here. move the [:k] inside

## steal from https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
def apk(actual, predicted, k=100):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual: #and p not in predicted[:i]: # not necessary for us since we will not make duplicated recs
            num_hits += 1.0
            score += num_hits / (i+1.0)

    # we handle this part before making the function call
    #if not actual:
    #    return np.nan

    return score / min(len(actual), k)



