#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy.random as npr
from sklearn.cluster import KMeans
from scipy.stats import invgamma

from scipy import sparse, stats


# plt.style.use('ggplot')


# In[2]:


# import seaborn as sns
# sns.set_style("white")
# sns.set_context("paper")

# color_names = ["red",
#                "windows blue",
#                "medium green",
#                "dusty purple",
#                "orange",
#                "amber",
#                "clay",
#                "pink",
#                "greyish",
#                "light cyan",
#                "steel blue",
#                "forest green",
#                "pastel purple",
#                "mint",
#                "salmon",
#                "dark brown"]
# colors = sns.xkcd_palette(color_names)


# In[3]:

for i in range(22):

    DATA_DIR = '../dat/raw/simulation'+str(i)


    # In[4]:


    OUT_DATA_DIR = '../dat/proc/simulation'+str(i)+'_wg'
    OUT_AFIT_DIR = '../out/simulation'+str(i)+'_wg_Afit'
    OUT_YFIT_DIR = '../out/simulation'+str(i)+'_wg_Yfit'

    if not os.path.exists(OUT_DATA_DIR):
        os.makedirs(OUT_DATA_DIR)
    if not os.path.exists(OUT_AFIT_DIR):
        os.makedirs(OUT_AFIT_DIR)
    if not os.path.exists(OUT_YFIT_DIR):
        os.makedirs(OUT_YFIT_DIR)    



    tr_vd_data = pd.read_csv(os.path.join(DATA_DIR, 'sim_data_'+str(i)+'_train_obs.csv'), sep=",", header=None, 
                           names=['userId', 'songId', 'rating'],engine="python")
    test_data = pd.read_csv(os.path.join(DATA_DIR, 'sim_data_'+str(i)+'_test.csv'), sep=",", header=None, 
                           names=['userId', 'songId', 'rating'],engine="python")


    # In[7]:


    tr_vd_data.head(), tr_vd_data.shape


    # In[8]:


    test_data.head(), test_data.shape


    # In[9]:


    def split_train_test_proportion(data, uid, test_prop=0.5, random_seed=0):
        data_grouped_by_user = data.groupby(uid)
        tr_list, te_list = list(), list()

        np.random.seed(random_seed)

        for u, (_, group) in enumerate(data_grouped_by_user):
            n_items_u = len(group)

            if n_items_u >= 5:
                idx = np.zeros(n_items_u, dtype='bool')
                idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True

                tr_list.append(group[np.logical_not(idx)])
                te_list.append(group[idx])
            else:
                tr_list.append(group)

            if u % 5000 == 0:
                print("%d users sampled" % u)
                sys.stdout.flush()

        data_tr = pd.concat(tr_list)
        data_te = pd.concat(te_list)
        
        return data_tr, data_te


    # In[10]:


    def get_count(tp, id):
        playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
        count = playcount_groupbyid.size()
        return count


    # In[11]:


    user_activity = get_count(tr_vd_data, 'userId')
    item_popularity = get_count(tr_vd_data, 'songId')


    # In[12]:


    unique_uid = user_activity.index
    unique_sid = item_popularity.index


    # In[13]:


    n_users = len(unique_uid)
    n_items = len(unique_sid)


    # In[14]:


    n_users, n_items


    # In[15]:


    song2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    user2id = dict((uid, i) for (i, uid) in enumerate(unique_uid))


    # In[16]:


    # for the test set, only keep the users/items from the training set

    test_data = test_data.loc[test_data['userId'].isin(unique_uid)]
    test_data = test_data.loc[test_data['songId'].isin(unique_sid)]


    # In[17]:


    with open(os.path.join(OUT_DATA_DIR, 'unique_uid.txt'), 'w') as f:
        for uid in unique_uid:
            f.write('%s\n' % uid)

    with open(os.path.join(OUT_DATA_DIR, 'unique_sid.txt'), 'w') as f:
        for sid in unique_sid:
            f.write('%s\n' % sid)


    # # Turn userId and songId to 0-based index

    # In[18]:


    def numerize(tp):
        uid = list(map(lambda x: user2id[x], tp['userId']))
        sid = list(map(lambda x: song2id[x], tp['songId']))
        tp.loc[:, 'uid'] = uid
        tp.loc[:, 'sid'] = sid
        return tp[['uid', 'sid', 'rating']]


    # In[19]:


    tr_vd_data = numerize(tr_vd_data)
    test_data = numerize(test_data)


    # In[20]:


    train_data, vad_data = split_train_test_proportion(tr_vd_data, 'uid', test_prop=0.6, random_seed=12345)
    obs_test_data, vad_data = split_train_test_proportion(vad_data, 'uid', test_prop=0.5, random_seed=12345)


    # In[21]:


    print("There are total of %d unique users in the training set and %d unique users in the entire dataset" % (len(pd.unique(train_data['uid'])), len(unique_uid)))


    # In[22]:


    print("There are total of %d unique items in the training set and %d unique items in the entire dataset" % (len(pd.unique(train_data['sid'])), len(unique_sid)))


    # In[23]:


    def move_to_fill(part_data_1, part_data_2, unique_id, key):
        # move the data from part_data_2 to part_data_1 so that part_data_1 has the same number of unique "key" as unique_id
        part_id = set(pd.unique(part_data_1[key]))
        
        left_id = list()
        for i, _id in enumerate(unique_id):
            if _id not in part_id:
                left_id.append(_id)
                
        move_idx = part_data_2[key].isin(left_id)
        part_data_1 = part_data_1.append(part_data_2[move_idx])
        part_data_2 = part_data_2[~move_idx]
        return part_data_1, part_data_2


    # In[24]:


    train_data, vad_data = move_to_fill(train_data, vad_data, np.arange(n_items), 'sid')
    train_data, obs_test_data = move_to_fill(train_data, obs_test_data, np.arange(n_items), 'sid')


    # In[25]:


    print("There are total of %d unique items in the training set and %d unique items in the entire dataset" % (len(pd.unique(train_data['sid'])), len(unique_sid)))


    # In[26]:


    train_data.to_csv(os.path.join(OUT_DATA_DIR, 'train.csv'), index=False)
    vad_data.to_csv(os.path.join(OUT_DATA_DIR, 'validation.csv'), index=False)
    tr_vd_data.to_csv(os.path.join(OUT_DATA_DIR, 'train_full.csv'), index=False)


    # In[27]:


    obs_test_data.to_csv(os.path.join(OUT_DATA_DIR, 'obs_test_full.csv'), index=False)
    test_data.to_csv(os.path.join(OUT_DATA_DIR, 'test_full.csv'), index=False)


    # # Load the data

    # In[28]:


    unique_uid = list()
    with open(os.path.join(OUT_DATA_DIR, 'unique_uid.txt'), 'r') as f:
        for line in f:
            unique_uid.append(line.strip())
        
    unique_sid = list()
    with open(os.path.join(OUT_DATA_DIR, 'unique_sid.txt'), 'r') as f:
        for line in f:
            unique_sid.append(line.strip())


    # In[29]:


    n_items = len(unique_sid)
    n_users = len(unique_uid)

    print(n_users, n_items)


    # In[30]:


    def load_data(csv_file, shape=(n_users, n_items)):
        tp = pd.read_csv(csv_file)
        rows, cols, vals = np.array(tp['uid']), np.array(tp['sid']), np.array(tp['rating']) 
        data = sparse.csr_matrix((vals, (rows, cols)), dtype=np.float32, shape=shape)
        return data


    # In[31]:


    def binarize_rating(data, cutoff=3, eps=1e-6):
        data.data[data.data < cutoff] = eps   # small value so that it will not be treated as 0 in sparse matrix 
        data.data[data.data >= cutoff] = 1
        return data


    # In[32]:


    def exp_to_imp(data, cutoff=0.5):
        # turn data (explicit feedback) to implict with cutoff
        data_imp = data.copy()
        data_imp.data[data_imp.data < cutoff] = 0
        data_imp.data[data_imp.data >= cutoff] = 1
        data_imp.data = data_imp.data.astype('int32')
        data_imp.eliminate_zeros()
        return data_imp


    # In[33]:


    def binarize_spmat(spmat):
        spmat_binary = spmat.copy()
        spmat_binary.data = np.ones_like(spmat_binary.data)
        return spmat_binary


    # In[34]:


    def subsample_negatives(data, full_data=None, random_state=0, verbose=False):
        # roughly subsample the same number of negative as the positive in `data` for each user
        # `full_data` is all the positives we *are supposed to* know
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


    # In[35]:


    train_data = load_data(os.path.join(OUT_DATA_DIR, 'train_full.csv'))


    # In[36]:


    # bins = np.histogram(train_data.data, bins=5)[0]
    # plt.bar(np.arange(1, 6), bins)
    # pass


    # In[37]:


    test_data = load_data(os.path.join(OUT_DATA_DIR, 'test_full.csv'))
    vad_data = load_data(os.path.join(OUT_DATA_DIR, 'validation.csv'))


    # In[38]:


    # bins = np.histogram(test_data.data, bins=5)[0]
    # plt.bar(np.arange(1, 6), bins)
    # pass


    # In[39]:


    # bins = np.histogram(vad_data.data, bins=5)[0]
    # plt.bar(np.arange(1, 6), bins)
    # pass

