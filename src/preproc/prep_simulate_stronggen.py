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


    OUT_DATA_DIR = '../dat/proc/simulation'+str(i)+'_sg'
    OUT_AFIT_DIR = '../out/simulation'+str(i)+'_sg_Afit'
    OUT_YFIT_DIR = '../out/simulation'+str(i)+'_sg_Yfit'

    if not os.path.exists(OUT_DATA_DIR):
        os.makedirs(OUT_DATA_DIR)
    if not os.path.exists(OUT_AFIT_DIR):
        os.makedirs(OUT_AFIT_DIR)
    if not os.path.exists(OUT_YFIT_DIR):
        os.makedirs(OUT_YFIT_DIR)    




    raw_data = pd.read_csv(os.path.join(DATA_DIR, 'sim_data_'+str(i)+'_train_obs.csv'), sep=",", header=None, 
                           names=['userId', 'songId', 'rating'],engine="python")
    test_data = pd.read_csv(os.path.join(DATA_DIR, 'sim_data_'+str(i)+'_test.csv'), sep=",", header=None, 
                           names=['userId', 'songId', 'rating'],engine="python")


    # In[7]:


    raw_data.head(), raw_data.shape


    # In[8]:


    test_data.head(), test_data.shape


    # In[9]:


    def split_train_test_proportion(data, uid, test_prop=0.5, random_seed=0, n_items_thresh=5):
        data_grouped_by_user = data.groupby(uid)
        tr_list, te_list = list(), list()

        np.random.seed(random_seed)

        for u, (_, group) in enumerate(data_grouped_by_user):
            n_items_u = len(group)
            
            if n_items_u >= n_items_thresh:
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


    user_activity = get_count(raw_data, 'userId')


    # In[12]:


    unique_uid = user_activity.index


    # In[13]:


    np.random.seed(98765)
    idx_perm = np.random.permutation(unique_uid.size)
    unique_uid = unique_uid[idx_perm]


    # In[14]:


    unique_uid.size


    # In[15]:


    n_users = unique_uid.size

    tr_users = unique_uid[:(n_users - int(0.4*n_users))]
    vd_users = unique_uid[(n_users - int(0.4*n_users)): (n_users - int(0.2*n_users))]
    te_users = unique_uid[(n_users - int(0.2*n_users)):]


    # In[16]:


    train_plays = raw_data.loc[raw_data['userId'].isin(tr_users)]
    unique_sid = pd.unique(train_plays['songId'])


    # In[17]:


    song2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    user2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))


    # In[18]:


    if not os.path.exists(OUT_DATA_DIR):
        os.makedirs(OUT_DATA_DIR)

    with open(os.path.join(OUT_DATA_DIR, 'unique_uid.txt'), 'w') as f:
        for uid in unique_uid:
            f.write('%s\n' % uid)

    with open(os.path.join(OUT_DATA_DIR, 'unique_sid.txt'), 'w') as f:
        for sid in unique_sid:
            f.write('%s\n' % sid)


    # In[19]:


    vad_plays = raw_data.loc[raw_data['userId'].isin(vd_users)]
    vad_plays = vad_plays.loc[vad_plays['songId'].isin(unique_sid)]


    # In[20]:


    vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays, 'userId', test_prop=0.5, random_seed=13579)


    # In[21]:


    test_plays = raw_data.loc[raw_data['userId'].isin(te_users)]
    test_plays = test_plays.loc[test_plays['songId'].isin(unique_sid)]


    # In[22]:


    test_plays_tr, test_plays_te = split_train_test_proportion(test_plays, 'userId', test_prop=0.5, random_seed=13579)


    # In[23]:


    rand_test_plays = test_data.loc[raw_data['userId'].isin(te_users)]
    rand_test_plays = test_plays.loc[test_plays['songId'].isin(unique_sid)]


    # In[24]:


    rand_test_plays_tr, rand_test_plays_te = split_train_test_proportion(rand_test_plays, 'userId', test_prop=0.5, random_seed=13579)


    # In[25]:


    print(len(train_plays), len(vad_plays), len(test_plays))


    # In[26]:


    print(len(vad_plays_tr), len(vad_plays_te))


    # In[27]:


    print(len(test_plays_tr), len(test_plays_te))


    # In[28]:


    print(len(rand_test_plays_tr), len(rand_test_plays_te))


    # # Turn userId and songId to 0-based index

    # In[29]:


    def numerize(tp):
        uid = list(map(lambda x: user2id[x], tp['userId']))
        sid = list(map(lambda x: song2id[x], tp['songId']))
        tp.loc[:, 'uid'] = uid
        tp.loc[:, 'sid'] = sid
        return tp[['uid', 'sid', 'rating']]


    # In[30]:


    train_data = numerize(train_plays)
    train_data.to_csv(os.path.join(OUT_DATA_DIR, 'train.csv'), index=False)


    # In[31]:


    vad_data_tr = numerize(vad_plays_tr)
    vad_data_tr.to_csv(os.path.join(OUT_DATA_DIR, 'validation_tr.csv'), index=False)


    # In[32]:


    vad_data_te = numerize(vad_plays_te)
    vad_data_te.to_csv(os.path.join(OUT_DATA_DIR, 'validation_te.csv'), index=False)


    # In[33]:


    test_data_tr = numerize(test_plays_tr)
    test_data_tr.to_csv(os.path.join(OUT_DATA_DIR, 'obs_test_tr.csv'), index=False)


    # In[34]:


    test_data_te = numerize(test_plays_te)
    test_data_te.to_csv(os.path.join(OUT_DATA_DIR, 'obs_test_te.csv'), index=False)


    # In[35]:


    rand_test_data_tr = numerize(rand_test_plays_tr)
    rand_test_data_tr.to_csv(os.path.join(OUT_DATA_DIR, 'test_tr.csv'), index=False)


    # In[36]:


    rand_test_data_te = numerize(rand_test_plays_te)
    rand_test_data_te.to_csv(os.path.join(OUT_DATA_DIR, 'test_te.csv'), index=False)

