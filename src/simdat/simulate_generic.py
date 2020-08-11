
# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy.random as npr
from sklearn.cluster import KMeans
from scipy.stats import invgamma

from scipy import sparse, stats

get_ipython().magic(u'matplotlib inline')

plt.style.use('ggplot')


# In[2]:


import seaborn as sns
sns.set_style("white")
sns.set_context("paper")

color_names = ["red",
               "windows blue",
               "medium green",
               "dusty purple",
               "orange",
               "amber",
               "clay",
               "pink",
               "greyish",
               "light cyan",
               "steel blue",
               "forest green",
               "pastel purple",
               "mint",
               "salmon",
               "dark brown"]
colors = sns.xkcd_palette(color_names)


# # simulate data

# In[3]:


n_users, n_items = 1000, 1000
K = 10


# In[4]:


def simulate_data(data_idx, corr, gamma):
    OUT_DATA_DIR = '../dat/raw/simulation'+str(data_idx)
    if not os.path.exists(OUT_DATA_DIR):
        os.makedirs(OUT_DATA_DIR)
    theta_A = npr.gamma(0.3, scale=0.3, size=(n_users, K))
    beta = npr.gamma(0.3, scale=0.3, size=(n_items, K))
    A = np.minimum(npr.poisson(theta_A.dot(beta.T)), 1)
    theta_Y = corr * theta_A + (1 - corr) * npr.gamma(0.3, scale=0.3, size=(n_users, K))
    y = npr.poisson(theta_Y.dot(beta.T) + gamma * theta_A.dot(beta.T))
    y = np.minimum(y+1, 5)
    y_obs = np.multiply(A, y)
    plt.hist(np.concatenate(y), bins=5)
    plt.hist(np.concatenate(y_obs), bins=5)
    print("large y_obs_prop", np.sum(y_obs==5)/np.sum(y_obs>0))
    print("large y_prop", np.sum(y==5)/np.sum(y>0))
    print("nonzero A_prop", np.sum(A>0)/np.sum(A>-1))
    y = sparse.coo_matrix(y)
    # dat = np.column_stack([y.row, y.col, y.data])
    A = sparse.coo_matrix(A)
    # Adat = np.column_stack([A.row, A.col, A.data])
    ydf = pd.DataFrame({'uid': y.row, 'sid': y.col, 'rating':y.data})
    Adf = pd.DataFrame({'uid': A.row, 'sid': A.col, 'obs':A.data})
    train_prop = 0.95
    idx = npr.permutation(np.arange(ydf.shape[0]))
    train_idx, test_idx = idx[:int(train_prop*ydf.shape[0])], idx[int(train_prop*ydf.shape[0]):]
    y_test = ydf.iloc[test_idx, :]
    y_train_all = ydf.iloc[train_idx, :]
    y_train_obs = Adf.merge(ydf.iloc[train_idx, :], on=['uid','sid'])
    assert np.sum(y_train_obs['obs']==1) == y_train_obs.shape[0]
    y_train_obs = y_train_obs[['uid','sid','rating']]
    print(y_train_obs.shape, y_test.shape)
    ydf.to_csv(os.path.join(OUT_DATA_DIR, 'sim_data_'+str(i)+'_full.csv'),                header=False, index=False)
    Adf.to_csv(os.path.join(OUT_DATA_DIR, 'sim_data_'+str(i)+'_missingness.csv'),                header=False, index=False)
    y_train_all.to_csv(os.path.join(OUT_DATA_DIR, 'sim_data_'+str(i)+'_train_all.csv'),                header=False, index=False)
    y_train_obs.to_csv(os.path.join(OUT_DATA_DIR, 'sim_data_'+str(i)+'_train_obs.csv'),                header=False, index=False)
    y_test.to_csv(os.path.join(OUT_DATA_DIR, 'sim_data_'+str(i)+'_test.csv'),                header=False, index=False)


# In[5]:


i = 0
params_list = []
for gamma in np.arange(0, 5.5, 0.5):
    for corr in [0.6]:
        print([i, gamma, corr])
        simulate_data(i, corr, gamma)
        i += 1
        params_list.append([i, gamma, corr])
        
for gamma in [3]:
    for corr in np.arange(0, 1.1, 0.1):
        print([i, gamma, corr])
        simulate_data(i, corr, gamma)
        i += 1
        params_list.append([i, gamma, corr])   
        


# In[9]:


params_list = np.array(params_list)
params_list = pd.DataFrame(params_list, columns = ['index', 'conf_strength', 'conf_corr'])
params_list.to_csv('../dat/raw/simulation_params_list.csv')


# In[10]:


print(params_list)

