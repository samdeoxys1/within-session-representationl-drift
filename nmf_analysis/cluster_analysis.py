import numpy as np
import scipy
import matplotlib.pyplot as plt
import sys,os,copy,pdb,importlib
from importlib import reload
import pandas as pd
sys.path.append('/mnt/home/szheng/projects/util_code')
import data_prep_pyn as dpp     
from sklearn.cluster import KMeans
import scipy.spatial 
from scipy.spatial.distance import pdist, squareform
import nmf_analysis as na

'''
X_normed: mostly df, sometimes can do ndarray, n_times (samples) x n_neurons (features)
'''

######## EVALUATION, N CLUSTER SELECTION########
def cluster_get_dispersion(X_normed,n_clusters=2):
    km = KMeans(n_clusters=n_clusters)
    km.fit(X_normed)
    labels = km.labels_
    cluster_centers = km.cluster_centers_
    w_k = 0
    if isinstance(X_normed, pd.DataFrame):
        X_normed = X_normed.values
    for c in range(n_clusters):
        # compute dispersion
        X_in_clust = X_normed[labels==c]
        D = pdist(X_in_clust,metric='euclidean')
        nr = X_in_clust.shape[0]
        w_k += (D**2).sum() / (2*nr)
    w_k = np.log(w_k)
    return w_k
        

def sample_null(X_normed,n_repeats=10):
    X_normed = X_normed - X_normed.values.mean(axis=0,keepdims=True)
    u,d,vt=np.linalg.svd(X_normed)
    X_primed = X_normed.values.dot(vt.T)
    X_p_min = X_primed.min(axis=0,keepdims=True)
    X_p_max = X_primed.max(axis=0,keepdims=True)
    nsample,nfeat = X_normed.shape
    z_ratio = np.random.rand(n_repeats,nsample,nfeat)
    z_primed = (X_p_max - X_p_min) * z_ratio + X_p_min
    z = z_primed.dot(vt)
    return z

def get_gap_stat(X_normed_null,X_normed,n_clusters_l=[1,2]):
    '''
    X_normed: df, N_sample x N_features
    X_normed_null: N_repeats x N_sample x N_features
    ===
    k_hat, : the optimal n_cluster
    test_stats, : gap(t) - (gap(t+1) - s_k(t+1))
    gap_k, : the gap statistics at each n_cluster=k, i.e. E[w_null_k] - E[w_k], NB the w here already contains log, different from the paper
    w_k_l, : log(sum of dispersion), each dispersion is the sum of pairwise distance / 2N_in_cluster
    w_null_k_b_l, : log(sum of dispersion) for each null data b
    s_k: monte carlo standard error, obtained from the std of w_null_k_b_l across the b dimension
    '''
    n_repeats = len(X_normed_null)
    w_null_k_b_l = np.zeros((n_repeats,len(n_clusters_l))) # B x (K to be tested)
    w_k_l = np.zeros((len(n_clusters_l),))
    for ii,k in enumerate(n_clusters_l):
        w_k = cluster_get_dispersion(X_normed,n_clusters=k)
        w_k_l[ii] = w_k
        for jj,Xn in enumerate(X_normed_null):
            w_null = cluster_get_dispersion(Xn,n_clusters=k)
            w_null_k_b_l[jj,ii] = w_null
    
    sd_k = np.std(w_null_k_b_l,axis=0) # standard deviation across null
    s_k = np.sqrt(1+1/n_repeats) * sd_k # standard error
    gap_k = w_null_k_b_l.mean(axis=0) - w_k_l
    test_stats = gap_k[:-1] - (gap_k[1:]-s_k[1:])
    k_hat = np.nonzero(test_stats >= 0)[0]
    if len(k_hat) > 0:
        k_hat = k_hat[0]
        k_hat =n_clusters_l[k_hat]
    return k_hat,test_stats, gap_k, w_k_l, w_null_k_b_l,s_k

######### MAIN stuff ########        
def cluster_and_sort(X_normed, n_clust):
    '''

    ===


    '''
    # model fitting
    kmeans=KMeans(n_clusters=n_clust).fit(X_normed)
    labels=kmeans.labels_

    # post tuning info
    fitted_mean = kmeans.cluster_centers_
    fitted_mean = pd.DataFrame(fitted_mean,columns=X_normed.columns).T
    fitted_info = get_cluster_tuning(fitted_mean)
    
    # sort neuron
    ind,_,_=na.sort_factors(fitted_info.iloc[:,:-1].values) # assuming fitted_info: [tuning_i, sum_activity] # sort factors based on tuning index
    uid_sorted=fitted_info.index[ind]
    fitted_mean = fitted_mean.loc[uid_sorted]
    fitted_info = fitted_info.loc[uid_sorted]

    # sort time
    labels_s=pd.Series(labels,index=X_normed.index)
    labels_s=labels_s.reset_index()
    labels_s = labels_s.sort_values([0,'index'])
    labels_s=labels_s.rename({0:'label'}, axis=1)

    res = {'uid_sorted':uid_sorted,'fitted_mean':fitted_mean, 'fitted_info':fitted_info, 'labels':labels,'labels_sorted':labels_s}

    return res





    
def get_cluster_tuning(fitted_mean):
    '''
    fitted_mean: n_neurons x n_factors
    (activation in cluster i - (sum of activation in the rest)) /  (sum of activation in all clusters)

    ===
    fitted_info: [tuning_i, sum_activity]
    '''
    ep = 1e-10
    col_sum = fitted_mean.sum(axis=1)
    tuning_i_d ={}
    for fac_ind,row  in fitted_mean.T.iterrows():
        tuning_i = (row -fitted_mean.loc[:,fitted_mean.columns!=fac_ind].sum(axis=1)) / (col_sum + ep)
        tuning_i_d[fac_ind] = tuning_i
    tuning_i_d = pd.DataFrame(tuning_i_d)
    fitted_info = tuning_i_d
    fitted_info.columns = [f'tuning_{i}' for i in fitted_info.columns]
    fitted_info['sum_activity'] = col_sum
    return fitted_info

    