import sys,os,pickle,copy,pdb
sys.path.append('/mnt/home/szheng/projects/seq_detection2/code')
sys.path.append('/mnt/home/szheng/projects/util_code')
from time import thread_time_ns
import pandas as pd
import numpy as np
import scipy
import data_prep_new as dpn
import place_cell_analysis as pa
import plot_helper as ph
from importlib import reload
import preprocess as prep
import nmf_analysis as na
from sklearn.decomposition import NMF, non_negative_factorization
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scipy.signal as ss
import tqdm

import ruptures as rpt

def test_contiguity(signal,n_shuffle=200,sig_thresh=0.05,n_change_pts=1):
    '''
    idea: if there are contiguous trials with different mean rates, then 
    the error given by a change point detection model (with one change point) 
    should be significantly lower than the errors with the data shuffled in trials
    '''
    model = rpt.Dynp(model='l2',jump=1)
    c = model.fit(signal)
    change_pts=np.array(model.predict(n_change_pts))
    error = c.cost.sum_of_costs(list(change_pts)) # list is important here!
    err_l = []
    signal_all = np.tile(signal,(n_shuffle,1))
    rng = np.random.default_rng(123)
    signal_all = rng.permuted(signal_all,axis=1)
    for ii in range(n_shuffle):
        model = rpt.Dynp(model='l2',jump=1,min_size=1)
        # perm = np.arange(signal.shape[0])
        # np.random.shuffle(perm)
        # c = model.fit(signal[perm])
        c = model.fit(signal_all[ii])
        change_pts=np.array(model.predict(n_change_pts))
        l = c.cost.sum_of_costs(list(change_pts))
        err_l.append(l)
    err_l = np.array(err_l)
    # p = np.quantile(err_l,error)
    p = np.count_nonzero(error > err_l) / len(err_l)
    issig = p < sig_thresh
    return error, err_l, p, issig

from pandarallel import pandarallel
def test_contiguity_allrows(X,n_shuffle=200,sig_thresh=0.05,n_change_pts=1):
    pandarallel.initialize(use_memory_fs=False,progress_bar=True)
    row_func = lambda x:test_contiguity(x.values,n_shuffle=n_shuffle,sig_thresh=sig_thresh,n_change_pts=n_change_pts)[2]
    res = X.parallel_apply(row_func,axis=1)
    return res

def test_contiguity_multiple_nchangepts(signal,n_shuffle=200,sig_thresh=0.05,n_change_pts_l=[1,2,3,4]):
    '''
    sweep through multiple changes points and do the shuffle test
    the hope is to find the optimal number of change points, which will have the lowest p value
    '''
    p_l = []
    error_l = []
    issig_l = []
    for n_change_pts in n_change_pts_l:
        error,_,p,issig = test_contiguity(signal,n_shuffle,sig_thresh,n_change_pts=n_change_pts)
        p_l.append(p)
        error_l.append(error)
        issig_l.append(issig)
    ind = np.argmin(p_l)
    opt_n = n_change_pts_l[ind]
    opt_p = p_l[ind]
    opt_issig = issig_l[ind]
    opt_error = error_l[ind]
    return opt_n, opt_error, opt_p, opt_issig, np.array(error_l), np.array(p_l), np.array(issig_l)


def test_contiguity_independent_multidim(signal,n_shuffle=100,sig_thresh=0.05,n_change_pts_l=[1,2,3,4]):
    nsamples,ndim = signal.shape
    error_all_dim = []
    err_l_all_dim = []
    p_all_dim = []
    issig_all_dim = []
    opt_n_all_dim = []
    opt_error_all_dim = []
    opt_p_all_dim = []
    opt_issig_all_dim = []
    for d in tqdm.tqdm(range(ndim),desc="dim"):
        # error,err_l,p,issig = test_contiguity(signal[:,d],n_shuffle=n_shuffle,sig_thresh=sig_thresh)
        opt_n, opt_error, opt_p, opt_issig, error_l, p_l, issig_l = test_contiguity_multiple_nchangepts(signal[:,d],n_shuffle,sig_thresh,n_change_pts_l=n_change_pts_l)
        error_all_dim.append(error_l)
        # err_l_all_dim.append(err_l)
        p_all_dim.append(p_l)
        issig_all_dim.append(issig_l)
        opt_n_all_dim.append(opt_n)
        opt_p_all_dim.append(opt_p)
        opt_error_all_dim.append(opt_error)
        opt_issig_all_dim.append(opt_issig)
    columns = pd.MultiIndex.from_product([['p','error','issig'],n_change_pts_l])
    instability_df = pd.DataFrame([],columns=columns)
    instability_df.loc[:,('p',n_change_pts_l)] = np.array(p_all_dim)
    instability_df.loc[:,('error',n_change_pts_l)] = np.array(error_all_dim)
    instability_df.loc[:,('issig',n_change_pts_l)] = np.array(issig_all_dim)
    instability_df['opt_n'] = np.array(opt_n_all_dim)
    instability_df['opt_p'] = np.array(opt_p_all_dim)
    instability_df['opt_error'] = np.array(opt_error_all_dim)
    instability_df['opt_issig'] = np.array(opt_issig_all_dim)


    return instability_df




def test_w(X_normed_restacked_df,w_original,H_original,n_shuffle=500):
    '''
    test the significance of each w, by shuffling the trial indices for each neuron, refit using the fitted H
    this is still a single neuron level measure, how much a single neuron matches a population level pattern?

    X_normed_restacked_df: (n_neurons x n_posbins) x n_trials, df
    w_original: (n_neurons x n_posbins) x n_compo, df
    H_original: n_compo x n_trials
    ===
    p_val_df: n_neurons x n_compo, df
    w_l: n_shuffle x (n_neurons x n_posbins) x n_compo, np
    '''
    X_normed_restacked_df
    n_compo = H_original.shape[0]
    
    w_l=[]
    for i in range(n_shuffle):
        X_shuffled= np.random.permutation(X_normed_restacked_df.values.T).T
        w,h,niter=non_negative_factorization(X_shuffled,n_components=n_compo,H=H_original,update_H=False)
        w_l.append(w)
    
    w_l = np.array(w_l)
    p_val = (w_original.values < w_l).sum(axis=0) / w_l.shape[0]
    p_val_df = pd.DataFrame(p_val,index=w_original.index)
    return p_val_df, w_l



def test_co_variation(X_normed_restacked_df,n_compo,n_shuffle=100):
    '''
    circular shuffle each neuron to preserve autocorrelation
    check the reconstruction error
    '''
    
    ntrials = X_normed_restacked_df.shape[1]
    err_original = na.nmf_get_error(X_normed_restacked_df.values,n_compo)
    err_shuffle_l =[]
    for i in range(n_shuffle):
        # circular shuffle
        X_circularly_shuffled = X_normed_restacked_df.groupby(level=0).apply(lambda x:pd.DataFrame(np.roll(x.values,np.random.randint(ntrials),axis=1),index=x.index,columns=x.columns))
        err = na.nmf_get_error(X_circularly_shuffled.values,n_compo)
        err_shuffle_l.append(err)
    err_shuffle_l = np.array(err_shuffle_l)
    fig,ax=plt.subplots()
    ax.hist(err_shuffle_l)
    ax.axvline(err_original)
    pval = (err_original > err_shuffle_l).sum() / n_shuffle

    return err_shuffle_l, pval