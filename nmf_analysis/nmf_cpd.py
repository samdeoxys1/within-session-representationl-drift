'''
NMF + Change point detection
constraining the H to be piecewise constant
'''

import sys,os,pickle,pdb,copy,itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ruptures as rpt
import scipy
import sklearn

from sklearn.decomposition import NMF

from ruptures.base import BaseCost

######## constant H rank1 nmf updates ########

def rank1_constant_h_update_h(r1,w1):
    '''
    r1: approx rank 1 (resid after subtracting out the rest of the wh outer products); 
    r1: n_samples x n_features
    '''
    n_sample,n_feature = r1.shape
    h = np.sum(w1.T.dot(r1)) / (np.sum(w1**2) * n_feature)
    h1 = np.ones((1,n_feature)) * np.maximum(h,0)
    return h1

def random_init_constant_h_rank1(X,n_compo):
    '''
    X: n_samples x n_features
    '''
    avg = np.sqrt(X.mean() / n_compo)
    H = np.ones((1,X.shape[1])) * avg 
    W = avg * np.random.normal(size=(X.shape[0], 1)).astype(
    X.dtype, copy=False
    )
    np.abs(H, out=H)
    np.abs(W, out=W)
    return W,H
        
    
######## piece-wise constant rank1 nmf ###########

# use rupture (change point detection) to update h
# here is a custom cost for rupture
class nmf_cost(BaseCost):
    """Custom cost for exponential signals."""
    # The 2 following attributes must be specified for compatibility.
    model = ""
    min_size = 2
            
    def __init__(self,w):
        super().__init__()
        self.w = w
        
    def fit(self, signal):
        """signal: n_feats x n_samples, important!! transpose to the nmf convention"""
        self.signal = signal
        return self

    def error(self, start, end):
        """Return the approximation cost on the segment [start:end].
        Args:
            start (int): start of the segment
            end (int): end of the segment
        Returns:
            float: segment cost
        """
        sub = self.signal[start:end].T
        h1=rank1_constant_h_update_h(sub,self.w)
        err = np.linalg.norm(sub-self.w.dot(h1),ord='fro')**2 # previously bug because i had w1 instead of self.w. Importnat to check variable scope!; important to square!!! Otherwise the least square problem and the cpd problem are no longer equivalent
        return err

def rank1_pwc_h_update_h_given_bkps(r1,w1,bkps):
    '''
    bkps includes 0 and T, unlike from rpt; so need to be prepped
    reconstruct h given change points, as well as the residual data matrix and
    one w factor:
    r1: n_sample x n_feature
    w1: n_sample x 1
    '''
    T = r1.shape[1]
    h1 = np.zeros((1,T))
    for tt in range(len(bkps)-1):
        h_part=rank1_constant_h_update_h(r1[:,bkps[tt]:bkps[tt+1]],w1)
        h1[:,bkps[tt]:bkps[tt+1]] = h_part
    return h1


def rank1_pwc_h_update_h_full(r1,w1,n_bkps=1):
    '''
    given the residual data matrix and one w factor, and the number of change points
    first do change point detection to get the change points
    then reconstruct h given the change points

    r1: n_sample x n_feature
    w1: n_sample x 1
    '''
    algo = rpt.Dynp(model="", min_size=2, jump=1,custom_cost=nmf_cost(w1)).fit((r1.T))
    my_bkps = algo.predict(n_bkps=n_bkps)
    # algo = rpt.Pelt(model="", min_size=2, jump=1,custom_cost=nmf_cost(w1)).fit((r1.T))
    # my_bkps = algo.predict(pen=10)
    bkps = np.insert(my_bkps,0,0)
    h1 = rank1_pwc_h_update_h_given_bkps(r1,w1,bkps)
    return h1,bkps


def rank1_update_w(r1,h1):
    '''
    a general update for w
    r1: n_sample x n_feature
    h1: 1 x n_feature
    '''
    w1 = np.maximum(r1.dot(h1.T) / (h1.dot(h1.T)),0)
    return w1
    

def rank1_pwc_h_fit(r1,n_bkps=1,max_iters=10,tol=1e-4,w_init=None,h_init=None):
    '''
    fit one w and h factor given the residual r1 and number of change points
    '''
    w1,h1=random_init_constant_h_rank1(r1,1)
    if w_init is not None:
        w1= w_init
    if h_init is not None:
        h1 = h_init
    err_l = []
    err_prev = 1e10
    for ii in range(max_iters):
        h1,bkps=rank1_pwc_h_update_h_full(r1,w1,n_bkps)
        w1=rank1_update_w(r1,h1)
        
        err = np.linalg.norm(r1 - w1.dot(h1),ord='fro')**2
        err_l.append(err)
        if (err_prev - err) / err_prev < tol:
            break
        err_prev = err
    return w1,h1,bkps,err_l

####### full model ########

def random_init(X,n_compo):
    '''
    X: n_samples x n_features
    '''
    avg = np.sqrt(X.mean() / n_compo)
    H = avg * np.random.normal(size=(n_compo,X.shape[1])).astype(
    X.dtype, copy=False
    )
    W = avg * np.random.normal(size=(X.shape[0], n_compo)).astype(
    X.dtype, copy=False
    )
    np.abs(H, out=H)
    np.abs(W, out=W)
    return W,H

def init_with_nmf(X,n_compo):
    # can try order the factors st the the more "important" factors are earlier
    model = NMF(n_compo)
    W=model.fit_transform(X)
    H = model.components_
    return W,H

def leave_one_out_residual(X,W,H,leave_out_ind):
    W_ = np.concatenate((W[:,:leave_out_ind],W[:,leave_out_ind+1:]),axis=1)
    H_ = np.concatenate((H[:leave_out_ind,:],H[leave_out_ind+1:,:]),axis=0)
    resid = X - W_.dot(H_)
    return resid

    # MAIN stuff #
def nmf_pwc_h_fit(X,n_compo,n_bkps=1,max_iters_outer=10,max_iters_rank1=4,tol=1e-8,w_init=None,h_init=None):
    if w_init is None or h_init is None:
        w_init,h_init = init_with_nmf(X,n_compo)
    w_=w_init
    h_=h_init
    bkps_l = np.empty(n_compo,dtype=object)
    err_tot_l=[]
    err_tot_prev = 1e10
    for ii in range(max_iters_outer):
        for c in range(n_compo):
            r1 = leave_one_out_residual(X,w_,h_,c)
            w1 = w_[:,[c]]
            h1 = h_[[c],:]
            w1,h1,bkps,err_l=rank1_pwc_h_fit(r1, n_bkps=n_bkps, w_init=w1, h_init=h1,tol=tol,max_iters=max_iters_rank1)
            w_[:,c] = w1[:,0]
            h_[c,:] = h1[0,:]
            bkps_l[c] = bkps
        err_tot = np.linalg.norm(X-w_.dot(h_),ord='fro')**2
        err_tot_l.append(err_tot)
        if (err_tot_prev - err_tot) / err_tot_prev < tol:
            break
        err_tot_prev = err_tot
        
    return w_,h_,bkps_l, err_tot_l
    

