import numpy as np
import scipy
import matplotlib.pyplot as plt
import sys,os,copy,pdb,importlib
from importlib import reload
import pandas as pd

from scipy.stats import poisson
def get_p_n_or_more_given_lambda(n,lam):
    return 1-poisson.cdf(n-1,lam)- 0.5 * poisson.pmf(n,lam)

def get_p_n_or_fewer_given_lambda(n,lam):
    return poisson.cdf(n,lam)

def get_pval_for_peak_in_range(ccg,ccg_filtered,peak_range_int,valence=1):
    '''
    do testing for one range, compare the peak (positive or negative) against the slow baseline lambda
    '''

    window_size = len(ccg)
    assert window_size%2==1
    mid_index = int(window_size // 2)
    peak_check_range = slice(mid_index + peak_range_int[0],mid_index + peak_range_int[1])
    if valence > 0 :
        peak_ind_within = np.argmax(ccg[peak_check_range])
    if valence < 0 :
        peak_ind_within = np.argmin(ccg[peak_check_range])
    
    peak_val = ccg[peak_check_range][peak_ind_within]
    peak_slow_lam = ccg_filtered[peak_check_range][peak_ind_within]
    if valence > 0:
        pval = get_p_n_or_more_given_lambda(peak_val, peak_slow_lam)
    elif valence < 0:
        pval = get_p_n_or_fewer_given_lambda(peak_val, peak_slow_lam)

    return pval, peak_val, peak_slow_lam
    

def get_peak_and_test(ccg,ccg_filtered,valence=1,bin_size=0.0004,peak_range=(0.0008,0.0028),peak_range_int=(2,7),
                      anti_causal_peak_range=(-0.002,0),anti_causal_peak_range_int=(-5,0),pfast_thresh=0.001, pcausal_thresh=0.0026
                     ):
    '''
    ccg: window_size, has to be odd
    ccg_filtered: same size as ccg; filtered from a partially hollowed gaussian
    
    do both fast and causal testing
    '''
    
    if peak_range_int is None:
        peak_range_int = (int(peak_range[0]//bin_size),int(peak_range[1]//bin_size))
    if anti_causal_peak_range_int is None:
        anti_causal_peak_range_int = (int(anti_causal_peak_range[0]//bin_size),int(anti_causal_peak_range[1]//bin_size))
    
    
    pval, peak_val, peak_slow_lam = get_pval_for_peak_in_range(ccg,ccg_filtered,peak_range_int,valence=valence)
    _, peak_val_anti_causal, peak_slow_lam_anti_causal = get_pval_for_peak_in_range(ccg,ccg_filtered,anti_causal_peak_range_int,valence=valence)
    
    if valence > 0:
        pval_causal = get_p_n_or_more_given_lambda(peak_val, peak_val_anti_causal)
    elif valence < 0:
        pval_causal = get_p_n_or_fewer_given_lambda(peak_val, peak_val_anti_causal)
        
    con = (pval < pfast_thresh) & (pval_causal < pcausal_thresh)
    
    return con, pval, pval_causal
        
def get_partially_hollowed_filter(filter_windowsize=25,sigma = 0.01,bin_size=0.0004,hollow_frac = 0.6):
    '''
    sigma, bin_size in s
    filter_windowsize in bin
    '''
    sigma_bin = sigma / bin_size
    gaus_filt = scipy.signal.gaussian(filter_windowsize,sigma_bin)
    gaus_filt[len(gaus_filt)//2] = gaus_filt[len(gaus_filt)//2] * hollow_frac
    gaus_filt = gaus_filt/gaus_filt.sum()
    return gaus_filt
    
    
def filter_ccg(ccg,filter_kwargs={}):
    '''
    ccg: ccg_windowsize x n_neurons x n_neurons
    '''
    gaus_filt = get_partially_hollowed_filter(**filter_kwargs)
    ccg_windowsize=ccg.shape[0]
    ccg_augmented = np.concatenate([np.flip(ccg,axis=0),ccg,np.flip(ccg,axis=0)],axis=0)
    ccg_filtered = scipy.ndimage.convolve1d(ccg_augmented, gaus_filt, mode='reflect',axis=0)[ccg_windowsize:ccg_windowsize*2]
    return ccg_filtered
from tqdm import tqdm
def get_mono_con(ccg=None,cell_cols=None):
    if ccg is None:
        print('not implemented')
        return
    
    ccg_filtered = filter_ccg(ccg,filter_kwargs={})
    n_neurons = ccg_filtered.shape[1]
    exc_res_l = np.zeros((n_neurons,n_neurons))
    inh_res_l = np.zeros((n_neurons,n_neurons))
    for i in tqdm(range(n_neurons),position=0,leave=True):
        for j in range(n_neurons):
            if i!=j:
                exc_res_l[i,j] = get_peak_and_test(ccg[:,i,j],ccg_filtered[:,i,j],valence=1)[0]
                inh_res_l[i,j] = get_peak_and_test(ccg[:,i,j],ccg_filtered[:,i,j],valence=-1,anti_causal_peak_range_int=(-10,0))[0]            
    if cell_cols is not None:
        exc_res_l = pd.DataFrame(exc_res_l,index=cell_cols,columns=cell_cols).T # post x pre
        inh_res_l = pd.DataFrame(inh_res_l,index=cell_cols,columns=cell_cols).T
    return exc_res_l, inh_res_l
            
            
            
