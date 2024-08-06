import numpy as np
import scipy
import matplotlib.pyplot as plt
import sys,os,copy,pdb,importlib
from importlib import reload
import pandas as pd


def decompose_variability_onetrialtype(Xhat_one,fr_trial_one,need_rescale=False):
    '''
    get: total variance, variance in fit, average squared residual, mean, normalized by mean
    '''
    res = {}
    res['tot_var'] = fr_trial_one.var(axis=1,ddof=0) # ddof=0 to make the equation holds; alternatively could have the mean of resid2 replaced by /(Ntrial-1)
    res['mean'] = fr_trial_one.mean(axis=1)
    if need_rescale:
        Xhat_rescaled_back = Xhat_one * fr_trial_one.max(axis=1).values[:,None] # inverse of the max normalization
    else:
        Xhat_rescaled_back = Xhat_one
    res['fit_var'] = Xhat_rescaled_back.var(axis=1,ddof=0)
    resid = fr_trial_one - Xhat_rescaled_back
    res['resid2_mean'] = (resid**2).mean(axis=1)
    res['fit_var_ratio'] = res['fit_var'] / res['tot_var']
    res['resid2_mean_ratio'] =1-res['fit_var_ratio']
    for k in ['tot_var','fit_var','resid2_mean']:
        res[k+'_norm'] = res[k] / res['mean']
    res = pd.DataFrame(res)
    return res

def post_decomp_corr(res,pcorr_vars=['mean','fit_var_norm','resid2_mean_norm']):
    all_corr = res.rcorr(stars=False)
    p_corr = res[pcorr_vars].pcorr()
    corr_res = {'all_corr':all_corr,'p_corr':p_corr}
    return corr_res