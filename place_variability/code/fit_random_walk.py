import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import sys,os,copy,pdb,importlib,pickle
from importlib import reload
import pandas as pd
sys.path.append('/mnt/home/szheng/projects/nmf_analysis')
import change_point_analysis as cpa

def fit_random_walk_get_ll(xx):
    '''
    xx:
    '''
    xx_diff = np.diff(xx)
    xx_diff_sq = xx_diff**2
    sigma2 = np.mean(xx_diff_sq)

    ll = -1/2*np.log(np.pi*2*sigma2)-xx_diff_sq / (2*sigma2)      
    ll = np.sum(ll)


    return ll,sigma2

def fit_random_walk_

def fit_cpd_get_ll(xx,ncpts,cost='l2',min_size=2,common_var=False):
    
    signal_pred,cpt=cpa.predict_from_cpts_wrapper(xx,ncpts,cost=cost,min_size=min_size,return_var=True)
    xx_pwc,xx_var=signal_pred
    if common_var:
        xx_var = xx.var()
    
    ll = -1/2*np.log(np.pi*2*xx_var) -(xx - xx_pwc)**2 / (2*xx_var)
    ll = np.sum(ll)

    return ll,xx_pwc

def fit_rw_cpd_get_ll_all(X_raw, order=2,cost='l2',min_size=2,common_var=False):
    r2_reg_l = {}
    r2_cpd_l = {}
    for i,row in X_raw.iterrows():
        xx = row.dropna().values
        try:
            r2_reg,xx_pred=fit_random_walk_get_ll(xx)
            r2_cpd,xx_pwc = fit_cpd_get_ll(xx,order,cost=cost,min_size=min_size,common_var=common_var)
            r2_reg_l[i] = r2_reg
            r2_cpd_l[i] = r2_cpd
        except:
            pass
    r2_reg_l = pd.Series(r2_reg_l)
    r2_cpd_l = pd.Series(r2_cpd_l)
    try:
        r2_df=pd.concat({'rw':r2_reg_l,'step':r2_cpd_l},axis=1)
        r2_df['step_minus_rw'] = r2_df['step'] - r2_df['rw']
    except:
        return None
    return r2_df
