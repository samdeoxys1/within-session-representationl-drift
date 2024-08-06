import ruptures as rpt
import numpy as np
import pandas as pd
import scipy
import os,sys,copy,itertools,pdb,importlib

import statsmodels
import statsmodels.api as sm
sys.path.append('/mnt/home/szheng/projects/nmf_analysis')
sys.path.append('/mnt/home/szheng/projects/place_variability/code')
sys.path.append('/mnt/home/szheng/projects/util_code')
import change_point_analysis as cpa
import plot_all_fr_map_x_pwc_one_session as plotfm
importlib.reload(plotfm)
import plot_helper as ph
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
def prep_one_sess(fr_map_trial_df_onett):
    '''
    fr_map_trial_df_onett: (neuron, pos) x trial
    '''
    xx=fr_map_trial_df_onett.dropna(axis=1,how='all')
    zscore_cell = True
    std_max=xx.groupby(level=0).apply(lambda x: np.std(x.values)).max()
    max_std_sc = 0.1 # adjusted such that more than 1/3 can achieve a var of 1, but others will be scaled
    zscore_func = lambda x:(x - np.mean(x.values)) / np.maximum(np.std(x.values),max_std_sc*std_max)
    if zscore_cell:
        xx=xx.groupby(level=0).apply(lambda x:zscore_func(x))
    xx=xx.dropna(axis=0,how='all')
    
    signal = xx.T.values
    n_components = signal.shape[0]
    pca = PCA(n_components=n_components)
    signal=pca.fit_transform(signal)
    pc_r2 = pca.explained_variance_ratio_
    ma= np.cumsum(pc_r2) > 0.95
    ind=np.min(np.nonzero(ma)[0])
    n_compo=ind+1
    print(pca.explained_variance_ratio_[:n_compo].sum())
    
    return signal[:,:n_compo]

###### fit cpd get r2
import ruptures as rpt
def fit_cpd_get_r2_multi_d(signal,ncpts, model_type='dyn',cost='l2',min_size=2,jump=1):
    '''
    assuming l2
    '''
    if model_type=='dyn':
        model = rpt.Dynp(model=cost,jump=jump,min_size=min_size)
    c=model.fit(signal)
    cpts = model.predict(ncpts)
    n_trial = signal.shape[0]
    err=c.cost.sum_of_costs(list(cpts))
    err_div_trial=err / n_trial
    sum_of_var = np.var(signal,axis=0).sum()
    r2=(sum_of_var - err_div_trial) / sum_of_var

    return cpts, r2

###### fit reg get r2
import statsmodels.api as sm
def fit_poly_regress_get_r2(xx,order=1,cost='l2',verbose=True):
    if cost=='l2':
        xs_l = []
        for o in range(1,order+1):
            xs = np.arange(len(xx)) ** o 
            xs_l.append(xs)
        xs_l = np.array(xs_l).T
#         pdb.set_trace()
        xs_l = sm.add_constant(xs_l)
        model = sm.OLS(xx,xs_l)
        results = model.fit()
        if verbose:
            print(results.summary())
        xx_pred=results.predict()
        exp_var=np.var(xx_pred)
        data_var = np.var(xx)
        r2 = results.rsquared
        return r2,xx_pred,exp_var,data_var
            
    else:
        print('not implemented')
        pass

def fit_poly_regress_get_r2_multi_d(signal,order=1):
    '''
    signal: n_sample (i.e. n_trial) x n_feature
    '''
    exp_var_all = []
    data_var_all = []
    xx_pred_all = []
    for ii in range(signal.shape[1]):
        r2,xx_pred,exp_var,data_var = fit_poly_regress_get_r2(signal[:,ii],order=order,verbose=False)
        exp_var_all.append(exp_var)
        data_var_all.append(data_var)
        xx_pred_all.append(xx_pred)
    r2 = np.sum(exp_var_all) / np.sum(data_var_all)
    xx_pred_all = np.array(xx_pred_all).T
    return r2,xx_pred_all


def fit_cpd_poly_regress_multi_order_all(
                    signal_all,
                    order_l=[1,2,3]):
    '''
    signal_all: 
        dict of preprocessed fr_map_trial_df, 
            key: ani, sess, ti, tt
            val: n_trial x n_feature
    '''
    r2_df_d_all = {}
    for order in order_l:
        ncpts = order
        r2_cpd_all = {}
        r2_reg_all = {}
        for k,signal in signal_all.items():
            try:
                _,r2_cpd = fit_cpd_get_r2_multi_d(signal,ncpts)
                r2_reg,_ = fit_poly_regress_get_r2_multi_d(signal,ncpts)
                r2_cpd_all[k] = r2_cpd
                r2_reg_all[k] = r2_reg
            except:
                pass
        r2_cpd_all = pd.Series(r2_cpd_all)
        r2_reg_all = pd.Series(r2_reg_all)
        r2_all = pd.concat({'step':r2_cpd_all,'reg':r2_reg_all},axis=1)
        r2_all['step_minus_reg'] = r2_all['step'] - r2_all['reg']
        r2_df_d_all[order]=r2_all
    r2_df_d_all = pd.concat(r2_df_d_all,axis=1)
    return r2_df_d_all

