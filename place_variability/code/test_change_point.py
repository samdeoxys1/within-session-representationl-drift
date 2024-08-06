'''
a newer version of the change_point_analysis
'''
import sys,os,pickle,copy, itertools,pdb
sys.path.append('/mnt/home/szheng/projects/seq_detection2/code')
sys.path.append('/mnt/home/szheng/projects/util_code')
sys.path.append('/mnt/home/szheng/projects/nmf_analysis')
import numpy as np

import pandas as pd
from importlib import reload
import pynapple as nap
import ruptures as rpt
import tqdm
import change_point_analysis as cpa
import pingouin as pg

def get_explained_var_ratio(signal,error):
    signal_var = np.var(signal)
    exp_var_ratio = (signal_var - error / len(signal)) / signal_var
    return exp_var_ratio


def test_contiguity(signal,n_shuffle=200,sig_thresh=0.05,n_change_pts=1,min_size=2,signal_shuffle=None):
    '''
    idea: if there are contiguous trials with different mean rates, then 
    the error given by a change point detection model (with one change point) 
    should be significantly lower than the errors with the data shuffled in trials
    '''
    model = rpt.Dynp(model='l2',jump=1)
    c = model.fit(signal)
    change_pts=np.array(model.predict(n_change_pts))
    error = c.cost.sum_of_costs(list(change_pts)) # list is important here!

    exp_var_ratio = get_explained_var_ratio(signal,error)

    err_l = []
    exp_var_ratio_l = []
    if signal_shuffle is None:
        signal_all = np.tile(signal,(n_shuffle,1))
        rng = np.random.default_rng(123)
        signal_all = rng.permuted(signal_all,axis=1)
    else: 
        signal_all = signal_shuffle
        n_shuffle = signal_all.shape[0]
    for ii in range(n_shuffle):
        model = rpt.Dynp(model='l2',jump=1,min_size=min_size)
        # perm = np.arange(signal.shape[0])
        # np.random.shuffle(perm)
        # c = model.fit(signal[perm])
        c = model.fit(signal_all[ii])
        change_pts=np.array(model.predict(n_change_pts))
        l = c.cost.sum_of_costs(list(change_pts))
        exp_var_ratio_sh = get_explained_var_ratio(signal_all[ii],l)
        exp_var_ratio_l.append(exp_var_ratio_sh)
        err_l.append(l)
    err_l = np.array(err_l)
    exp_var_ratio_l = np.array(exp_var_ratio_l)
    exp_var_ratio_sh_med = np.median(exp_var_ratio_l)
    exp_var_ratio_sh_high = np.quantile(exp_var_ratio_l,0.975)
    exp_var_ratio_sh_low = np.quantile(exp_var_ratio_l,0.025)

    # p = np.quantile(err_l,error)
    p = np.count_nonzero(error > err_l) / len(err_l)
    issig = p < sig_thresh
    res = pd.Series([p,exp_var_ratio,exp_var_ratio_sh_med,exp_var_ratio_sh_high,exp_var_ratio_sh_low],index=['pval','exp_var_ratio','exp_var_ratio_sh_med','exp_var_ratio_sh_high','exp_var_ratio_sh_low'])
    # return error, err_l, p, issig
    # return res
    return res, err_l,signal_all

from pandarallel import pandarallel
def test_contiguity_allrows(X,n_shuffle=200,sig_thresh=0.05,n_change_pts=1,min_size=2):
    pandarallel.initialize(use_memory_fs=False,progress_bar=True)
    # row_func = lambda x:test_contiguity(x.dropna().values,n_shuffle=n_shuffle,sig_thresh=sig_thresh,n_change_pts=n_change_pts,min_size=min_size)[2]
    row_func = lambda x:test_contiguity(x.dropna().values,n_shuffle=n_shuffle,sig_thresh=sig_thresh,n_change_pts=n_change_pts,min_size=min_size)
    res = X.parallel_apply(row_func,axis=1)
    return res

def test_contiguity_multi_n_change_pts(X,n_shuffle=200,n_change_pts_l=None,n_change_pts_max_MAX=4,min_size=2):
    ntrials = X.shape[1]
    gpb = X.groupby(level=(0,1))
    pval_all = {}
    for k,val in gpb:
        val = val.loc[k]
        ntrials = val.dropna(axis=1,how='all').shape[1]
        n_change_pts_max = np.minimum(int(ntrials // 4),n_change_pts_max_MAX) # 4 here is kinda arbitrary
        n_change_pts_l = np.arange(1,n_change_pts_max+1)
        pval_d = {}
        for ncp in n_change_pts_l:
            pval_d[ncp]=test_contiguity_allrows(val,n_shuffle=n_shuffle,sig_thresh=0.05,n_change_pts=ncp,min_size=min_size)
        pval_d = pd.concat(pval_d,axis=1)
        pval_all[k] = pval_d
    pval_all = pd.concat(pval_all,axis=0)
    return pval_all

# get the best n
def get_n_cpd(row,alpha=0.05):
    pval = row.loc[slice(None),'pval']
    ntest = len(pval.dropna())
    alpha_bc = alpha / ntest
#     row_passed = row[row <= alpha_bc]
#     row_passed= row
    # anysig = (row <= alpha_bc).sum()
    
    anysig = (pval <= alpha_bc).sum()

#     if len(row_passed)==0:
    if anysig == 0:
        n = 0
    else:
        # n = row.idxmin()
        n = pval.idxmin()
    return n

# once figure out the best n for each field
def get_change_points_all_fields_different_n(X,best_n,min_size = 2):
    '''
    X: n_fields x n_trials, can have multiple tasks/trialtypes stacked
    
    '''
    cp_d = {}
    signal_pred_d = {}
    for k,signal in X.iterrows():
        n = best_n.loc[k]
        signal = signal.dropna()
        signal_pred,change_pts = cpa.predict_from_cpts_wrapper(signal.values,n,min_size=min_size)
        cp_d[k] = change_pts
        signal_pred_d[k] = pd.Series(signal_pred,index=signal.index)
    cp_d = pd.DataFrame(cp_d.values(),index=cp_d.keys())
    signal_pred_d = pd.DataFrame(signal_pred_d).T
        
    return signal_pred_d,cp_d

def decompose_variability_onetrialtype(Xhat_one,fr_trial_one,rescale_to_max=False):
    '''
    get: total variance, variance in fit, average squared residual, mean, normalized by mean
    '''
    res = {}
    res['tot_var'] = fr_trial_one.var(axis=1,ddof=0) # ddof=0 to make the equation holds; alternatively could have the mean of resid2 replaced by /(Ntrial-1)
    res['mean'] = fr_trial_one.mean(axis=1)
    if rescale_to_max:
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
    var_res = pd.DataFrame(res)
    
    
    
    return var_res



def get_changes_df(X_pwc_norm,X,switch_magnitude=0.4,low_thresh=1.,high_thresh=0.):
    # get changes_df
    changes_df_all = []
    
    for k,val in X_pwc_norm.groupby(level=(0,1)):
        val = val.dropna(axis=1,how='all')
        sw_on,sw_off,changes_df = cpa.detect_switch_pwc(val,switch_magnitude=switch_magnitude,low_thresh=low_thresh,high_thresh=high_thresh)
    #     changes_df.columns=  val.columns
    #     changes_df.index=val.index
        changes_df_all.append(changes_df)
        
    changes_df_all = pd.concat(changes_df_all,axis=0)
    changes_df_all = changes_df_all.sort_index(axis=1)
    return changes_df_all

def decompose_variability_all_trialtype(X_pwc,X,rescale_to_max=False):
    '''
    var_res_all: (task x trialtype x uid x field) x [all types of variance]
    '''
    var_res_all = {}
    for k,val in X_pwc.groupby(level=(0,1)):
        val = val.dropna(axis=1,how='all').loc[k]
        fr_one_tt = X.loc[k].dropna(axis=1,how='all')
        var_res = decompose_variability_onetrialtype(val,fr_one_tt)
        var_res_all[k] = var_res    
    var_res_all = pd.concat(var_res_all,axis=0)
    
    all_corr = var_res_all.groupby(level=0).apply(lambda x:x.rcorr(stars=False))
    p_corr = var_res_all.groupby(level=0).apply(lambda x:x.pcorr())
    
    return var_res_all,all_corr,p_corr
    

def detect_switch_all_steps(X,n_shuffle=100,n_change_pts_max_MAX=5,min_size=2):
    '''
    X: pf_res['params_recombined'].loc['fr_peak']
    '''

    # shuffle test get pval; slow
    pval_df = test_contiguity_multi_n_change_pts(X,n_shuffle=n_shuffle,n_change_pts_max_MAX=n_change_pts_max_MAX,min_size=min_size)
    
    # get best n change points
    best_n = pval_df.apply(get_n_cpd,axis=1)
    
    # get change points and fitted piecewise constant
    X_pwc, cp_d=get_change_points_all_fields_different_n(X,best_n)
    X_pwc_norm = X_pwc / np.nanmax(X_pwc.values,axis=1,keepdims=True)
    
    # get changes df
    changes_df_all = get_changes_df(X_pwc_norm,X)

    # decompose variance
    var_res_all,all_corr,p_corr = decompose_variability_all_trialtype(X_pwc,X,rescale_to_max=False)


    res = {'pval':pval_df,'best_n':best_n,'X_pwc':X_pwc,'X_raw':X,'X_pwc_norm':X_pwc_norm,
            'changes_df':changes_df_all, 'var_res':var_res_all,
            'corr':all_corr,'p_corr':p_corr}
    return res

#========using PELT=========#
def detect_switch_one_penalty(X_raw,pen=0.3,min_size=2,switch_magnitude=0.4):
    X_pwc_norm, cpts = cpa.predict_from_cpts_wrapper_allrows(X_raw,pen,cost='l2',min_size=min_size,model_type=rpt.Pelt)
    
    # restore the proper trial index
    gpb_raw=X_raw.groupby(level=(0,1)) # task_ind, trial_type
    gpb_pwc=X_pwc_norm.groupby(level=(0,1))
    val_pwc_l = []
    for k,val in gpb_raw:
        cols=val.dropna(axis=1,how='all').columns
        val_pwc=gpb_pwc.get_group(k).dropna(axis=1,how='all')
        val_pwc.columns=cols
        val_pwc_l.append(val_pwc)
    val_pwc_l = pd.concat(val_pwc_l,axis=0)
    val_pwc_l = val_pwc_l.sort_index(axis=1)
    X_pwc_norm = val_pwc_l
    
    X_pwc =  X_pwc_norm * X_raw.max(axis=1).values[:,None]
    changes_df = get_changes_df(X_pwc_norm,X_raw,switch_magnitude=switch_magnitude,low_thresh=1.,high_thresh=0.)

    # decompose variance
    var_res_all,all_corr,p_corr = decompose_variability_all_trialtype(X_pwc,X_raw,rescale_to_max=False)    
    best_n = ((changes_df!=0)&(changes_df.notna())).astype(int).sum(axis=1)

    res = {'pval':None,'best_n':best_n,'X_pwc':X_pwc,'X_raw':X_raw,'X_pwc_norm':X_pwc_norm,
            'changes_df':changes_df, 'var_res':var_res_all,
            'corr':all_corr,'p_corr':p_corr}
    return res

def detect_switch_sweep_penalty(X_raw,pen_l=[0.3,0.4,0.5,0.6],min_size=2,switch_magnitude=0.4):
    res_d = {}
    for pen in pen_l:
        res = detect_switch_one_penalty(X_raw,pen=pen,min_size=min_size,switch_magnitude=switch_magnitude)
        res_d[pen] = res
    return res_d


import pickle
def correct_detect_switch_all_steps(sw_res_fn):
    sw_res = pickle.load(open(sw_res_fn,'rb'))
    pval_df = sw_res['pval']
    X = sw_res['X_raw']
    # get best n change points
    best_n = pval_df.apply(get_n_cpd,axis=1)
    
    # get change points and fitted piecewise constant
    X_pwc, cp_d=get_change_points_all_fields_different_n(X,best_n)
    X_pwc_norm = X_pwc / np.nanmax(X_pwc.values,axis=1,keepdims=True)
    
    # get changes df
    changes_df_all = get_changes_df(X_pwc_norm,X)

    # decompose variance
    var_res_all,all_corr,p_corr = decompose_variability_all_trialtype(X_pwc,X,rescale_to_max=False)


    res = {'pval':pval_df,'best_n':best_n,'X_pwc':X_pwc,'X_raw':X,'X_pwc_norm':X_pwc_norm,
            'changes_df':changes_df_all, 'var_res':var_res_all,
            'corr':all_corr,'p_corr':p_corr}
    pickle.dump(res,open(sw_res_fn,'wb'))
    print(f'{sw_res_fn} updated')
    return res


