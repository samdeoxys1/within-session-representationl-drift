import numpy as np
import scipy
import matplotlib.pyplot as plt
import sys,os,copy,pdb,importlib
from importlib import reload
import pandas as pd

import sys
sys.path.append('/mnt/home/szheng/projects/nmf_analysis')
import change_point_analysis as cpa
import switch_metrics as sm



def gen_circular_shuffle_trialtype_seperated(changes_df,nrepeats=200,min_cpd_win=2):
    '''
    changes_df: (task_ind x trialtype_ind x uid x field_id) x ntrials
    '''
    changes_df = changes_df.dropna(axis=1,how='all')
    gpb = changes_df.groupby(level=(0,1),group_keys=False)
    x_shuffle_d = {}
    changes_df_shuffle = []
    original_col = changes_df.columns
    for k,val in gpb:
        x = val.loc[k].dropna(axis=1,how='all')
        x_shuffle = cpa.gen_circular_shuffle(x,nrepeats=nrepeats,min_cpd_win=min_cpd_win)
        x_shuffle_d[k] = x_shuffle
    for r in range(nrepeats):
        x_shuffle_d_one = {}
        for k,x_sh in x_shuffle_d.items():
            x_shuffle_d_one[k] = x_sh[r]
        x_shuffle_d_one = pd.concat(x_shuffle_d_one,axis=0)
        changes_df_shuffle.append(x_shuffle_d_one[original_col])

    return changes_df_shuffle

def gen_circular_shuffle_trialtype_seperated_get_all_sw(changes_df,pf_params_recombined,spk_beh_df,nrepeats=200,min_cpd_win=2):
    changes_df_shuffle = gen_circular_shuffle_trialtype_seperated(changes_df,nrepeats=nrepeats,min_cpd_win=min_cpd_win)
    all_sw_shuffle = []
    for cd_sh_one in changes_df_shuffle:
        all_sw_sh_one,_=sm.get_all_sw_add_metrics_all_tasks(cd_sh_one,pf_params_recombined,spk_beh_df,is_changes_df=True,do_add_metrics=False)
        all_sw_shuffle.append(all_sw_sh_one)
    # return all_sw_shuffle
    return {'all_sw':all_sw_shuffle,'changes_df':changes_df_shuffle}

#####=======new shuffle test method, based on changes_df======#######
## just count the number of coswitching per trial, then use combinatorics to get the number of pairs
## for counting across trials, use rolling; then need to subtract the double counts, which are the n-1 tuple counts on the overlapping trials
## the overlapping trials should start from lag:(ntrial-1); during the subtraction, the raw combination is subtracted , not the subtracted combination (i.e. ignoring overlap on the n-2 tuples)



def get_n_pair_per_trial_sliding_window(changes_df,window_l=[0,1,2],
                                            n_co=2,
                                            onoff=1):
    '''
    window_l: trial lag to consider co pop up
    
    add up the within windows, subtract the lower order count on the overlapping part
    e.g. suppose we define coswitch to be within 2 trials. Trial 1,2 are grouped, 2,3 are grouped, then counting pairs
    within both and sum them would lead to double counting the pairs in trial 2, and need to subtract.
    k: how many in pair to count
    '''
    n_pair_per_trial_d = {}
    n_pair_per_trial_pre_subtract_d={} # before subtracting lower order, need this to be subtracted from the higher order
    n_pair_d = {}
    prev_win = 0
    changes_df = changes_df.dropna(axis=1,how='all')
    changes_int = (changes_df==onoff).astype(int)
    ntrials = changes_df.shape[1]
    for win in window_l:
        if win==0:
            n_sw_per_trial = changes_int.sum(axis=0)
            n_pair_per_trial=scipy.special.comb(n_sw_per_trial,n_co)
            n_pair_per_trial_d[win] = n_pair_per_trial
            n_pair_per_trial_pre_subtract_d[win] = n_pair_per_trial
        else:

            n_sw_per_trial = changes_int.rolling(win+1,axis=1).sum().sum(axis=0)
            n_pair_per_trial=scipy.special.comb(n_sw_per_trial,n_co)
            n_pair_per_trial_pre_subtract_d[win] = copy.copy(n_pair_per_trial)
            n_pair_per_trial[win:-1] = n_pair_per_trial[win:-1] - n_pair_per_trial_pre_subtract_d[prev_win][win:-1] # the overlap indices are from win to the second to last one
            n_pair_per_trial_d[win] = n_pair_per_trial
        n_pair_d[win] = n_pair_per_trial.sum()
        prev_win = win
    
    return n_pair_d,n_pair_per_trial_d,n_pair_per_trial_pre_subtract_d

def get_n_pair_per_trial_sweep(changes_df,window_l=[0,1,2],n_co_l=[2,3,4]):
    '''
    sweeping on off, also the number within tuple
    '''
    n_pair_all_n_co_d={}
    for n_co in n_co_l:
        n_pair_per_n_co = {}
        for onoff in [1,-1]:
            n_pair_d,_,_ = get_n_pair_per_trial_sliding_window(changes_df,window_l=window_l,
                                            n_co=n_co,
                                            onoff=onoff)
            n_pair_per_n_co[onoff] = pd.Series(n_pair_d)
        n_pair_all_n_co_d[n_co] = pd.concat(n_pair_per_n_co)
    n_pair_all_n_co_d = pd.DataFrame(n_pair_all_n_co_d)
    n_pair_all_n_co_d=n_pair_all_n_co_d.stack()
    n_pair_all_n_co_d.index.names=['onoff','n_trial_lag','n_in_tuple']

    return n_pair_all_n_co_d


def test_co_switch(changes_df,changes_df_shuffle,window_l=[0,1,2],n_co_l=[2,3,4]):
    # work for one task!
    n_pair_all_n_co_d=get_n_pair_per_trial_sweep(changes_df,window_l=[0,1,2],n_co_l=[2,3,4])
    n_pair_all_n_co_d_l = []
    for cd_sh in changes_df_shuffle:
        n_pair_all_n_co_d_sh = get_n_pair_per_trial_sweep(cd_sh,window_l=[0,1,2],n_co_l=[2,3,4])
        n_pair_all_n_co_d_l.append(n_pair_all_n_co_d_sh)
    n_pair_all_n_co_d_l = pd.concat(n_pair_all_n_co_d_l,axis=1)
    pval=(n_pair_all_n_co_d_l>=n_pair_all_n_co_d.values[:,None]).mean(axis=1) 
    return pval,n_pair_all_n_co_d,n_pair_all_n_co_d_l

def test_co_switch_all_task(changes_df_all_task,changes_df_shuffle_all_task=None,window_l=[0,1,2],n_co_l=[2,3,4],shuffle_kwargs={}):
    gpb = changes_df_all_task.groupby(level=0)
    pval_all_task = {}
    n_pair_all_n_co_d_all_task={}
    n_pair_all_n_co_d_l_all_task={}
    nrepeats=shuffle_kwargs.get('nrepeats',100)
    min_cpd_win = shuffle_kwargs.get('min_cpd_win',2)
    if changes_df_shuffle_all_task is None:
        changes_df_shuffle_all_task = gen_circular_shuffle_trialtype_seperated(changes_df_all_task,nrepeats=nrepeats,min_cpd_win=min_cpd_win)
    for task,val in gpb:
        changes_df_shuffle = [x.loc[task] for x in changes_df_shuffle_all_task]
        pval,n_pair_all_n_co_d,n_pair_all_n_co_d_l = test_co_switch(val,changes_df_shuffle=changes_df_shuffle,window_l=window_l,n_co_l=n_co_l)
        pval_all_task[task] = pval
        n_pair_all_n_co_d_all_task[task] = n_pair_all_n_co_d
        n_pair_all_n_co_d_l_all_task[task] = n_pair_all_n_co_d_l
    pval_all_task = pd.concat(pval_all_task,axis=0)
    n_pair_all_n_co_d_all_task = pd.concat(n_pair_all_n_co_d_all_task,axis=0)
    n_pair_all_n_co_d_l_all_task = pd.concat(n_pair_all_n_co_d_l_all_task,axis=0)

    return pval_all_task, n_pair_all_n_co_d_all_task, n_pair_all_n_co_d_l_all_task
    
#========#

#====post process====#
def get_shuffle_summary_combine_with_data(n_co_sw_all,n_co_sw_shuffle_all):
    '''
    n_co_sw_all: series, nrows (each is a setup (isnovel, onoff, n_trial_lag, n_in_tuple), can also be concat from all sessions)
    n_co_sw_shuffle_all: df: nrows x nrepeats
    '''
    # get summary stats of shuffle
    shuffle_median=n_co_sw_shuffle_all.median(axis=1)
    shuffle_ci_high = n_co_sw_shuffle_all.quantile(0.975,axis=1)
    shuffle_ci_low = n_co_sw_shuffle_all.quantile(0.025,axis=1)
    n_co_sw_shuffle_all_summary=pd.concat({'median':shuffle_median,'ci_high':shuffle_ci_high,'ci_low':shuffle_ci_low},axis=1)
    
    # combine with data
    n_co_sw_shuffle_all_summary['data'] = n_co_sw_all
    return n_co_sw_shuffle_all_summary

#========#



def shuffle_test_pair_share_onoff(changes_one,nrepeats=100,alpha=0.025,min_cpd_win=2):
    '''
    alpha: precomputed, no further transformation to one-sided or two-sided
    '''
    changes_shuffle_l = gen_circular_shuffle_trialtype_seperated(changes_one,nrepeats=nrepeats,min_cpd_win=min_cpd_win)
    
    share_onoff_count_l=[]
    share_on_count_l = []
    share_off_count_l = []
    share_onoff_ratio_on_l = []
    
    
    for changes_shuffle in changes_shuffle_l:
        _, share_onoff_count, share_on_count, share_off_count,share_onoff_ratio_on = cpa.get_shared_onoff(changes_shuffle)
        share_onoff_count_l.append(share_onoff_count)
        share_on_count_l.append(share_on_count)
        share_off_count_l.append(share_off_count)
        share_onoff_ratio_on_l.append(share_onoff_ratio_on)
    
    share_shuffle = {'onoff':np.array(share_onoff_count_l),'on':np.array(share_on_count_l),
                     'off':np.array(share_off_count_l),'onoff_ratio_on':np.array(share_onoff_ratio_on_l)}
    
    share_data = {}
    share_onoff_inds, share_data['onoff'], share_data['on'], share_data['off'],share_data['onoff_ratio_on'] = cpa.get_shared_onoff(changes_one)
    
    cdf_d = {}
    issig_d = {}
    for k,shuffle in share_shuffle.items():
        data = share_data[k]
        cdf = np.mean(data > shuffle)
        issig = cdf > (1-alpha)
        cdf_d[k] = cdf
        issig_d[k] = issig
    res = {'count':share_data,'cdf':cdf_d,'issig':issig_d}
    res = pd.DataFrame(res).unstack()
    return res,share_shuffle