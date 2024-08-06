'''
plotting for change point related stuff
'''
import ruptures as rpt
import numpy as np
import pandas as pd
import scipy
import os,sys,copy,itertools,pdb,importlib

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
sys.path.append('/mnt/home/szheng/projects/util_code')
import plot_helper as ph
mpl.rcParams['image.cmap'] = 'Greys'

def plot_change_ratio(changes_df,fig=None,axs=None,skip_last_col=True,**kwargs):
    '''
    changes_df: 
    '''
    changes_df.columns = changes_df.columns.values.astype(int)
    if skip_last_col:
        changes_df = changes_df.loc[:,changes_df.columns!=-1] # get rid of the last column
    if axs is None:
        fig,axs = plt.subplots(3,1,sharex=True,figsize=(6,6))
    for ii,ch in enumerate([1,-1]):

        (changes_df==ch).mean(axis=0).plot(ax=axs[ii],xticks=changes_df.columns,title=ch)
    changes_df.mean(axis=0).plot(ax=axs[-1],xticks=changes_df.columns,title='both and off')
    plt.tight_layout()
    return fig,axs

def plot_switch_ratio_with_shuffle(sr_data,sr_l,alpha=0.05):
    '''
    sr_data: switch ratio from the data; dict of series
    sr_l: a list of sr, from shuffle
    '''
    keys = sr_data.keys()
    nplots = len(keys)
    fig,axs=plt.subplots(nplots,1,figsize=(6,nplots*3))
    for ii,k in enumerate(keys):
        sr_on=pd.DataFrame([sr[k] for sr in sr_l]).T
    
        yerr=pd.DataFrame([sr_on.quantile(1-alpha/2,axis=1),sr_on.quantile(alpha/2,axis=1)])
        # yerr = yerr - np.mean(sr_on.values,axis=1,keepdims=True).T

        # sr_on_mean = sr_on.mean(axis=1)
        # sr_on_mean.plot(ax=axs[ii])
        # axs[ii].fill_between(sr_on_mean.index,yerr.iloc[0],yerr.iloc[1])
        axs[ii].fill_between(np.arange(yerr.shape[1]),yerr.iloc[0],yerr.iloc[1],alpha=0.4)
        axs[ii].plot(sr_data[k].values,marker='o')
        axs[ii].set_title(k)
        axs[ii].set_xticks(np.arange(len(sr_data[k].index)))
        axs[ii].set_xticklabels(sr_data[k].index,rotation=45)

        plt.tight_layout()
    return fig,axs

def plot_sweep_test_switch_ratio(cdf_d_d,sr_d_d,sig_d_d,tosweep_key):
    '''
    sr_d_d: (nsweeps x nkeys) x ntrials
    '''
    keys = cdf_d_d.columns.get_level_values(0).unique()
    fig,axs = plt.subplots(len(keys),2,figsize=(6*2,3*len(keys)),sharey=False)
    for ii,k in enumerate(keys):
        axs[ii,0]=sns.heatmap(cdf_d_d[k],mask=~sig_d_d[k],xticklabels=cdf_d_d[k].columns.values.astype(int),ax=axs[ii,0])
        axs[ii,0].set_title(k)
        axs[ii,0].set_ylabel(tosweep_key,fontsize=12)
        annot = sig_d_d[k].applymap(lambda x:{True:'*',False:''}[x])
        
        axs[ii,1] = sns.heatmap(sr_d_d.loc[(slice(None),k),:],annot=annot.values,fmt='',ax=axs[ii,1],yticklabels=False,xticklabels=cdf_d_d[k].columns.values.astype(int))
        axs[ii,1].set(ylabel='')
    plt.tight_layout()
    return fig,axs

# old version, only significance, no switch ratio
# def plot_sweep_test_switch_ratio(cdf_d_d,tosweep_key,alpha=0.05,do_bonf=True):
#     if do_bonf:
#         alpha = alpha / cdf_d_d['on'].shape[1] # in some cases, like when there's sustain, this might be incorrect! 
#     sig_d_d = (cdf_d_d > (1-alpha/2)) | (cdf_d_d < alpha / 2)
#     keys = cdf_d_d.columns.get_level_values(0).unique()
#     fig,axs = plt.subplots(len(keys),1,figsize=(6,3*len(keys)))
#     for ii,k in enumerate(keys):
#         axs[ii]=sns.heatmap(cdf_d_d[k],mask=~sig_d_d[k],xticklabels=cdf_d_d[k].columns.values.astype(int),ax=axs[ii])
#         axs[ii].set_title(k)
#         axs[ii].set_ylabel(tosweep_key,fontsize=12)
#     plt.tight_layout()
#     return fig,axs

def plot_count_by_trial_with_thresh_sweep(count_l,sig_ct_l,thresh_l):
    '''
    each subplot: event count by trial; threshold is a hline
    all args: (nthresh,)
    '''
    nplots = len(thresh_l)
    fig,axs=ph.subplots_wrapper(nplots,return_axs=True,sharex=True)
    for ii, thresh in enumerate(thresh_l):
        ax=axs.ravel()[ii]
        ax=count_l.iloc[ii].plot(marker='o',xticks=count_l.iloc[ii].index,ax=ax)
        ax.axhline(sig_ct_l[ii])
        ax.set_title(thresh)
    plt.tight_layout()
    return fig,axs

def plot_fields_switching_each_trial(X,inds_d_one):
    '''
    for each trial heatmap plot the fields that switch on that trial; given by 
    inds_d = cpa.get_inds_switch_sametrial_sorted(changes_df)
    inds_d_one = inds_d.loc[1] or -1
    plotting on and off seperately
    '''
    trials = inds_d_one.index.get_level_values(0).unique()
    nplots = len(trials)
    fig,axs=ph.subplots_wrapper(nplots,return_axs=True)
    
    if nplots==1:
        axs=np.array([axs]) # otherwise ravel would give error
    for ii,tt in enumerate(trials):
        ax=axs.ravel()[ii]
        sns.heatmap(X.loc[inds_d_one.loc[tt].values],ax=ax,cmap='Greys')
        ax.set_title(tt)
    plt.tight_layout()
    return fig,axs

    

# using varying window sizes for defining co-switching
def plot_rate_vs_time_shuffle_test_res(test_res_df,time_key='window_median',fig=None,ax=None,**kwargs):
    '''
    test_res_df: from change_point_analysis_central_arm_seperate (cpacas.sweep_test_coswitch)
    time_key: which column in test_res_df to use as the time (x-axis)
    '''
    kwargs_ = dict(alpha=0.4)
    kwargs_.update(kwargs)
    if ax is None:
        fig,ax=plt.subplots()
    
    xs = test_res_df[time_key]
    ax.fill_between(xs,test_res_df['rate_ci_low'],test_res_df['rate_ci_up'],alpha=kwargs_['alpha'],label='shuffle ci')
    ax.plot(xs,test_res_df['rate_p_thresh'],linestyle=':',label='significant threshold')
    ax.plot(xs,test_res_df['rate'],label='data')
    ax.legend()

    ax.set_xlabel('Window of co-switch (sec)')
    ax.set_ylabel('Rate of co-switch\n(# pairs per sec)')
    
    return fig,ax

### plot for pairwise count test vs window size


def plot_pairwise_coswitch_test_result_one(test_res_df_onoff,normalize_by_key='window_size',fig=None,ax=None,plot_kws={}):
    '''
    test_res_df_onoff: test_res_df.loc[onoff]; test_res_df: from test_res = cpacas.sweep_test_coswitch_wrapper, test_res['test_res_df']
    lineplot + ci shade
    '''
    if ax is None:
        fig,ax = plt.subplots()
    xs=test_res_df_onoff['window_median'].round(2).values
    plot_kws_ = {}
    plot_kws_.update(plot_kws)
    ys = test_res_df_onoff['count'].values
    ci_up = test_res_df_onoff['count_ci_up'].values
    ci_low = test_res_df_onoff['count_ci_low'].values
    if normalize_by_key is not None:
        ys = ys / test_res_df_onoff[normalize_by_key].values
        ci_up = ci_up / test_res_df_onoff[normalize_by_key].values
        ci_low = ci_low / test_res_df_onoff[normalize_by_key].values
    ax.plot(xs,ys)
    ax.fill_between(xs,ci_low.astype(float),ci_up.astype(float),alpha=0.3)
    ax.set(xlabel='window size',ylabel=f'count / {normalize_by_key}')
    return fig,ax

def plot_pairwise_coswitch_test_result_both(test_res_df,normalize_by_key='window_size',fig=None,ax=None,plot_kws={}):
    '''
    test_res_df: from test_res = cpacas.sweep_test_coswitch_wrapper, test_res['test_res_df']
    2 x 2 lineplot+ ci shade; row: on vs off; col: large window vs small window
    '''
    fig,axs=plt.subplots(2,2,figsize=(8,8))
    window_thresh=10
    for ii,onoff in enumerate([1,-1]):
        test_res_df_onoff = test_res_df.loc[onoff]
        test_res_df_onoff_largewindow = test_res_df_onoff.query('window_low>@window_thresh')
        fig,ax=plot_pairwise_coswitch_test_result_one(test_res_df_onoff_largewindow,normalize_by_key=normalize_by_key,fig=fig,ax=axs[ii,0],plot_kws=plot_kws)
        ax.set_title(f'onoff={onoff}, large window')
        test_res_df_onoff_smallwindow = test_res_df_onoff.query('window_low<=@window_thresh')
        fig,ax=plot_pairwise_coswitch_test_result_one(test_res_df_onoff_smallwindow,normalize_by_key=normalize_by_key,fig=fig,ax=axs[ii,1],plot_kws=plot_kws)
#         ax.set_xlim([0,10])
        ax.set_title(f'onoff={onoff}, small window')
    plt.tight_layout()
    return fig,axs
    