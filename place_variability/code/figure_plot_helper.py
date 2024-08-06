import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import sys,os,copy,pdb,importlib,pickle
from importlib import reload
import pandas as pd

#### figure 1 ###
import matplotlib
def plot_exp_var_ratio_vs_shuffle(best_n_all,pval_all=None,best_pval_all=None):
    if best_pval_all is None:
        best_n_no_zero_all=pval_all.loc[:,(slice(None),'pval')].droplevel(axis=1,level=1).idxmin(axis=1)
        best_pval_all = pval_all.apply(lambda x:x[best_n_no_zero_all[x.name]],axis=1)
    
    fig,ax=plt.subplots(figsize=(8,6))
    ax.set_aspect('equal')
    issig_all = best_n_all > 0
    c = [matplotlib.colors.to_rgb('C0') if i else matplotlib.colors.to_rgb('C0') for i in issig_all]
    c_dict = {True:'C3',False:'C0'}
    label_d = {True:'significant',False:'not sig.'}
    for tf in [False,True]:
        ma = issig_all == tf
        best_pval_ma_all = best_pval_all.loc[ma]
    #     errorbar=best_pval_ma[['exp_var_ratio_sh_low','exp_var_ratio_sh_high']].T - best_pval_ma['exp_var_ratio_sh_med']
    #     errorbar = [-errorbar.iloc[0],errorbar.iloc[1]]
        c = c_dict[tf]
        ax.scatter(best_pval_ma_all['exp_var_ratio_sh_med'],best_pval_ma_all['exp_var_ratio'],
                c=c,label=label_d[tf],alpha=1.,s=0.1,
               )
    ax.plot([0,1],[0,1],color='k',linestyle=':',label='data=shuffle')
    ax.plot([0,0.5],[0,1],color='k',linestyle='-.',label='2-fold increase')
    ax.plot([0,1/3],[0,1],color='k',linestyle='--',label='3-fold increase')
    ax.spines[['top','right']].set_visible(False)
    ax.set_xlabel('Explained variance ratio (shuffled)')
    ax.set_ylabel('Explained variance ratio')
    legend=ax.legend(bbox_to_anchor=[1,1.05],fontsize=15)
    legend.legendHandles[0]._sizes = [30]
    legend.legendHandles[1]._sizes = [30]
    
    return best_pval_all, fig,ax


