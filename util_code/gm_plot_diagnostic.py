'''
For disagnostic plots like checking cross validation
'''

# %%
import numpy 
import scipy
from scipy.signal import find_peaks
import data_prep_new as dpn
import place_cell_analysis as pa
import plot_helper as ph
import plot_raster as pr
from importlib import reload
import itertools, sys, os, copy, pickle,pathlib
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

import jax
import jax.numpy as np
import jax.scipy as scipy
from jax import value_and_grad, grad, jit, vmap, jacfwd, jacrev
from jax.example_libraries import optimizers as jax_opt
import submitit
import gm
reload(gm)
import tqdm

#%%
def plot_masks(mask,displacement,xs=None,fig=None,ax=None,bin_to_lin=None):
    if ax is None:
        fig,ax=plt.subplots()
    if xs is None:
        xs = gm.get_regressor()['xs']
    ntrials = mask.shape[1]
    y_baseline_all_trials=np.arange(-displacement/2,(ntrials+1)*displacement,displacement)
    if bin_to_lin is not None:
        xs = bin_to_lin[xs.astype(int)]
    for tr in range(ntrials):
        ax.fill_between(xs,y_baseline_all_trials[tr],y_baseline_all_trials[tr+1],where=np.logical_not(mask[:,tr]),alpha=0.3,color='r')

    return fig,ax

def fit_good_bad_trials(ys_l,reg_pars_good,reg_pars_bad,mask_ratio=0.2,mask=None,regressors_={},reg_type='quad_variation',loss_type='mse',niters=4000):
    '''
    ys_l: target for one neuron one trial: npos x ntimes for rate
    reg_pars_good, reg_pars_bad: hyperparms to update for good and bad cv result
    '''
    reg_pars_dict = {'good':reg_pars_good,'bad':reg_pars_bad}
    pars_learned_dict = {}
    if mask is None:
        mask  = gm.get_train_test_mask(ys_l.shape[0],ys_l.shape[1],mask_ratio)
    test_loss_dict={}
    for key,val in reg_pars_dict.items():
        rp_ = val
        nfields=rp_['nfields']
        pars_learned,loss_l,test_loss=gm.train_and_test(ys_l,regressors_=regressors_,reg_type = reg_type,reg_pars_=rp_,nfields=nfields,mask=mask,loss_type=loss_type,niters=niters)
        pars_learned_dict[key] = pars_learned
        test_loss_dict[key] = test_loss
    return pars_learned_dict, test_loss_dict, mask

def get_yhat_multi_fit(pars_learned_dict,regressors_={}):
    '''
    get the ys_hat for each pars_learned in a dictionary

    pars_learned_dict: dictionary of pars_learned
    '''
    ys_hat_dict={}
    regressors=gm.get_regressor(regressors_)
    for key,val in pars_learned_dict.items():
        pars = val
        ys_hat_dict[key] = gm.gm_func_by_trial(regressors,pars)
    return ys_hat_dict

def trial_by_trial_good_bad_comparison(ys_l,ys_hat_dict,tr_l=None,fig=None,ax=None,mask=None):
    ''' 
    trial by trial comparison between good and bad hyperparam fit
    '''
    if tr_l is None:
        tr_l = range(ys_l.shape[1])
    nplots = len(tr_l)
    if ax is None:
        fig,ax=plt.subplots(nplots,1,figsize=(6,nplots*2))
    ymax = ys_l.max()
    for ii,tr in enumerate(tr_l):
        ax[ii].plot(ys_l[:,tr],label='target',linestyle='-')
        plot_masks(mask[:,[tr]],5,fig=fig,ax=ax[ii])
        for key,val in ys_hat_dict.items():
            ys_hat = val
            ax[ii].plot(ys_hat[:,tr],label=key,linestyle=':',linewidth=2)
        ax[ii].legend()
        ax[ii].set_ylim([0,ymax])
        ax[ii].set_title(f'trial={tr}')
        plt.tight_layout()
    return fig,ax
