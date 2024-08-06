import numpy 
import scipy
from scipy.stats import rankdata
from scipy.signal import find_peaks
import data_prep_new as dpn
import place_cell_analysis as pa
import plot_helper as ph
from importlib import reload
import itertools, sys, os, copy, pickle
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import jax
import jax.numpy as np
import jax.scipy.stats as stats
from jax import value_and_grad, grad, jit, vmap, jacfwd, jacrev
from jax.example_libraries import optimizers as jax_opt
import submitit

import blackjax
from jax import lax

import gm
reload(gm)
import functools
from scipy.ndimage import gaussian_filter1d

# EPS = 1e-10



from math_functions import *
import math_functions as mf
reload(mf)

'''
log here just means the result of some inverse transform from positive R to R, not necessarily the log of sth
'''

def find_dt(fr):
    '''
    helper function for robustly getting the dt for the binned spike count df
    '''
    st = 0
    while fr['times'].index[st] - fr['times'].index[st+1] > 1:
        st+=1
    dt = fr['times'].iloc[st+1] - fr['times'].iloc[st] # careful 
    return dt

# def forward(regressors,pars,sigma_min=1):
    
#     nfields = pars['logws'].shape[1]
#     trial_inds_int = regressors['trial_inds_int']
    
#     # mus_l = pars['mus'][trial_inds_int]
#     # logsigmas_l = pars['logsigmas'][trial_inds_int]
#     mus = pars['mus']
#     logsigmas = pars['logsigmas']
#     logws_l = pars['logws'][trial_inds_int]
#     logb_l = pars['logb'][trial_inds_int]
#     # b_l = pars['b'][trial_inds_int]

#     fs_all_trial_one_field_l=[] # can also do +=
#     for k in range(nfields):
#         # fs_all_trial_one_field = stats.norm.pdf(regressors['position'],loc=mus_l[:,k],scale=softplus(logsigmas_l[:,k]))
#         fs_all_trial_one_field = unnormalized_normal_pdf(regressors['position'],loc=mus[:,k],scale=softplus(logsigmas[:,k])+sigma_min)
#         fs_all_trial_one_field_l.append(fs_all_trial_one_field)
#     fs_all_trial_one_field_l_stacked = np.array(fs_all_trial_one_field_l)
#     ws_l = softplus(logws_l) # make sure ws nonnegative
#     fs_l = np.einsum('kt,tk->t',fs_all_trial_one_field_l_stacked,ws_l)
#     fs_l_final = (fs_l + softplus(logb_l).flatten()) * regressors['dt'] 
    
#     return fs_l_final


# def forward(regressors,pars_trans,sigma_min=1):
#     '''
#     updated: use pars_trans
#     '''
#     pars = pars_invtransform(pars_trans,regressors)
#     nfields = pars['mus'].shape[1]
#     trial_inds_int = regressors['trial_inds_int']
#     trial_type = regressors['trial_type']
    
#     # mus_l = pars['mus'][trial_inds_int]
#     # logsigmas_l = pars['logsigmas'][trial_inds_int]
#     mus = pars['mus'][trial_type]
#     # logsigmas = pars['logsigmas']
#     sigmas = pars['sigmas'][trial_type]
#     # logws_l = pars['logws'][trial_inds_int]
#     ws_l = pars['ws'][trial_inds_int]
#     # logb_l = pars['logb'][trial_inds_int]
#     b_l = pars['b'][trial_inds_int]
    

#     fs_all_trial_one_field_l=[] # can also do +=
#     for k in range(nfields):
#         # fs_all_trial_one_field = stats.norm.pdf(regressors['position'],loc=mus_l[:,k],scale=softplus(logsigmas_l[:,k]))
#         fs_all_trial_one_field = unnormalized_normal_pdf(regressors['position'],loc=mus[:,k],scale=sigmas[:,k]+sigma_min)
#         fs_all_trial_one_field_l.append(fs_all_trial_one_field)
#     fs_all_trial_one_field_l_stacked = np.array(fs_all_trial_one_field_l)
#     # ws_l = softplus(logws_l) # make sure ws nonnegative
#     fs_l = np.einsum('kt,tk->t',fs_all_trial_one_field_l_stacked,ws_l)
#     # fs_l_final = (fs_l + softplus(logb_l).flatten()) * regressors['dt'] 
#     fs_l_final = (fs_l + b_l.flatten()) * regressors['dt'] 
    
#     return fs_l_final

@jit
def forward(regressors,pars_trans,nfields_mask,sigma_min=1):
    '''
    updated: use pars_trans
    nfields_mask: bool jax np array, mask out the non contributing fields; necessary because jit vmap require static arrays, so all pars have to have the same max_nfields at initialization
    '''
    pars = pars_invtransform(pars_trans,regressors)
    trial_inds_int = regressors['trial_inds_int']
    trial_type = regressors['trial_type']
    
    # mus_l = pars['mus'][trial_inds_int]
    # logsigmas_l = pars['logsigmas'][trial_inds_int]
    mus = pars['mus'][trial_type]
    # logsigmas = pars['logsigmas']
    sigmas = pars['sigmas'][trial_type]
    # logws_l = pars['logws'][trial_inds_int]
    ws_l = pars['ws'][trial_inds_int]
    # logb_l = pars['logb'][trial_inds_int]
    b_l = pars['b'][trial_inds_int]
    
    
    output_one_field_func = lambda mu,sigma,w:w*unnormalized_normal_pdf(regressors['position'],loc=mu,scale=sigma+sigma_min)
    fr_all_trial_one_field_l=vmap(jit(output_one_field_func),in_axes=(1,1,1))(mus,sigmas,ws_l)
    fr_all_trial_one_field_l_masksum = np.dot(nfields_mask,fr_all_trial_one_field_l)
    fs_l_final = (fr_all_trial_one_field_l_masksum + b_l.flatten()) * regressors['dt']
    
    
    return fs_l_final


def get_regressors(fr,regressors_={}):
    regressors={}
    regressors['position'] = np.array(fr['lin'])
    regressors['trial_inds_int'] = fr['trial'].values.astype(int)
    regressors['dt'] = find_dt(fr)
    regressors['trial_type_inds'] = dict(fr[['visitedArm','trial']].groupby('visitedArm').apply(lambda x:x['trial'].unique().astype(int))) # {0.0:[trial inds...],1.0:[]}
    regressors['trial_type'] = fr['visitedArm'].values.astype(int)
    regressors.update(regressors_)
    return regressors

def subselect_regressors(regressors, target_spk_allneurons, trial_type=0,**kwargs):
    tt = trial_type
    mask = regressors['trial_type']==tt
    regressors_one_trial={}
    regressors_one_trial['position'] = regressors['position'][mask]
    regressors_one_trial['trial_inds_int'] = regressors['trial_inds_int'][mask]
    nunique_trials = len(np.unique(regressors_one_trial['trial_inds_int']))
    regressors_one_trial['trial_inds_int'] = np.array(rankdata(regressors_one_trial['trial_inds_int'],'dense')-1) # don't want the index within the whole trial
    # regressors_one_trial['trial_type_inds'] = {tt:regressors['trial_type_inds'][tt]}
    regressors_one_trial['trial_type_inds'] = {tt:np.arange(nunique_trials)}
    regressors_one_trial['trial_type'] = regressors['trial_type'][mask]
    regressors_one_trial['dt'] = regressors['dt']
    target_spk_one_trial = target_spk_allneurons[mask]

    return regressors_one_trial, target_spk_one_trial


@jit
def get_parbar(par_init,regressors):
    '''
    to use the gaussian_logprior, need parbar for each par, so add this to the init param
    '''
    # for k in ['b','sigmas','ws']:
    #     par_init[f'{k}_bar'] = np.mean(softplus(par_init[f'log{k}']),axis=0)
    # par_init['mus_bar'] = np.mean(par_init['mus'],axis=0)
    # for k in ['b','ws']:
    #     par_init[f'{k}_bar'] = np.mean(softplus(par_init[f'log{k}']),axis=0,keepdims=True)
    par_init[f'b_bar'] = np.mean(softplus(par_init[f'logb']),axis=0,keepdims=True) # for baseline, average across all trials
    
    ws_bar=[]
    for tt, inds in regressors['trial_type_inds']:
        ws_bar.append(np.mean(softplus(par_init[f'logws']),axis=0,keepdims=True)) # for baseline, average across all trials
    par_init[f'ws_bar'] = np.array(ws_bar)
    return par_init
    

# def logprior(regressors,pars_trans,reg_pars,nfields_mask,reg_type='gaussian_logprior_laplacian',mask=None,uncentered=True):
# def logprior(regressors,pars_trans,reg_pars,nfields_mask,reg_type=0,mask=None,uncentered=True):
#     '''
#     notice this should have the opposite sign as in gm
#     updated: use pars_trans; 
#     '''
#     pars = pars_invtransform(pars_trans,regressors)
#     R_sigma_thresh=force_sigma_big_prior(reg_pars['g_sigma_thresh'],reg_pars['sigma_thresh'],pars['sigmas'],nfields_mask)
#     R_order = order_prior(reg_pars['g_order'],pars['mus'],nfields_mask)
#     R_sigma_shrinkage = sigma_shrinkage_prior(reg_pars['g_sigma_shrinkage'],pars['sigmas'],nfields_mask)
    
#     R_base = R_sigma_thresh + R_order + R_sigma_shrinkage

#     # if (reg_type=='gaussian_logprior_laplacian') and (not uncentered):
#     if (reg_type==0) and (not uncentered):
#         # R_mu_var = 1/reg_pars['g_mu'] * gaussian_logprior_laplacian(pars['mus'],pars['mus_bar'])
#         # R_sigma_var = 1/reg_pars['g_sigma'] * gaussian_logprior_laplacian(softplus(pars['logsigmas']),pars['sigmas_bar'])
#         R_w_var = 1/reg_pars['g_w'] * gaussian_logprior_laplacian(softplus(pars['logws']),pars['ws_bar'])
#         # R_b_var = 1/reg_pars['g_b'] * gaussian_logprior_laplacian(pars['b'])
#         R_b_var = 1/reg_pars['g_b'] * gaussian_logprior_laplacian(softplus(pars['logb']),pars['b_bar'])
#         # return R_mu_var + R_sigma_var + R_w_var+ R_b_var + R_sigma_thresh
        
#         return R_base + R_w_var+ R_b_var 
#     # if (reg_type=='gaussian_logprior_laplacian') and (uncentered):
#     if (reg_type==0) and (uncentered):
#         R_w_var = 0.
#         for inds in regressors['trial_type_inds'].values():
#             R_w_var += 1/reg_pars['g_w'] * gaussian_logprior_laplacian(pars_trans['delta_ws'][inds],0,nfields_mask) # notice the other if has not been updated to include nfields_mask as an argument
#         R_b_var = 1/reg_pars['g_b'] * gaussian_logprior_laplacian(pars_trans['delta_b'],0,np.array([1])) # for b the mask is just a single True
        
        
#         return R_base + R_w_var+ R_b_var 

#     else:
#         return R_base


def logprior(regressors,pars_trans,reg_pars,nfields_mask,reg_type=0,mask=None,uncentered=True):
    '''
    notice this should have the opposite sign as in gm
    updated: use pars_trans; 
    '''
    pars = pars_invtransform(pars_trans,regressors)
    R_sigma_thresh=force_sigma_big_prior(reg_pars['g_sigma_thresh'],reg_pars['sigma_thresh'],pars['sigmas'],nfields_mask)
    R_order = order_prior(reg_pars['g_order'],pars['mus'],nfields_mask)
    R_sigma_shrinkage = sigma_shrinkage_prior(reg_pars['g_sigma_shrinkage'],pars['sigmas'],nfields_mask)
    
    R_base = R_sigma_thresh + R_order + R_sigma_shrinkage

    
    R_w_var = 0.
    for inds in regressors['trial_type_inds'].values():
        R_w_var += 1/reg_pars['g_w'] * gaussian_logprior_laplacian(pars_trans['delta_ws'][inds],0,nfields_mask) # notice the other if has not been updated to include nfields_mask as an argument
    R_b_var = 1/reg_pars['g_b'] * gaussian_logprior_laplacian(pars_trans['delta_b'],0,np.array([1])) # for b the mask is just a single True
        
        
    return R_base + R_w_var+ R_b_var 


@jit
def logpdf_poisson_reduce(fs_l_final,target,mask):
    logpdf_element = stats.poisson.logpmf(target,fs_l_final)
    # logpdf = np.sum(logpdf_element * mask) 
    logpdf = np.sum(logpdf_element * mask) /np.sum(mask)
    return logpdf
@jit
def logpdf_gaussian_reduce(fs_l_final,target,mask):
    logpdf_element = -1/2 * (fs_l_final - target)**2
    logpdf = np.sum(logpdf_element*mask)
    return logpdf

# def logpdf_no_reg(regressors,pars,target,nfields_mask,mask=None,noise_type="poisson"):
def logpdf_no_reg(regressors,pars,target,nfields_mask,mask=None,noise_type=0):
    if mask is None:
        mask = np.ones_like(target)
    fs_l_final = forward(regressors,pars,nfields_mask)
    # if noise_type==0:
    # # if noise_type=='poisson':
    #     logpdf = logpdf_poisson_reduce(fs_l_final,target,mask)
    # elif noise_type==1:
    #     logpdf = logpdf_gaussian_reduce(fs_l_final,target,mask)

    logpdf = lax.cond(noise_type==0,logpdf_poisson_reduce,logpdf_gaussian_reduce,fs_l_final,target,mask)

    return logpdf

def logpdf(regressors,pars,target,nfields_mask,mask=None,noise_type=0,reg_pars=None,reg_type=None):
    loglikelihood = logpdf_no_reg(regressors,pars,nfields_mask,target,mask=mask,noise_type=noise_type)
    
    if mask is None:
        mask = np.ones_like(target)
    lpr = logprior(regressors,pars,reg_pars,nfields_mask,reg_type=reg_type,mask=mask) 
    # lpr = logprior(pars,reg_pars,reg_type=reg_type,mask=mask) / mask.sum()
        
    return loglikelihood + lpr

def negative_logpdf_no_reg(regressors,pars,target,nfields_mask,mask=None,loss_type=0,reg_pars=None,reg_type=None):
    logpdf=logpdf_no_reg(regressors,pars,target,nfields_mask,mask=mask,noise_type=loss_type)
    return -logpdf
    
# def negative_logpdf(regressors,pars,target,nfields_mask,mask=None,loss_type="poisson",reg_pars=None,reg_type=None):
def negative_logpdf(regressors,pars,target,nfields_mask,mask=None,loss_type=0,reg_pars=None,reg_type=0):
    loss=negative_logpdf_no_reg(regressors,pars,target,nfields_mask,mask=mask,loss_type=loss_type)
    # R = gm.regularization(pars,reg_pars,reg_type=reg_type,mask=mask)
    

    if mask is None:
        mask = np.ones_like(target)
    R = -logprior(regressors,pars,reg_pars,nfields_mask,reg_type=reg_type,mask=mask)
    # R = -logprior(pars,reg_pars,reg_type=reg_type,mask=mask) / mask.sum()
    
    return loss + R 


def fit(regressors,pars,target,mask=None,sampler_func=blackjax.mala,sampler_kw={'step_size':1e-5},num_samples=1000,rng_key_int=0,reg_pars=None,reg_type=None):
    rng_key = jax.random.PRNGKey(rng_key_int)
    logprob = lambda x:logpdf_no_reg(regressors,x,target,mask)
    
    sampler = sampler_func(logprob,**sampler_kw)
    initial_state=sampler.init(pars)
    kernel = jax.jit(sampler.step)
    
    @jax.jit
    def one_step(state,rng_key):
        state, _ =kernel(rng_key,state)
        return state, state
    
    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)
    return states

def fit_multichain(regressors,pars_multi,target,reg_pars=None,reg_type=None,mask=None,sampler_func=blackjax.mala,sampler_kw={'step_size':1e-5},num_samples=1000,rng_key_int=0):
    rng_key = jax.random.PRNGKey(rng_key_int)
    # logprob = lambda x:logpdf_no_reg(regressors,x,target,mask)
    logprob = lambda x:logpdf(regressors,x,target,mask,reg_pars=reg_pars,reg_type=reg_type)
    check_key = 'logws' if 'logws' in pars_multi.keys() else 'delta_ws' # key for checking dimension of the params
    if len(pars_multi[check_key].shape)==3:
        num_chains = pars_multi[check_key].shape[0]
    else:
        print('not enough dim for multi chain')
        return
    try:
        sampler = sampler_func(logprob,**sampler_kw)
    except: # for some sampler it gives an unexpected keyword error
        sampler = sampler_func(logprob,*sampler_kw.values())
    initial_states = jax.vmap(sampler.init, in_axes=(0))(pars_multi)
    
    kernel = jax.jit(sampler.step)
    
    @jax.jit
    def one_step(states,rng_key):
        keys = jax.random.split(rng_key, num_chains)
        states, info =jax.vmap(kernel)(keys,states)
        return states, (states,info)
    
    keys = jax.random.split(rng_key, num_samples)
    _, (states,infos) = jax.lax.scan(one_step, initial_states, keys)
    return states,infos


# def random_init_jax(rngkey,regressors,nfields_max=5,init_max_w = 100., init_max_sigma = 50.,init_max_b = 5.,init_max_mu = 200.,init_max_deltaw=20.,init_max_deltab=2.,dobar=True, uncentered=True):
def random_init_jax(rngkey,regressors,nfields_max=5,init_max_w = 40., init_max_sigma = 50.,init_max_b = 5.,init_max_mu = 200.,init_max_deltaw=4.,init_max_deltab=2.,dobar=True, uncentered=True):
    '''
    replacing the old gm.init_all_trials, to use rng keys in jax, easier for vmap and multichain
    dobar: initialize with the param_bar that is useful for the gaussian prior
    '''
    ntrial_types = len(regressors['trial_type_inds'].keys())
    # ntrials = len(numpy.unique(regressors['trial_inds_int']))
    ntrials = regressors['ntrials']
    
    
    def init_mus(key,ntrial_types):
        mu_segments = np.linspace(0,init_max_mu,nfields_max+1)
        mus = []
        for k in range(nfields_max):  
            uni = jax.random.uniform(key,(ntrial_types,))*(mu_segments[k+1]-mu_segments[k] )+mu_segments[k]  #same uniform random var but in different segments to ensure seperate init
            mus.append(uni)   
        
        # return np.sort(np.array(mus))[None,:]
        return np.sort(np.array(mus).T)
    if uncentered:
        pars_trans = {}
        keys = jax.random.split(rngkey,num=6)
        pars_trans['delta_ws']= jax.random.uniform(keys[0],(ntrials,nfields_max)) * 2*init_max_deltaw - init_max_deltaw
        pars_trans['delta_b']= jax.random.uniform(keys[1],(ntrials,1)) * init_max_deltab * 2 - init_max_deltab

        pars_trans['logws_bar']=inv_softplus(jax.random.uniform(keys[2],(ntrial_types,nfields_max)) * init_max_w) 
        pars_trans['logb_bar']=inv_softplus(jax.random.uniform(keys[3],(1,1)) * init_max_b) 
        pars_trans['logsigmas']=inv_softplus(jax.random.uniform(keys[4],(ntrial_types,nfields_max)) * init_max_sigma)  
        pars_trans['mus'] = init_mus(keys[5],ntrial_types)
        
        return pars_trans

    else:
        pars={}
        keys = jax.random.split(rngkey,num=4)
        pars['logws']=inv_softplus(jax.random.uniform(keys[0],(ntrials,nfields_max)) * init_max_w) 

        pars['logsigmas']=inv_softplus(jax.random.uniform(keys[1],(1,nfields_max)) * init_max_sigma)  
        pars['logb'] = inv_softplus(jax.random.uniform(keys[2],(ntrials,1)) * init_max_b)

        
        pars['mus']=init_mus(keys[3])

        if dobar:
            pars=get_parbar(pars,regressors)
        return pars

random_init_jax_allneurons = vmap(random_init_jax,in_axes=(0, None, None)) # note we migth need to change the named arguments defaults


def pars_transform(pars,regressors):
    pars_trans = {}
    pars_trans['logb_bar'] = inv_softplus(pars['b_bar'])
    pars_trans['delta_b'] = inv_softplus(pars['b']) - pars_trans['logb_bar']

    pars_trans['logws_bar'] = inv_softplus(pars['ws_bar'])
    pars_trans['delta_ws'] = np.zeros_like(pars['ws'])
    for ii,inds in regressors['trial_type_inds'].items():
        delta = inv_softplus(pars['ws'][inds]) - pars_trans['logws_bar'][[int(ii)]]
        pars_trans['delta_ws']=pars_trans['delta_ws'].at[inds].set(delta)
    
    pars_trans['mus'] = pars['mus']
    pars_trans['logsigmas'] = inv_softplus(pars['sigmas'])
    

    return pars_trans

def pars_invtransform(pars_trans,regressors):
    pars = {}
    pars['mus'] = pars_trans['mus']

    pars['sigmas'] = softplus(pars_trans['logsigmas'])

    pars['ws_bar'] = softplus(pars_trans['logws_bar'])
    pars['ws'] = np.zeros_like(pars_trans['delta_ws'])
    for ii,inds in regressors['trial_type_inds'].items():
        ws = softplus(pars_trans['logws_bar'][int(ii)] + pars_trans['delta_ws'][inds])
        pars['ws']=pars['ws'].at[inds].set(ws)
        
    # pars['ws'] = softplus(pars_trans['logws_bar'] +  pars_trans['delta_ws']) 

    pars['b_bar'] = softplus(pars_trans['logb_bar'])
    pars['b'] = softplus(pars_trans['logb_bar'] + pars_trans['delta_b'])

    return pars


def gm_func_by_trial(regressors,pars_trans,sigma_min=1):
    '''
    pars:
        logws/: ntrial x Kfields;    amplitude
        mus/logsigmas_l:1 x Kfields; center/width
        b_l: ntrial x 1;      baseline
        S: npos x ntrial;    sparse element

    regressors:
        xs: np.arange(npos);    probe positions
        trial_type_by_trial: vector of length trial, indicating the index of trialtype; for selecting the appropriate mu and sigma
        # occupancy_in_bins:
    ======
    fs: npos x ntrial
    '''

    pars = pars_invtransform(pars_trans,regressors)
    
    
    
    mus = pars['mus'][regressors['trial_type_by_trial']]
    sigmas = pars['sigmas'][regressors['trial_type_by_trial']]
    ws_l = pars['ws']
    b_l = pars['b']
    
    
    
    ntrials = ws_l.shape[0]

    xs = regressors['xs']
    if 'S' in pars.keys():
        S = pars['S']
    else:
        S = np.zeros((len(xs),ntrials))
    assert len(xs)==S.shape[0]

    nt,K = ws_l.shape
    xs_l = np.tile(xs[:,None],[1,nt])
    fs_all_trial_one_field_l = []
    
    for k in range(K):
        
        # fs_all_trial_one_field=scipy.stats.norm.pdf(xs_l,loc=mus_l[:,k],scale=softplus(logsigmas_l[:,k]))
        fs_all_trial_one_field = unnormalized_normal_pdf(xs_l,loc=mus[:,k],scale=sigmas[:,k]+sigma_min)
        assert fs_all_trial_one_field.shape == (xs.shape[0],nt)
        fs_all_trial_one_field_l.append(fs_all_trial_one_field)
    fs_all_trial_one_field_l_stacked = np.array(fs_all_trial_one_field_l)
    # ws_l = softplus(logws_l) # make sure ws nonnegative
    fs_l = np.einsum('kpn,nk->pn',fs_all_trial_one_field_l_stacked,ws_l)

    # fs_l_final = fs_l + b_l[None,:] + S
    # fs_l_final = fs_l + b_l[None,:] 
    # fs_l_final = fs_l + softplus(logb_l[None,:])
    fs_l_final = fs_l + b_l.flatten()
    fs_l_final = fs_l_final
    return fs_l_final

@jit
def train_adam_schedule(regressors,pars,ys_l_smthed,reg_pars,nfields_mask,reg_type=0,lr_l=[0.1],niters_l=[100],smthwin_l=[30],mask=None):
    '''
    wrapper of train_adam for scheduling
    especially, smooth the target first and fit using mse, then reduce to poisson loss
    '''
    pars_curr = pars
    train_adam_mse=lambda pars_curr,ys_l_filt,lr,niters:train_adam(regressors,pars_curr,ys_l_filt,reg_pars,nfields_mask,loss_type=1,reg_type=reg_type,lr=lr,niters=niters,mask=mask) # if smoothing use gaussian
    train_adam_poisson=lambda pars_curr,ys_l_filt,lr,niters:train_adam(regressors,pars_curr,ys_l_filt,reg_pars,nfields_mask,loss_type=0,reg_type=reg_type,lr=lr,niters=niters,mask=mask) # when smthwin too low, signal using poisson loss
    
    def conditional_train(pars_curr,operands):
        ys_l_filt, lr, niters,smthwin=operands
        # pars_curr,loss_l=lax.cond(smthwin >1, train_adam_mse, train_adam_poisson, pars_curr, ys_l_filt, lr, niters)
        pars_curr=lax.cond(smthwin >1, train_adam_mse, train_adam_poisson, pars_curr, ys_l_filt, lr, niters)
        return pars_curr,None#loss_l

    # for smthwin,lr,niters in zip(smthwin_l,lr_l, niters_l):
    # for ys_l_filt,smthwin,lr,niters in zip(ys_l_smthed,smthwin_l,lr_l, niters_l):
    #     # ys_l_filt = gaussian_filter1d(ys_l.astype(float) ,smthwin,mode='constant',cval=0)
    #     # ys_l_filt = mf.smooth(ys_l.astype(float), smthwin) # from math_functions
    #     # if smthwin >1:
    #     #     pars_curr = train_adam(regressors,pars_curr,ys_l_filt,reg_pars,nfields_mask,loss_type=1,reg_type=reg_type,lr=lr,niters=niters,mask=mask) # if smoothing use gaussian
    #     # else:
    #     #     pars_curr = train_adam(regressors,pars_curr,ys_l_filt,reg_pars,nfields_mask,loss_type=0,reg_type=reg_type,lr=lr,niters=niters,mask=mask)  # when smthwin too low, signal using poisson loss
    #     pars_curr=lax.cond(smthwin>1, train_adam_mse, train_adam_poisson, pars_curr, ys_l_filt, lr, niters)
    # return pars_curr

    # pars_learned = lax.fori_loop(0,n_schedule,conditional_train,pars)
    # pars_learned,loss_l_l=lax.scan(conditional_train,pars,(ys_l_smthed, lr_l, niters_l,smthwin_l))
    pars_learned,_=lax.scan(conditional_train,pars,(ys_l_smthed, lr_l, niters_l,smthwin_l))
    return pars_learned#, loss_l_l

@jit
def train_adam(regressors,pars,ys_l,reg_pars,nfields_mask,loss_type=0,reg_type=0,lr=0.1,niters=100,mask=None):
    '''
    # for efficiency and jittability, fix the loss function; use fori_loop; unable to return loss_l
    '''
    opt_init,opt_update,get_params=jax_opt.adam(lr)
    func = negative_logpdf
    # func = negative_logpdf_no_reg  
    @jit
    def train_step(step_i,opt_state):
        params=get_params(opt_state)
        loss,grads = value_and_grad(func,argnums=1)(regressors,params,ys_l,nfields_mask,reg_pars=reg_pars,loss_type=loss_type,reg_type=reg_type,mask=mask) # notice the params here
        return opt_update(step_i, grads, opt_state)

    # def train_step(states,inp):
    #     opt_state,step_i=states
    #     params = get_params(opt_state)
    #     loss,grads = value_and_grad(func,argnums=1)(regressors,params,ys_l,nfields_mask,reg_pars=reg_pars,loss_type=loss_type,reg_type=reg_type,mask=mask) # notice the params here
    #     return (opt_update(step_i, grads, opt_state), step_i+1), loss
    
    opt_state=opt_init(pars)
    opt_state= jax.lax.fori_loop(0,niters,train_step,opt_state)
    # states,loss_l = jax.lax.scan(train_step,(opt_state,0),None,length=niters)
    # opt_state = states[0]

    return get_params(opt_state)#, loss_l

train_adam_allneurons_same_regpars = vmap(train_adam,in_axes=(None,0,1,None,None,None,None,None,None,None))
# train_adam_allneurons_diff_regpars
train_adam_allneurons_same_regpars_schedule = vmap(train_adam_schedule,in_axes=(None,0,2,None,None,None,None,None,None,None))

def smooth_target(target, smthwin_l):
    '''
    target: ntimes, or ntimes x nneurons
    smthwin_l: n_smthwin

    target_smth: n_smthwin x ntimes x nneurons
    '''
    target_smth = []
    for win in smthwin_l:
        target_smth.append(gaussian_filter1d(target,win,mode='constant',cval=0,axis=0))
    target_smth = np.array(target_smth)
    return target_smth