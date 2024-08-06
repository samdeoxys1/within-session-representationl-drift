import numpy 
import scipy
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
import jax.scipy as scipy
from jax import value_and_grad, grad, jit, vmap, jacfwd, jacrev, lax
from jax.example_libraries import optimizers as jax_opt
import submitit


from math_functions import *

# par lives in the constrained domain, whereas par_trans is unconstrained and therefore directly optimized
# x_bar: in par, trial average of x; in par_trans: trial average of inv_softplus(x)
# make sure x = softplus(trans[delta_x] + trans[x_bar]), this way both delta_x and x_bar in trans can be unconstrained
# so x_bar in par and trans are not directly related! x_bar in par only there for completeness sake

def par_transform(par):
    to_trans_from_delta = ['b','ws','sigmas','mus']
    do_softplus_dict = {'b':True,'ws':True,'sigmas':True,'mus':False}
    par_trans = {}
    for k in to_trans_from_delta:
        if do_softplus_dict[k]:            
            # par_trans[f'{k}_bar'] = inv_softplus(par[f'{k}_bar'])
            par_trans[f'{k}_bar'] = np.mean(inv_softplus(par[f'{k}']),axis=0,keepdims=True)
            par_trans[f'{k}_delta'] = inv_softplus(par[k]) - par_trans[f'{k}_bar']
        else:
            par_trans[f'{k}_bar'] = par[f'{k}_bar']
            par_trans[f'{k}_delta'] = par[k] - par_trans[f'{k}_bar']
    return par_trans


def par_invtransform(par_trans,common_mu=False,common_sigma=False):
    par = {}
    to_trans_from_delta = ['b','ws','sigmas','mus']
    do_softplus_dict = {'b':True,'ws':True,'sigmas':True,'mus':False}
    

    @jit
    def set_0_func(x):
        return x.at[:,:].set(0.)
    @jit
    def do_nothing_func(x):
        return x
    
    par_trans['mus_delta'] = lax.cond(common_mu,set_0_func,do_nothing_func,par_trans['mus_delta'])
    par_trans['sigmas_delta'] = lax.cond(common_sigma,set_0_func,do_nothing_func,par_trans['sigmas_delta'])
    
    # par_trans['mus_delta'] = set_0_func(par_trans['mus_delta'])
    # par_trans['sigmas_delta'] = set_0_func(par_trans['sigmas_delta'])
    
    
    for k in to_trans_from_delta:
        if do_softplus_dict[k]:
            par[k] = softplus(par_trans[f'{k}_bar']+par_trans[f'{k}_delta'])    
            # par[f'{k}_bar'] = softplus(par_trans[f'{k}_bar'])
            par[f'{k}_bar'] = np.mean(par[k],axis=0,keepdims=True)
        else:
            par[k] = par_trans[f'{k}_bar']+par_trans[f'{k}_delta']
            par[f'{k}_bar'] = par_trans[f'{k}_bar']
    

    return par

## jax version
def init(rngkey,regressors,nfields, 
    range_dict={'ws':[0,40],'b':[0,4],'sigmas':[3,10],'mus':[10,90]}):
    ntrials = regressors['ntrials']
    def init_mus():
        mu_segments = np.linspace(regressors['xs'][0],regressors['xs'][-1],nfields+1)
        mus = []
        for k in range(nfields):  
            # uni = numpy.random.rand(ntrials)*(mu_segments[k+1]-mu_segments[k] )+mu_segments[k]  #same uniform random var but in different segments to ensure seperate init
            uni = jax.random.uniform(rngkey,(ntrials,))*(mu_segments[k+1]-mu_segments[k] )+mu_segments[k]  #same uniform random var but in different segments to ensure seperate init
            mus.append(uni)   
        
        return np.sort(np.array(mus).T)
        # return numpy.sort(numpy.array(mus).T)
    par = {}
    # par['mus'] = init_mus()
    for k,rg in range_dict.items():
        if k!='b':
            nfeats = nfields
        else:
            nfeats = 1
        # par[k] = numpy.random.rand(ntrials,nfeats) * (rg[1]-rg[0]) + rg[0]
        par[k] = jax.random.uniform(rngkey,(ntrials,nfeats)) * (rg[1]-rg[0]) + rg[0]
    
    par['mus'] = np.sort(par['mus'],axis=1) # important when not initializing by segments!

    # var_to_be_averaged=list(range_dict.keys()) + ['mus']
    var_to_be_averaged=list(range_dict.keys()) 

    for k in var_to_be_averaged:
        # par[f'{k}_bar'] = numpy.mean(par[k],axis=0,keepdims=True)
        par[f'{k}_bar'] = np.mean(par[k],axis=0,keepdims=True)

    par_trans = par_transform(par)
    return par, par_trans

def list_of_dict_into_pytree(ll):
    '''
    convert a list of dictionary of jnp arrays into a pytree
    '''
    tree= {}
    for k in ll[0].keys():
        tree[k]=np.stack([l[k] for l in ll])
    return tree

## np version
# def init_population(regressors,nfields,n_neurons,**kwargs):
#     par_l = []
#     par_trans_l = []
#     for n in range(n_neurons):
#         par,par_trans = init(regressors,nfields,**kwargs)
#         par_l.append(par)
#         par_trans_l.append(par_trans)
#     #reformat into pytrees
#     par_d = list_of_dict_into_pytree(par_l)
#     par_trans_d = list_of_dict_into_pytree(par_trans_l)

#     return par_d, par_trans_d

## jax version
def init_population(rngkey_l,regressors,nfields,**kwargs):
    func = lambda key:init(key,regressors,nfields,**kwargs)
    par_l, par_trans_l = vmap(func)(rngkey_l)
    return par_l, par_trans_l


def get_regressors(regressors_={},nbins=100):
    regressors={}
    xs=np.arange(nbins,dtype=float)
    regressors['xs']=xs
    regressors.update(regressors_)
    return regressors

# !!!temporarily changed par_trans to par, commented out par_invtransform
# and set sigma_min=0
# to test out the gibbs way


@jit
def forward_one_neuron(regressors, par_trans, nfields_mask, common_mu=False, common_sigma=False):
# def forward_one_neuron(regressors, par, nfields_mask, common_mu=False, common_sigma=False):
    SIGMA_MIN =2. # careful!!!

    par = par_invtransform(par_trans,common_mu,common_sigma)
    mus_l = par['mus']
    sigmas_l = par['sigmas']
    ws_l = par['ws']
    b_l = par['b']
    
    ntrials,nfields = ws_l.shape
    
    xs = regressors['xs']
    xs_l = np.tile(xs[:,None],[1,ntrials])

    one_field_func=lambda mus,sigmas,ws:unnormalized_normal_pdf(xs_l,loc=mus,scale=sigmas + SIGMA_MIN) * ws # should be the corresponding mus[:,k], ntrials,
    multi_field_res = vmap(one_field_func,in_axes=(1,1,1))(mus_l,sigmas_l,ws_l)
    fr_all_trial_one_field_l_masksum = np.einsum('f,fpt->pt',nfields_mask,multi_field_res)
    multi_field_res_final = fr_all_trial_one_field_l_masksum + np.squeeze(b_l)
    
    return multi_field_res_final

def gen_nfields_mask_l(nfields_l,nfields_max=4):
    mask_l = numpy.zeros((len(nfields_l),nfields_max)) # nneurons x nfieldsmax
    for nn,nfields in enumerate(nfields_l):
        mask_l[nn,:nfields] = 1
    return np.array(mask_l)

def forward_population(regressors, par_trans_l, nfields_mask_l,common_mu=False,common_sigma=False):
    return vmap(forward_one_neuron,in_axes=(None,0,0,None,None))(regressors,par_trans_l,nfields_mask_l,common_mu,common_sigma)

@jit
def loss_no_reg(target, pred, mask):
    l = mse_no_reduce(pred, target) # n_neurons x n_trials x n_position
    l_masked_summed = np.einsum('npt,pt->n',l,mask)
    l_masked_mean = l_masked_summed / mask.sum()
    return l_masked_mean

@jit
def forward_and_loss_no_reg(target, regressors, pars_trans_l, nfields_mask_l, mask, common_mu=False,common_sigma=False):
    '''
    population
    '''
    if mask is None:
        mask = np.ones_like(target[0]) # the shape of one' neuron's target
    pred = forward_population(regressors, pars_trans_l,nfields_mask_l,common_mu=common_mu,common_sigma=common_sigma)
    l = loss_no_reg(target,pred,mask)

    return pred, l

def get_reg_pars(reg_pars_={}):
    reg_pars = {}
    reg_pars['ws'] = 1.
    reg_pars['sigmas'] = 1.
    reg_pars['mus'] = 1.
    reg_pars['b'] = 1.
    reg_pars['nfields'] = 2
    reg_pars['nfields_max'] = 4

    reg_pars['mus_seperation'] = 1.
    reg_pars['mus_seperation_thresh'] = 10.

    reg_pars['order'] = 1.

    reg_pars.update(reg_pars_)
    reg_pars['nfields_mask'] = np.zeros(reg_pars['nfields_max'])
    reg_pars['nfields_mask'] = reg_pars['nfields_mask'].at[:reg_pars['nfields']].set(1)


    return reg_pars

def get_reg_pars_population(n_neurons,reg_pars_={}):
    '''
    same reg_pars repeated for neurons
    '''
    reg_pars_l = lax.map(lambda x:get_reg_pars(reg_pars_),np.arange(n_neurons))
    # reg_pars_l = vmap(get_reg_pars,in_axes=(None,))(reg_pars_)
    return reg_pars_l




@jit
def quadratic_seperation(mus,thresh,nfields_mask):
    '''
    mus: ntrials x nfields
    '''
    
    r=jax.nn.relu(thresh - np.abs(np.diff(mus,axis=1)) ) #if seperation smaller than threshold, then penalty; otherwise no penalty
    r_masked = r.dot(nfields_mask[1:]) / np.sum(nfields_mask) # eg: if 2 fields, then only one difference

    return r_masked.mean()


@jit
def regularization(par_trans,regressors,reg_pars,common_mu=False, common_sigma=False):
    par = par_invtransform(par_trans,common_mu=common_mu, common_sigma=common_sigma)
    R = 0
    nfields_mask = reg_pars['nfields_mask']
    # quadratic variaton; [!!!] need to make a masked version, right now perhaps ok
    for k in ['mus','sigmas','ws','b']:
        R += reg_pars[k] * quadratic_variation(par[k]) # different from before, larger reg_pars larger penalty
    
    # field seperation
    R += reg_pars['mus_seperation'] * quadratic_seperation(par['mus'],reg_pars['mus_seperation_thresh'],nfields_mask)

    # order:
    R -=order_prior(reg_pars['order'],par['mus'],nfields_mask)

    return R

@jit
def regularization_population(par_trans_l,regressors,reg_pars_l,common_mu=False,common_sigma=False):
    R_l = vmap(regularization,in_axes=(0,None,0,None,None))(par_trans_l,regressors,reg_pars_l,common_mu, common_sigma)
    return R_l

@jit
def forward_and_loss(target,regressors,par_trans_l,reg_pars_l,mask, common_mu=False, common_sigma=False): # just population; if single, then just do population in the single neuron case
    nfields_mask_l = reg_pars_l['nfields_mask']
    pred,l = forward_and_loss_no_reg(target, regressors, par_trans_l, nfields_mask_l,mask,common_mu=common_mu,common_sigma=common_sigma)
    R_l=regularization_population(par_trans_l,regressors,reg_pars_l,common_mu=common_mu,common_sigma=common_sigma)
    return R_l + l
@jit
def forward_and_loss_summed_across_neurons(target,regressors,par_trans_l,reg_pars_l,mask,common_mu=False,common_sigma=False):
    '''
    final loss need to be a scalar
    '''
    l = forward_and_loss(target,regressors,par_trans_l,reg_pars_l,mask,common_mu=common_mu,common_sigma=common_sigma)
    return l.sum()

# def fit_one_neuron():
#     pass

@jit
def fit_population(target, regressors, par_trans_l, reg_pars_l, lr=0.1, niters=100, mask=None, opt_state=None,**kwargs):
    
    opt_init,opt_update,get_params=jax_opt.adam(lr)
    
    func = forward_and_loss_summed_across_neurons
    @jit
    def train_step(step_i,opt_state):
        params=get_params(opt_state)
        loss,grads = value_and_grad(func,argnums=2)(target,regressors,params,reg_pars_l,mask,**kwargs) # notice the params here
        
        return opt_update(step_i, grads, opt_state)
    if opt_state is None:
        opt_state=opt_init(par_trans_l)
    
    opt_state = jax.lax.fori_loop(0,niters,train_step,opt_state)

    return get_params(opt_state),  opt_state


#=====util helpers, might need to be moved out===========

def index_into_pytree(tree,ind):
    return jax.tree_map(lambda x:x[ind],tree)

def mask_fields(pars,nfields_mask,to_mask=['ws','sigmas','mus']):
    pars_ = {}
    nfields_mask = nfields_mask.astype(bool) # make sure it's bool!

    for k,val in pars.items():
        domask = False
        for tm in to_mask:
            if tm in k:
                domask=True
        if domask:    
            pars_[k] = pars[k][:,nfields_mask]
        else:
            pars_[k] = pars[k]
    return pars_