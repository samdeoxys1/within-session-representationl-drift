import numpy
import jax.numpy as np
import matplotlib.pyplot
from jax import vmap
from jax.tree_util import tree_map
import pandas as pd
import scipy
import scipy.stats
from math_functions import *
import gm as gm
import plot_helper as ph
import gm_glm_bayesian as glm
from importlib import reload
reload(glm)

def duplicate_dict(d,N=(10,1),to_exclude=['dt'],trial_len=None):
    dd = {}
    for k,v in d.items():
        if k not in to_exclude:
            dd[k] = np.squeeze(np.tile(v,N))
        else:
            dd[k]=v
    if trial_len is not None: # the case of NOT TRIAL AVERAGING!
        dd['trial_inds_int'] = np.repeat(np.arange(N),trial_len)

    return dd

def plot_fitted_dist(pars_fit_,true_pars=None):
    '''
    only works for one trial
    '''
    nsamples = pars_fit_['mus'].shape[0]
    mean_dict = {}
    nplots = len(pars_fit_)
    fig,axs=ph.subplots_wrapper(nplots,return_axs=True)
    if true_pars is not None:
        pars_fit = tree_map(lambda x,y:x-y,pars_fit_,true_pars)
    else:
        pars_fit=pars_fit_
        
    for ii,(k,v) in enumerate(pars_fit.items()):
        eff_v = numpy.array(v[nsamples//2:])
        mean_dict[k] = eff_v.mean(axis=0)
        for jj in range(eff_v.shape[-1]):

            axs.ravel()[ii].hist(numpy.squeeze(eff_v[...,jj]),label=jj,histtype='step',density=True)
            axs.ravel()[ii].set_title(k)
            axs.ravel()[ii].axvline(mean_dict[k][...,jj],label=jj,linestyle=':',color=f'C{jj}')
            
        
    return mean_dict,fig,axs
        
# sort fitted
def sort_pars_fit(pars_fit_gd_trial):
    sorted_field_inds = np.argsort(pars_fit_gd_trial['mus'],axis=-1)
    pars_fit_gd_trial_sorted={}
    for k in ['mus','logsigmas','logws']:
        pars_fit_gd_trial_sorted[k] = vmap(lambda x,ind:x[ind],in_axes=(0,0))(pars_fit_gd_trial[k],sorted_field_inds)
    pars_fit_gd_trial_sorted['logb']=pars_fit_gd_trial['logb']
    return pars_fit_gd_trial_sorted

def sort_pars_fit_multichain(pars_samples_multichain):
    '''
    vmap sort_pars_fit to samples; not sure if this is correct though, since we might want the samples at one field to remain for the same field?
    '''
    chain_func = vmap(sort_pars_fit,in_axes=(0,))
    all_res = vmap(chain_func,in_axes=(0,))(pars_samples_multichain)

    return all_res


def get_df(regressors,nbins=220,**kwargs):
    df = pd.DataFrame(regressors,columns=['trial_inds_int','position'])
    posbin = pd.cut(df['position'],bins=nbins,retbins=True,labels=False)[0]
    df['position_bin']=posbin
    if kwargs is not None:
        for k,v in kwargs.items():
            df[k] = v
    return df

def getcom(df):
    gpb=df.groupby('position_bin')
    map_hat=gpb['target_spk'].mean()
    muhat=numpy.sum(map_hat*map_hat.index)/map_hat.sum()
    return muhat


# L = get_modified_laplacian_matrix(ntrials) 
# Lcov = np.linalg.inv(L)

# helper for simulating drifting parameters
def add_noise_par(par,Lcov,reg_pars,log_names=['sigma','b','w'],nolog_names=['mu']):
    pars_sim_np = {}
    for k,v in par.items():
        pars_sim_np[k]=numpy.array(v)
    par_copy = {}
    ntrials = Lcov.shape[0]
    for n in log_names:
        if n!='b':
            name_in_par = f'log{n}s'
            nfields = pars_sim_np[name_in_par].shape[1]
            par_copy[name_in_par] = numpy.zeros((ntrials,nfields))
            for k in range(nfields):
                noise = scipy.stats.multivariate_normal.rvs(cov=Lcov,size=1) * numpy.sqrt(reg_pars[f'g_{n}'])
                par_copy[name_in_par][:,k] = pars_sim_np[name_in_par][:,k] + noise 
        else:
            name_in_par = 'logb'
            par_copy[name_in_par] = numpy.zeros(ntrials)
            noise = scipy.stats.multivariate_normal.rvs(cov=Lcov,size=1) * numpy.sqrt(reg_pars[f'g_{n}'])
            par_copy[name_in_par] = pars_sim_np[name_in_par] + noise 
    for n in nolog_names:
        name_in_par = f'{n}s'
        nfields= pars_sim_np[name_in_par].shape[1]
        par_copy[name_in_par] = numpy.zeros((ntrials,nfields))
        for k in range(nfields):
            noise = scipy.stats.multivariate_normal.rvs(cov=Lcov,size=1) * numpy.sqrt(reg_pars[f'g_{n}'])
            par_copy[name_in_par][:,k] = pars_sim_np[name_in_par][:,k] + noise 
    return par_copy
                
def index_into_pytree(tree,ind):
    return tree_map(lambda x:x[ind],tree)

# def reorder(pars):
#     '''
#     reorder parameters according to ascending mus
#     '''
#     pars_copy = {}
#     mus_bar = pars['mus'].mean(axis=0)
#     sorted_inds = np.argsort(mus_bar)
#     for k,v in pars.items():
#         if 'b' not in k:
#             pars_copy[k] = v[:,sorted_inds]
#         else:
#             pars_copy[k]= v
#     return pars_copy

def get_one_sample(ind,samples,chain_ind=None):
    if chain_ind is None:
        return tree_map(lambda x:x[ind], samples)
    else:
        return tree_map(lambda x:x[ind,chain_ind], samples)



def get_prediction_one(par_one_sample,regressors,nbins=220):
    ys_l_hat=glm.gm_func_by_trial(gm.get_regressor({'xs':np.arange(nbins),'trial_type_inds':regressors['trial_type_inds']}),par_one_sample)
    return ys_l_hat

def get_pred_distribution_multichain(par_multichain,nbins=220):
    '''
    for firing map
    '''
    func = lambda x: get_prediction_one(x,nbins=nbins)
    sample_func = vmap(func,in_axes=(0,))
    yhat_distribution=vmap(sample_func,in_axes=(0,))(par_multichain)
    return yhat_distribution

def get_pred_distribution_multichain_intime(regressors, par_multichain):
    '''
    for rate in time
    '''
    func = lambda par:glm.forward(regressors,par)
    sample_func = vmap(func,in_axes=(0,))
    yhat_distribution = vmap(sample_func,in_axes=(0,))(par_multichain)
    return yhat_distribution
    


def get_logpdf_multichain(regressors,par_multichain,target,mask=None,noise_type="poisson"):
    func = vmap(lambda par:glm.logpdf_no_reg(regressors,par,target,mask=mask,noise_type=noise_type),in_axes=(0,))
    logpdf_dist = vmap(func,in_axes=(0,))(par_multichain)
    return logpdf_dist
    

def get_pred_mean_and_ci(yhat_dist,mean_axis=[0,1],ci_sc=1.96):
    yhat_mean = yhat_dist.mean(axis=mean_axis)
    nsamples = np.prod(numpy.array(yhat_dist.shape)[mean_axis])
    yhat_ci = ci_sc * yhat_dist.std(axis=mean_axis) / np.sqrt(nsamples)

    return yhat_mean, yhat_mean+yhat_ci, yhat_mean-yhat_ci


def split_par_trial_type(pars,regressors,type_keys = ['mus','sigmas','logsigmas','logws_bar','logb_bar'], trial_keys=['delta_ws','delta_b','ws','b']):
    pars_dict = {}
    for k,inds in regressors['trial_type_inds'].items():
        pars_dict[k] = {}
        for kk in pars.keys():
            if kk in type_keys:
                pars_dict[k][kk] = pars[kk][int(k)][None,:]
            elif kk in trial_keys:
                pars_dict[k][kk] = pars[kk][inds]
    return pars_dict

def mask_fields(pars,nfields_mask,to_mask=['w','sigma','mu']):
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
            