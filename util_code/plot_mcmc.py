import jax.numpy as np
import matplotlib.pyplot as plt
import math_functions as mf
import numpy
import copy
import gm_glm_bayesian as glm
import gm

def plot_one_trial_one_par(key,trial_ind,samples_filtered_sorted,fig=None,ax=None):
    if trial_ind is None:
        data = mf.softplus(samples_filtered_sorted[key])    
    else:
        data = mf.softplus(samples_filtered_sorted[key][:,:,trial_ind])
    if 'log' in key:
        data = mf.softplus(data)
    if ax is None:
        fig,ax=plt.subplots()
    
    for kk in range(data.shape[-1]):
        ax.hist(numpy.array(data[...,kk].flatten()),alpha=0.5,color=f'C{kk}')
        ax.axvline(data[...,kk].mean(),linestyle=':',color=f'C{kk}')
    return fig,ax
                       

# loss landscape
def loss_landscape_1d(par,key,ind,val_l,par_arg_index=1,loss_func=glm.negative_logpdf_no_reg,loss_args=[],loss_kwargs={'reg_pars':gm.get_reg_pars(),'reg_type':'gaussian_logprior_laplacian'}):
    par_copy = copy.copy(par)
    loss_l = []
    for val in val_l:
        par_copy[key]=par_copy[key].at[ind].set(val)
#         l = glm.negative_logpdf_no_reg(regressors,par_copy,target)
#         l = glm.negative_logpdf(regressors,par_copy,target,reg_pars=reg_pars,reg_type='gaussian_logprior_laplacian')
        loss_args[par_arg_index] = par_copy
        l = loss_func(*loss_args,**loss_kwargs)
        loss_l.append(l)
    return numpy.array(loss_l)