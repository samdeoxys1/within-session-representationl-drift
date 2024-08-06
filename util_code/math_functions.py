import numpy 
import scipy
import jax
import jax.numpy as np
import jax.scipy as scipy
from jax import jit
import functools


@jit
def unnormalized_normal_pdf(x,loc,scale):
    return np.exp(-(x-loc)**2 / (2* scale **2))

def get_modified_laplacian_matrix(N, inv=False):
    L = numpy.eye(N) * 2
    L -= numpy.diag(numpy.ones(N-1),k=1)
    L -= numpy.diag(numpy.ones(N-1),k=-1)
    if inv:
        L = np.linalg.inv(L)
    return L
@jit
def force_sigma_big_prior(g_sigma_thresh,sigma_thresh,sigmas,nfields_mask):
    '''
    used in prior, thus the negative sign
    '''
    # -reg_pars['g_sigma_thresh']*np.mean(jax.nn.relu(reg_pars['sigma_thresh'] - pars['sigmas'])) # if sigma too small, penalized; otherwise no penalty
    return -g_sigma_thresh * np.sum(jax.nn.relu(sigma_thresh-sigmas) * nfields_mask) / np.sum(nfields_mask)
@jit
def order_prior(g_order,mus,nfields_mask):
    '''
    # enforcing lower index corresponding to field with lower location
    mus: ntrialtypes x nmaxfields
    mask: nmaxfields,
    '''
    diff = np.diff(mus,axis=1,prepend=0.)
    # masked_diff = np.dot(diff,nfields_mask)
    nonzeroparts = -jax.nn.relu(-diff)
    R_order = np.mean(g_order * np.dot(nonzeroparts,nfields_mask) / np.sum(nfields_mask) )
    return R_order
@jit
def sigma_shrinkage_prior(g_sigma_shrinkage,sigmas,nfields_mask):
    # make sure sigma don't grow too large; only want a weak penalty
    mean_across_valid = np.dot(sigmas **2, nfields_mask) / np.sum(nfields_mask)
    mean_across_dim0 = np.mean(mean_across_valid)
    R_sigma_shrinkage = -g_sigma_shrinkage * mean_across_dim0 
    return R_sigma_shrinkage

@jit
def gaussian_logprior_laplacian(x,x_bar,nfields_mask):
    '''
    x:ntrials x nfields
    x_bar: 1 x nfields
    the precision is close to a laplacian, except for the top and bottom diagonal are 2 instead of 1
    '''
    # quad_var_ = np.sum(np.diff(x,axis=0)**2)
    quad_var_ = np.mean(np.diff(x,axis=0)**2,axis=0)
    quad_var_masked = np.dot(quad_var_,nfields_mask) / np.sum(nfields_mask)
    beg_end = np.dot(((x[0]-x_bar)**2 + (x[-1]-x_bar)**2) /2, nfields_mask) / np.sum(nfields_mask)
    # quad_var_ = np.mean(np.diff(x,axis=0)**2)
    
    return - 1/2*(quad_var_masked + np.mean(beg_end))
    
    # return - (1/2*(quad_var_ + np.mean((x[0]-x_bar)**2) + np.mean()) )

@jit
def quadratic_variation(x):
    return np.mean(np.diff(x,axis=0)**2)

@jit
def mse(a,b):
    return np.sum(1/2*(a-b)**2)

@jit
def mse_no_reduce(a,b):
    return (1/2 * (a-b)**2)

@jit
def softplus(x):
    # return np.log(1+np.exp(x))

    return np.logaddexp(x,0.)
@jit
def inv_softplus(y):
    # threshold = np.log(eps) + 2.
    # is_too_small = x < np.exp(threshold)
    # is_too_large = x > -threshold
    # too_small_value = tf.math.log(x)
    # too_large_value = x
    # This `where` will ultimately be a NOP because we won't select this
    # codepath whenever we used the surrogate `ones_like`.
    # x = tf.where(is_too_small | is_too_large, tf.ones([], x.dtype), x)
    # y = x + tf.math.log(-tf.math.expm1(-x))  # == log(expm1(x))
    eps = 1e-8
    x = np.log(1-np.exp(-y)+eps) + y # a stable version of inv softplus
    return x

# @functools.partial(jit,static_argnums=1)
def smooth(x,win):
    window = np.ones(win) / win
    x_smth = np.convolve(x,window,mode='same')
    return x_smth