from re import M
from tracemalloc import Snapshot
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['image.cmap'] = 'Greys'

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
import sys,os,pickle,pdb,copy
sys.path.append('/mnt/home/szheng/projects/util_code')
import plot_helper as ph

from scipy.ndimage import gaussian_filter1d
import nmf_analysis as na

def plot_wh(W,H,fig=None,axs=None,scale = 1,factor_trial_ratio=4,factor_neuron_ratio=50,
            clim_quantile=0.9,trial_ticklabels=None,spacing=2):
    '''
    W: n_neurons x nfactors
    H: nfactors x n_trials
    '''
    N,K =W.shape
    K,T = H.shape

    def length_from_ratio(factor_unit_length,factor_trial_ratio,factor_neuron_ratio,N,T,K):
        trial_unit_length = factor_unit_length / factor_trial_ratio
        neuron_unit_length = factor_unit_length / factor_neuron_ratio

        neuron_length = neuron_unit_length * N
        trial_length = trial_unit_length * T
        factor_length = factor_unit_length * K

        return neuron_length, trial_length, factor_length

    if fig is None:
        factor_unit_length = 1
        neuron_length, trial_length, factor_length  = length_from_ratio(factor_unit_length,factor_trial_ratio,factor_neuron_ratio,N,T,K)
        
        while neuron_length < factor_length:
            factor_neuron_ratio = factor_neuron_ratio / 2
            neuron_length, trial_length, factor_length  = length_from_ratio(factor_unit_length,factor_trial_ratio,factor_neuron_ratio,N,T,K)

        # assert neuron_length > factor_length, "neuron length should be bigger than factor length to have W be a tall skinny matrix"

        p = factor_length/ neuron_length 
        height_ratios = [p, 1 - p] # make sure the height of H == factor length; total height is defined by neuron_length; with some math this can be shown
        width_ratios = [factor_length, trial_length]
        figsize = ((factor_length+trial_length) *scale , neuron_length * scale) # !!! width first then height!!!

        
        gs_kw = dict(width_ratios=width_ratios, height_ratios=height_ratios)
        fig,axd=plt.subplot_mosaic([['W','H'],['W','.']],figsize=figsize,gridspec_kw=gs_kw,constrained_layout=True)
        im=axd['W'].imshow(W,vmin=0,vmax=np.quantile(W,clim_quantile),aspect='auto')
        fig.colorbar(im,ax=axd['W'])
        axd['W'].set_xticks(np.arange(K))
        axd['W'].set_xlabel('component')
        axd['W'].set_ylabel('neuron')

        im=axd['H'].imshow(H,vmin=0,vmax=np.quantile(H,clim_quantile),aspect='auto')
        fig.colorbar(im,ax=axd['H'])
        axd['H'].set_yticks(np.arange(K))
        # tick_inds = np.arange(0,T,2)
        # axd['H'].set_xticks(tick_inds)
        # if trial_ticklabels is not None:
        #     axd['H'].set_xticklabels(trial_ticklabels[tick_inds])
        # axd['H'].set_xlabel('trial')
        axd['H']=set_trial_ticks(axd['H'],T,spacing=spacing,trial_ticklabels=trial_ticklabels)
        axd['H'].set_ylabel('component')
        return fig,axd

def set_trial_ticks(ax,T,spacing=2,trial_ticklabels=None):
    '''
    T: the number of trials
    trial_ticklabels: sometimes we want the actual trial index within the whole trial sequence

    '''
    tick_inds = np.arange(0,T,spacing)
    ax.set_xticks(tick_inds)
    if trial_ticklabels is not None:
        ax.set_xticklabels(trial_ticklabels[tick_inds])
    return ax

def plot_X_sorted(X_sorted,fig=None,ax=None,spacing=2,trial_ticklabels=None):
    '''
    X_sorted: nneurons x ntrials
    '''
    if ax is None:
        fig,ax=plt.subplots()
    im=ax.imshow(X_sorted,aspect='auto')
    T = X_sorted.shape[1]
    ax=set_trial_ticks(ax,T,spacing=spacing,trial_ticklabels=trial_ticklabels)
    fig.colorbar(im,ax=ax)
    return fig,ax

def plot_example_fr_across_trials(fr_to_be_nmfed,sample_inds,non_na_original_ind=None,pyr_uid=None,fig=None,axs=None):
    '''
    fr_to_be_nmfed: fr, ntrials x nneurons; processed for nmf, only including neuron columns, not normalized
    sample_inds : example indices to plot;
    non_na_original_ind: if not None, then the actual index within the pyr population is  non_na_original_ind[ind] (this showed up when some pyr are filtered out after the normalization for nmf)
    pyr_uid: turn index into uid
    '''
    if axs is None:
        fig,axs=ph.subplots_wrapper(len(sample_inds),return_axs=True)
    for ii,ind in enumerate(sample_inds):
        if non_na_original_ind is not None:
            original_ind = non_na_original_ind[ind]
        else:
            original_ind = ind
        toplot = fr_to_be_nmfed.iloc[:,original_ind]
        toplot_smth = gaussian_filter1d(toplot,sigma=1,mode='nearest')
        axs.ravel()[ii].plot(toplot.values)
        axs.ravel()[ii].plot(toplot_smth)
        if pyr_uid is not None:
            title = f'unit_{int(pyr_uid[original_ind])}'
        else:
            title = f'{original_ind}'
        axs.ravel()[ii].set_title(title)
    return fig,axs

def plot_example_rate_maps_across_trials(fr_map_trial,sample_inds,non_na_original_ind=None,pyr_uid=None):
    '''
    similar to plot_example_fr_across_trials
    fr_map_trial: n_neurons x n_position x n_trials
    '''
    fig,axs=ph.subplots_wrapper(len(sample_inds),return_axs=True)
    for ii,ind in enumerate(sample_inds):
        if non_na_original_ind is not None:
            original_ind = non_na_original_ind[ind]
        else:
            original_ind = ind
        toplot = fr_map_trial[original_ind].T
        axs.ravel()[ii].imshow(toplot,aspect='auto')
        if pyr_uid is not None:
            title = f'unit_{int(pyr_uid[original_ind])}'
        else:
            title = f'{original_ind}'
        axs.ravel()[ii].set_title(title)
    return fig,axs

def plot_ratemap_distribution_within_clusters(fr_map_avg,sample_inds_within_W_sorted_l,non_na_original_ind_joint,cell_cols_pyr,normalize=False,sort_by_com=False,fig=None,axs=None):
    '''
    plot heatmap of all trial-averaged rate maps belonging to a cluster, as well as the population average rate map within that cluster
    sort_by_com: if true, the neurons will be sorted by the com of their rate maps; otherwise use the order from the sorted W
    normalize: if true: normalize by the max of each neuron; 
    '''
    nplots = len(sample_inds_within_W_sorted_l) + 1
    if axs is None:
        fig,axs=ph.subplots_wrapper(nplots,return_axs=True,sharex=True)
    for ii in range(nplots):
        if ii == nplots-1:
            sample_inds_within_W_sorted = slice(0,None)
            title = 'all'
        else:
            sample_inds_within_W_sorted = sample_inds_within_W_sorted_l[ii]
            title = ii
        ax=axs.ravel()[ii]
        
        frmap_toplot = fr_map_avg.loc[cell_cols_pyr[non_na_original_ind_joint][sample_inds_within_W_sorted]]    
    
        if sort_by_com or (ii==(nplots-1)):
            inds=na.get_com(frmap_toplot,axis=1).sort_values().index
        else:
            inds=na.get_com(frmap_toplot,axis=1).index
        if normalize:
            frmap_toplot = frmap_toplot / np.max(frmap_toplot.values,axis=1,keepdims=True)
        ax.imshow(frmap_toplot.loc[inds],aspect='auto')

        ax.set_title(title)
        ax.set_xlabel('position')
        ax.set_ylabel('neurons')
        ax2=ax.twinx()
        pop_average = frmap_toplot.mean(axis=0)
        ax2.plot(pop_average)
    
    plt.tight_layout()
    return fig,axs

# def plot_example_W_and_ratemaps(W_df, inds_within_W_l, non_na_original_ind, fr_map_trial,X_normed_restacked_df,cell_cols_pyr,n_compo=4,fig=None,axs=None):
def plot_example_W_and_ratemaps(W_df, inds_within_W_l,X_normed_restacked_df,n_compo=4,fig=None,axs=None):

    '''
    plotting for nmf with neuron x position as the W rows and trials as H columns
    '''
    n_to_plot = len(inds_within_W_l)
    if axs is None:
        fig,nrows,ncols=ph.subplots_wrapper(n_to_plot)
        plt.close(fig)
        # fig,axs=  plt.subplots(n_to_plot,3,figsize=(8,4*n_to_plot))
        ncols = ncols*3
        fig,axs = plt.subplots(nrows,ncols,figsize = (3 * ncols, 4*nrows))
    counter = 0

    trial_ys = np.arange(X_normed_restacked_df.shape[1])
    for iii,inds_pair in enumerate(inds_within_W_l):
        ii = inds_pair[0] # neuron index
        axs.ravel()[counter].plot(W_df.loc[ii,0:n_compo-1])
        axs.ravel()[counter].legend(range(n_compo))
        # axs.ravel()[counter].set_title(cell_cols_pyr[non_na_original_ind[ii]])
        axs.ravel()[counter].set_title(ii) #now all df should have the unit names instead of a integer index
        # axs.ravel()[counter+1].imshow(fr_map_trial[non_na_original_ind[ii]].T,aspect='auto')
        axs.ravel()[counter+1].invert_yaxis()
        axs.ravel()[counter+1].plot(X_normed_restacked_df.loc[inds_pair],trial_ys)
        axs.ravel()[counter+1].set_title(f'pos bin {int(inds_pair[1])}')
        im=axs.ravel()[counter+2].imshow(X_normed_restacked_df.loc[ii].T,aspect='auto')
        plt.colorbar(im,ax=axs.ravel()[counter+2])
        # fig.suptitle(cell_cols_pyr[non_na_original_ind[ii]])
        counter = counter + 3
        plt.tight_layout()
    return fig,axs

