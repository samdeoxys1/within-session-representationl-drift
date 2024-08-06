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
import tqdm


# %%
# sess_name="e15_13f1_220117"
# py_data_dir = "/mnt/home/szheng/ceph/ad/roman_data/e15/e15_13f1/e15_13f1_220117/py_data"
# data_dir_full =str(Path(py_data_dir).parent)
# fr_map_ = pickle.load(open(os.path.join(py_data_dir,'fr_map.p'),'rb'))
# fr_map=fr_map_['fr_map']
# fr_map_trial=fr_map_['fr_map_trial']

# %% =======================Plotting====================================

def plot_fit_original(ys_l,pars_learned,nfields_mask,bin_to_lin=None,fig=None,ax=None,nooriginal=False,displacement=None):
    regressors={}
    npos,nt=ys_l.shape
    regressors['xs']=np.arange(npos)
    # ys_hat_l = gm.gm_func_by_trial(regressors,pars_learned)
    ys_hat_l = gm.forward_one_neuron(regressors,pars_learned,nfields_mask)
    nplots = nt
    # fig,axs = ph.subplots_wrapper(nplots,return_axs=True)
    # for ii,tr in enumerate(range(nt)):
    #     ax=axs.ravel()[ii]
    #     ax.plot(ys_l[:,tr],label='data')
    #     ax.plot(ys_hat_l[:,tr],label='fit')
    #     ax.set_title(f'{tr}')
    if not nooriginal:
        fig,ax,displacement=plot_fr_trial(ys_l,bin_to_lin=bin_to_lin,color='grey',linestyle='-',alpha=0.5,fig=fig,ax=ax,displacement=displacement)
    fig,ax,displacement=plot_fr_trial(ys_hat_l,bin_to_lin=bin_to_lin,color='grey',linestyle=':',fig=fig,ax=ax,displacement=displacement)


    return fig,ax,displacement


def plot_fr_trial(ys_l,bin_to_lin=None,color='grey',linestyle='-',alpha=1,fig=None,ax=None,displacement=20):

    if bin_to_lin is None:
        bin_to_lin = np.arange(ys_l.shape[0])
    if ax is None:
        fig,ax=plt.subplots()
    if displacement is None:
        displacement = ys_l.max(axis=0).mean() # adjust displacement according to the data;
    npos,nt=ys_l.shape
    # displacement = 20
    # spk_displacement = 2
    yticks = displacement * np.arange(nt)
    for ii,tr in enumerate(range(nt)):
        ax.plot(bin_to_lin,ys_l[:,tr]  + yticks[ii],color=color,linestyle=linestyle,alpha=alpha)
    # plt.show()
    ax.set_yticks(yticks)
    ax.set_yticklabels(np.arange(nt))
    return fig,ax, displacement

def convert_pars_from_bin_to_lin(pars,bin_to_lin):
    mus_in_bin=np.round(pars['mus']).astype(int)
    mus_in_lin = numpy.zeros_like(mus_in_bin)
    for i in range(mus_in_bin.shape[0]):
        for j in range(mus_in_bin.shape[1]):
            mus_in_lin[i,j]=bin_to_lin[mus_in_bin[i,j]]
    pars['mus'] = mus_in_lin
    pars['sigmas'] = numpy.mean(numpy.diff(bin_to_lin)) * pars['sigmas'] # stretch the sigma into unit of actual length
    return pars

from matplotlib.ticker import MaxNLocator
def plot_params(pars_,fig=None,ax=None,bin_to_lin=None,section_colordict_trial=None):
    # toplot=['ws','mus','sigmas','b','peaks']
    toplot=['ws','mus','sigmas','b']
    section_markers = np.array([0,74,111,185,222]) # currently hardcoded; need to make a function that produce this both for plot_raster and this
    nplots = len(toplot)
    pars = copy.copy(pars_)
    if bin_to_lin is not None:
        pars = convert_pars_from_bin_to_lin(pars,bin_to_lin)
        mus_mean = pars['mus'].mean(axis=0)
        color_assignment = assign_color_to_each_field(mus_mean,section_markers,section_colordict_trial)
    else:
        color_assignment = [{'color':f'C{k}'} for k in range(pars_['mus'].shape[1])]
    if ax is None:
        fig,ax=plt.subplots(1,nplots,figsize=(2*nplots,4),sharey=True)
    
    for ii,k in enumerate(toplot):
        val = pars[k]
        if k!='b':
            for kk in range(pars[k].shape[1]):
                ax[ii].plot(val[:,kk],np.arange(len(val[:,kk])),**color_assignment[kk])
        else:
            ax[ii].plot(val,np.arange(len(val)),color='k') # toplot on the x axis; trial ind on y
        ax[ii].set_title(k)
        ax[ii].yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    return fig,ax


#%% if you need to just check one neuron fit, need to uncomment the below

# args=pr.preprocess_for_plot(sess_name,sigma=30,speedmask=5,n_pos_bins=100)

# %%
# section_colordict,spk_triggered_positions_trial_all_speedmasked,pos_bins_dict,choice = args[4:8]
# ind = 0
# uid_l = fr_map[0].index
# uid = uid_l[ind] # 1 indexed
# uid_num = int(uid.split('_')[1])-1 # 0 indexed

# spk_triggered_positions_trial_one_unit = spk_triggered_positions_trial_all_speedmasked[uid_num]
# spk_triggered_positions_trial  = spk_triggered_positions_trial_one_unit


# ch=0
# ys_l = fr_map_trial[ch][ind]
# trial_mask = choice==ch
# spk_triggered_positions_trial_ = spk_triggered_positions_trial[trial_mask]

def plot_raster_1d_1trial(spk_triggered_positions_trial_,ch=0,fig=None,ax=None,spk_displacement_ratio=0.1,displacement=20,section_colordict=None,scatter_kwargs_={}):
    '''
    ch: choice; not so necessary; but if given can make the colors for the two choice types different
    '''
    if section_colordict is None:
        section_colordict = pr.make_section_colordict()
    if ax is None:
        fig,ax=plt.subplots()
    trial_spk_pair_l=[]
    for (trial,spk_trial) in enumerate(spk_triggered_positions_trial_):
        trial_col=np.ones(len(spk_trial),dtype=np.int32)*trial
        trial_spk_pair=np.concatenate([spk_trial[:,None],trial_col[:,None]],axis=1)
        trial_spk_pair_l.append(trial_spk_pair)
    trial_spk_pair_l=np.concatenate(trial_spk_pair_l,axis=0)
    # then get color
    section_markers = np.array([0,74,111,185,222])
    section_assignments=pd.cut(trial_spk_pair_l[:,0],section_markers,labels=False,include_lowest=True,right=True)
    c_l = [section_colordict[(ch,sec)] for sec in section_assignments]
    scatter_kwargs={'marker':'|','linewidth':0.7,'s':80}
    scatter_kwargs.update(scatter_kwargs_)
    xs=trial_spk_pair_l[:,0]
    
    spk_displacement = spk_displacement_ratio * displacement
    ys=trial_spk_pair_l[:,1]*displacement + spk_displacement
    ax.scatter(xs,ys,c=c_l,**scatter_kwargs)
    
    return fig,ax



# bin_to_lin=copy.copy(pos_bins_dict['lin'])
# bin_to_lin=np.concatenate([bin_to_lin[:-1,None],bin_to_lin[1:,None]],axis=1).mean(axis=1)


# fig,ax,displacement=plot_fr_trial(ys_l,bin_to_lin=bin_to_lin,color='grey',linestyle='-')






# %%

# count_map_l = fr_map_['count_map_trial'][ch][ind]
# occ_map_l = fr_map_['occupancy_map_trial'][ch]
# %%
# gm_fit_dir = '/mnt/home/szheng/ceph/ad/roman_data/e15/e15_13f1/e15_13f1_220117/py_data/gm_fit/nfields1_reg_quad_variation_mu_sigma_noS'

# ind = 0

# ch = 0

def plot_fit_original_raster_one_trial(spk_triggered_positions_trial_,ys_l,pars_learned,ch=0,fig=None,ax=None,nooriginal=False,displacement=None,bin_to_lin=None):
    '''
    plot the 1d raster with firing rate and fitted rate
    '''
    
    fig,ax,displacement=plot_fit_original(ys_l, pars_learned,bin_to_lin=bin_to_lin,fig=fig,ax=ax,nooriginal=nooriginal,displacement=displacement)
    fig,ax=plot_raster_1d_1trial(spk_triggered_positions_trial_,ch=ch,fig=fig,ax=ax,displacement=displacement)
    return fig,ax

def plot_fit_original_raster_both_trials(fr_map_trial,ind,choice,spk_triggered_positions_trial,gm_fit_dir=None,fig=None,axs=None,nooriginal=False,displacement=None,bin_to_lin=None):
    '''
    todo: directly feed in pars_learned for both trials
    ind: numerical index for fr_map_trial; not the same for spk_triggered_position, here it doesn't matter because spk_triggered_positions_trial is only for one neuron
    '''
    if axs is None:
        fig,axs = plt.subplots(len(fr_map_trial.keys()),1,sharex=True)
    if gm_fit_dir is None:
        print('directly feeding pars_learned is not implemented yet but will be')
        return 
    for ch in fr_map_trial.keys():

        fit_fn = os.path.join(gm_fit_dir,f'ch_{ch}_ind_{ind}.p')
        pars_learned = pickle.load(open(fit_fn,'rb'))['pars']

        ys_l = fr_map_trial[ch][ind]
        trial_mask = choice==ch
        spk_triggered_positions_trial_ = spk_triggered_positions_trial[trial_mask]
        fig,axs[ch]=plot_fit_original_raster_one_trial(spk_triggered_positions_trial_,ys_l,pars_learned,ch=ch,fig=fig,ax=axs[ch],nooriginal=nooriginal,displacement=displacement,bin_to_lin=bin_to_lin)        
    return fig,axs


# plot_fit_original_raster_one_trial(spk_triggered_positions_trial_,ys_l,pars_learned,fig=None,ax=None)
# fig,axs=plot_fit_original_raster_both_trials(fr_map_trial,ind,choice,spk_triggered_positions_trial,fig=None,axs=None,gm_fit_dir=gm_fit_dir)
# plt.show()



# %%
def plot_fit_original_raster_param_1trial(fr_map_trial,ind,choice,spk_triggered_positions_trial,fig=None,axs=None,pars_learned=None,ch=0,gm_fit_dir=None,nooriginal=False,figsize=(23,5),bin_to_lin=None,nparams=5,displacement=None,section_colordict_trial=None):
    ys_l  = fr_map_trial[ch][ind]
    if axs is None:
        fig,axs=plt.subplots(1,nparams+1,figsize=figsize,gridspec_kw={'width_ratios':[2.5]+[1]*nparams})
    plot_fit_original_raster_one_trial(spk_triggered_positions_trial,ys_l,pars_learned,ch=0,fig=fig,ax=axs[0],nooriginal=nooriginal,displacement=displacement,bin_to_lin=bin_to_lin)
    if pars_learned is None:
        fit_fn = os.path.join(gm_fit_dir,f'ch_{ch}_ind_{ind}.p')
        pars_learned = pickle.load(open(fit_fn,'rb'))['pars']
    fig,axs[1:]=plot_params(pars_learned,fig=fig,ax=axs[1:],bin_to_lin=bin_to_lin,section_colordict_trial=section_colordict_trial)
    plt.tight_layout()
    return fig,axs

def plot_fit_original_raster_param_both_trials(fr_map_trial,ind,choice,spk_triggered_positions_trial,fig=None,axs=None,pars_learned=None,gm_fit_dir=None,nooriginal=False,figsize=(23,10),bin_to_lin=None,nparams=5,section_colordict=None):
    '''
    remember to update nparams if that changes
    '''
    if axs is None:
        fig,axs=plt.subplots(2,nparams+1,figsize=figsize,gridspec_kw={'height_ratios':[1,1],'width_ratios':[2.5]+[1]*nparams})
    fig,axs[:,0]=plot_fit_original_raster_both_trials(fr_map_trial,ind,choice,spk_triggered_positions_trial,fig=fig,axs=axs[:,0],gm_fit_dir=gm_fit_dir,nooriginal=nooriginal,displacement=None,bin_to_lin=bin_to_lin)
    # plot params both trials
    for row,ch in enumerate(fr_map_trial.keys()):
        fit_fn = os.path.join(gm_fit_dir,f'ch_{ch}_ind_{ind}.p')
        pars_learned = pickle.load(open(fit_fn,'rb'))['pars']
        if section_colordict is not None:
            section_colordict_trial = {k[-1]:section_colordict[k] for k in section_colordict.keys() if k[0]==ch}
        else:
            section_colordict_trial=None
        fig,axs[row,1:]=plot_params(pars_learned,fig=fig,ax=axs[row,1:],section_colordict_trial=section_colordict_trial)
    plt.tight_layout()
    return fig,axs

# fig,axs=plot_fit_original_raster_param_both_trials(fr_map_trial,ind,choice,spk_triggered_positions_trial,fig=None,axs=None,gm_fit_dir=gm_fit_dir,nooriginal=True,bin_to_lin=bin_to_lin)
# fig
# plt.show()

# %%
def plot_fit_result_all_neurons(sess_name,data_dir_full=None,gm_fit_dir=None,nooriginal=False,savefig=False,force_reload=False):
    # preprocess dirs
    if data_dir_full is None:
        data_dir_full = db.query(f'sess_name=="{sess_name}"').loc[:,'data_dir_full'].iloc[0]
    pydata_folder = os.path.join(data_dir_full, 'py_data')
    sess_dir_full = data_dir_full
    if gm_fit_dir is None:
        gm_fit_dir = 'gm_fit/nfields1_reg_quad_variation_mu_sigma_noS'
    gm_fit_dir_full = os.path.join(sess_dir_full,'py_data',gm_fit_dir)
    fig_save_dir_full = os.path.join(sess_dir_full,'py_figures',gm_fit_dir) # use the same subtrees
    
    if not os.path.exists(fig_save_dir_full):
        os.makedirs(fig_save_dir_full)
        print(f'{fig_save_dir_full} made!')
    
    # load fr_map_trial
    fr_map_ = pickle.load(open(os.path.join(pydata_folder,'fr_map.p'),'rb'))
    fr_map = fr_map_['fr_map']
    fr_map_trial=fr_map_['fr_map_trial']

    # get/load spike_triggered positions
    args=pr.preprocess_for_plot(sess_name,sigma=30,speedmask=5,n_pos_bins=100)
    spk_triggered_positions_trial_all_speedmasked,pos_bins_dict,choice = args[5:8]
    bin_to_lin=copy.copy(pos_bins_dict['lin'])
    bin_to_lin=np.concatenate([bin_to_lin[:-1,None],bin_to_lin[1:,None]],axis=1).mean(axis=1)

    # fr_map: indexed by uid, only contains pyr; spk_triggered_positions_trial_all_speedmasked contains all units
    for ind,uid in tqdm.tqdm(enumerate(fr_map[0].index)):
        uid_num  = int(uid.split('_')[1])-1 # converted to 0 indexed
        fn_full = os.path.join(fig_save_dir_full,f'{uid}.pdf')
        if os.path.exists(fn_full) and (not force_reload):
            continue
        spk_triggered_positions_trial_one_unit = spk_triggered_positions_trial_all_speedmasked[uid_num]
        spk_triggered_positions_trial  = spk_triggered_positions_trial_one_unit
        try:
            fig,axs=plot_fit_original_raster_param_both_trials(fr_map_trial,ind,choice,spk_triggered_positions_trial,fig=None,axs=None,gm_fit_dir=gm_fit_dir_full,nooriginal=nooriginal,bin_to_lin=bin_to_lin)
            fig.suptitle(uid,fontsize=20)
            if savefig:
                
                fig.savefig(fn_full,bbox_inches='tight')
                print(f'{fn_full} saved!')
                plt.close(fig)
        except Exception as e:
            print(e)

    print('Done!')

# plot_fit_result_all_neurons(sess_name,data_dir_full=data_dir_full,gm_fit_dir=None,nooriginal=False,savefig=True)


# %%
def assign_color_to_each_field(mus_mean,section_markers,section_colordict_trial):
    '''
    mus_mean: 1d array, mean place field centers across trials
    # section_markers = np.array([0,74,111,185,222]) # currently hardcoded; need to make a function that produce this both for plot_raster and this
    section_colordict_trial:{index:(a,b,c)...} different from section_colordict: no trial in the key
    '''
    section_assignment = pd.cut(mus_mean,section_markers,labels=False)
    section_assignment = pd.DataFrame({'sec':section_assignment})
    K_fields=len(mus_mean)
    color_grad_scalar = numpy.linspace(1,0.3,K_fields)
    # color_assignment = section_colordict_trial[]
    gpb=section_assignment.groupby('sec')
    color_assignment = {}

    iii = 0
    for k, val in gpb:
        for ii, vv in enumerate(val.values):
            color_assignment[iii] = {'color':section_colordict_trial[k],'alpha':color_grad_scalar[ii]}
            iii+=1
    return color_assignment

