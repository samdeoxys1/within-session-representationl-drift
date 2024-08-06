import pygam
from pygam import PoissonGAM,s,l,te
import numpy as np
import scipy
import pandas as pd

import data_prep_pyn as dpp
import place_cell_analysis as pa
import place_field_analysis as pf
import pdb
import plot_helper as ph
import matplotlib
import matplotlib.pyplot as plt

def load_data(data_dir_full):
    prep_res = dpp.load_spk_beh_df(data_dir_full,force_reload=False,extra_load=dict(filtered='*thetaFiltered.lfp.mat'))
    spk_beh_df=prep_res['spk_beh_df']
    _,spk_beh_df = dpp.group_into_trialtype(spk_beh_df)
    # spk_beh_df,_=dpp.add_lin_binned(spk_beh_df,nbins=100)
    spk_beh_df,_=dpp.add_lin_binned(spk_beh_df,bin_size=2.2,nbins=None)
    cell_cols_d = prep_res['cell_cols_d']

    cell_cols = cell_cols_d['pyr']
    fr_map_dict=pa.get_fr_map_trial(spk_beh_df,cell_cols,trialtype_key='trial_type',speed_key='v',speed_thresh=1.,order=['smooth','divide'])
    fr_map_trial_d = {k:val[0] for k,val in fr_map_dict.items()}
    fr_map_trial_df_d=pd.concat({k:pf.fr_map_trial_to_df(fr_map_trial_d[k],cell_cols) for k in fr_map_dict.keys()},axis=0)

    trial_index_to_index_within_trialtype= dpp.trial_index_to_index_within_trialtype(spk_beh_df)
    load_res  = {}
    load_res['spk_beh_df'] = spk_beh_df
    load_res['cell_cols_d'] = cell_cols_d
    load_res['fr_map_trial_df_d']  =fr_map_trial_df_d
    load_res['trial_index_to_index_within_trialtype'] = trial_index_to_index_within_trialtype

    return load_res


def prep_data(load_res,task_ind=0,**kwargs):
    kwargs_ = {'bin_size_time':0.1}
    kwargs_.update(kwargs)
    bin_size_time = kwargs_['bin_size_time'] 


    spk_beh_df = load_res['spk_beh_df'] 
    cell_cols_d = load_res['cell_cols_d']
    
    spk_beh_df = spk_beh_df.query('task_index==@task_ind')
    dt = np.median(np.diff(spk_beh_df.index))
    n_to_bin = int(bin_size_time / dt)
    spk_beh_df_binned_spike = spk_beh_df[cell_cols_d['all']].rolling(n_to_bin).sum()[::n_to_bin]
    spk_beh_df_binned = spk_beh_df.iloc[::n_to_bin]
    spk_beh_df_binned.loc[:,cell_cols_d['all']]=spk_beh_df_binned_spike

    load_res['spk_beh_df_binned'] = spk_beh_df_binned
    prep_res = load_res
    return prep_res

def prep_regression(spk_beh_df_binned,y_key,x_keys,**kwargs):    
    kwargs_ = {}
    kwargs_.update(kwargs)

    y = spk_beh_df_binned[y_key]
    X = spk_beh_df_binned[x_keys]
    ma = y.notna() & X.notna().all(axis=1)
    y=y.loc[ma].values
    X=X.loc[ma].values
    return X, y

def post_fit_analysis():
    
    fit_res = {}

    return fit_res

def post_fit_plots(fit_res,fig=None,axs=None,**kwargs):
    kwargs_ = {}
    kwargs_.update(kwargs)

# def post_fit_plots_direct(gam,fig=None,axs=None,**kwargs):
#     kwargs_ = {}
#     kwargs_.update(kwargs)
#     nplots = len(gam.terms)
#     fig,axs = ph.subplots_wrapper(nplots)
#     for i, term in enumerate(gam.terms):
#         ax=axs.ravel()[i]
#         if term.isintercept:
#             continue
#         if term.istensor:
#             meshgrid=True
#             XX = gam.generate_X_grid(term=i,meshgrid=meshgrid)
#             pdep, confi = gam.partial_dependence(term=i, X=XX, meshgrid=meshgrid,width=0.95)    
#             dtype = np.array(term.dtype)
#             ma = dtype == 'categorical'
#             if ma.sum() > 0:
#                 pdep = pd.DataFrame(pdep,columns=)

#             ax.pcolormesh(XX[0],XX[1],pdep)
#         else:
#             meshgrid=False
#             XX = gam.generate_X_grid(term=i,meshgrid=meshgrid)
#             pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)    
#             ax.plot(XX[:, term.feature], pdep)
#             ax.plot(XX[:, term.feature], confi, c='r', ls='--')
        
#         ax.set_title(repr(term))
#     plt.show()

def plot_lin_dependence_by_trialtype(gam,lin,trialtype,fig=None,ax=None,n_grid=100):
    xx = np.linspace(lin.min(),lin.max(),n_grid)
    trialtype = np.unique(trialtype)
    if ax is None:
        fig,ax=plt.subplots()
    for ii,tt in enumerate(trialtype):
        xx_ = np.stack([xx,np.ones_like(xx)*tt],axis=1)
        pdep,confi = gam.partial_dependence(term=0,X=xx_,width=0.95)
        ax.plot(xx_[:,0],pdep,label=tt,c=f'C{ii}')
        ax.plot(xx_[:,0],confi,c=f'C{ii}',ls='--')
    
    return fig,ax
        








    

def main_analysis():
    pass




