'''
create the axis by which to organize cells/fields according to their type of variability
'''

import numpy as np
import scipy
import matplotlib.pyplot as plt
import sys,os,copy,pdb,importlib
from importlib import reload
import pandas as pd
sys.path.append('/mnt/home/szheng/projects/util_code')
sys.path.append('/mnt/home/szheng/projects/nmf_analysis')
import place_cell_analysis as pa
import tsp

from scipy.spatial.distance import pdist, squareform

def stability_by_corr_one_cell(mat):
    '''
    mat: npos x ntrial
    '''
    corr=mat.dropna(axis=1).corr().values
    corr_triu_flatten=corr[np.triu_indices(corr.shape[0],k=1)]
    return np.nanmedian(corr_triu_flatten)
    

# gpb=fr_map_trial_df_all.groupby(level=(0,1,2,3,4),sort=False)
# stab_all = gpb.apply(stability_by_corr_one_cell)


def get_per_neuron_metrics(spk_beh_df_all,fr_map_trial_df_all,fr_map_all=None,occu_map_all=None,stab_level=(0,1,2,3,4),si_level=(0,1)):
    per_tt_neuron_metrics = {}

    gpb=fr_map_trial_df_all.groupby(level=stab_level,sort=False)
    stab_all = gpb.apply(stability_by_corr_one_cell)
    per_tt_neuron_metrics['lap_correlation']=stab_all

    gpb = spk_beh_df_all.groupby(level=si_level,sort=False)
    s_i_all = {}
    for k,val in gpb:
        cell_cols = fr_map_trial_df_all.loc[k,:].index.get_level_values(2).unique()
        s_i = pa.get_bits_per_spike(val.loc[k],cell_cols=cell_cols,gauss_width = 2.5,speed_key = 'directed_locomotion',speed_thresh = 0.5,
                        trialtype_key='trial_type',
                        )
        s_i_all[k]=s_i
    s_i_all = pd.concat(s_i_all,axis=0)
    per_tt_neuron_metrics['si'] = s_i_all

    if (fr_map_all is not None) and (occu_map_all is not None):
        gpb = fr_map_all.groupby(level=(0,1,2,3),sort=False)
        spa_all = []
        for k,val in gpb:
            val = val.dropna(axis=1)
            spa = np.average(val**2 / (val.mean(axis=1)**2).values[:,None],weights=occu_map_all.loc[k],axis=1)
            spa=pd.Series(spa,index=val.index)
            spa_all.append(spa)
        spa_all = pd.concat(spa_all,axis=0)
        per_tt_neuron_metrics['sparsity'] = spa_all

    # per tt-neuron property
    per_tt_neuron_metrics=pd.concat(per_tt_neuron_metrics,axis=1)

    return per_tt_neuron_metrics

def get_per_field_var_metrics(X_raw_all,pf_params_recombined_all=None,all_fields_recombined_all=None,var_res_all=None,level=(0,1,2,3),pf_params_loc_ind=2,active_thresh=2.):

    '''
    var_res_all: from switch detection, variance decomposition, n_fields x [mean,tot_var, fit_var_ratio, etc...]
    '''
    per_field_var_metrics = {}
    frac_trial_active=X_raw_all.groupby(level=level,group_keys=False,sort=False).apply(lambda x:(x.dropna(axis=1)>active_thresh).mean(axis=1))
    per_field_var_metrics['frac_trial_active']=frac_trial_active
    fr_cv_beh = X_raw_all.std(axis=1)/X_raw_all.mean(axis=1)
    per_field_var_metrics['fr_cv_beh']=fr_cv_beh

    # # soft frac trial active
    # soft_frac_trial_active=np.exp(-X_raw_all*0.3).mean(axis=1)
    # per_field_var_metrics['soft_frac_trial_active']=soft_frac_trial_active


    if pf_params_recombined_all is not None:
        loc_key = 'peak'
        pf_params_ind = tuple([slice(None)]*pf_params_loc_ind+[loc_key])
        # loc_std = pf_params_recombined_all.loc[(slice(None),slice(None),loc_key),:].std(axis=1).droplevel(pf_params_loc_ind)
        loc_std = pf_params_recombined_all.loc[pf_params_ind,:].std(axis=1).droplevel(pf_params_loc_ind)
        per_field_var_metrics['loc_std'] = loc_std
    if all_fields_recombined_all is not None:
        field_width_all = all_fields_recombined_all['end']-all_fields_recombined_all['start']
        per_field_var_metrics['field_width'] = field_width_all

    per_field_var_metrics = pd.concat(per_field_var_metrics,axis=1)

    if var_res_all is not None:
        
        per_field_var_metrics = pd.concat([per_field_var_metrics,var_res_all],axis=1)
        per_field_var_metrics['log_mean'] = np.log(per_field_var_metrics['mean'])
    
    return per_field_var_metrics





def assign_per_neuron_property_to_field(per_field_var_metrics,per_tt_neuron_metrics):
    gpb = per_field_var_metrics.groupby(level=(0,1,2,3,4),sort=False)
    per_tt_neuron_metrics_expanded = {}
    # for k,val in gpb:
    for k,row in per_field_var_metrics.iterrows():
        ani,sess,ti,tt,uid,field_id=k
        if tt=='both':
            tt = slice(None)
            row=per_tt_neuron_metrics.loc[(ani,sess,ti,tt,uid)].mean(axis=0)
        else:
            row=per_tt_neuron_metrics.loc[(ani,sess,ti,tt,uid)]
        
        per_tt_neuron_metrics_expanded[k]=row
    per_tt_neuron_metrics_expanded=pd.concat(per_tt_neuron_metrics_expanded,axis=0).unstack()
    per_field_metrics_all = pd.concat([per_field_var_metrics,per_tt_neuron_metrics_expanded],axis=1)
    return per_field_metrics_all

def tsp_from_data(data):
    data_dist=squareform(pdist(data.values))
    tsp_inds,_=tsp.solve_tsp(data_dist)
    data_tsp = data.iloc[tsp_inds]
    return data_tsp,tsp_inds

