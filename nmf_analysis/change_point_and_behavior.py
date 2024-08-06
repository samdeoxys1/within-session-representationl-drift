import ruptures as rpt
import numpy as np
import pandas as pd
import scipy
import os,sys,copy,itertools,pdb,importlib
import change_point_plot as cpp
importlib.reload(cpp)
import change_point_analysis as cpa
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
import sklearn
import statsmodels
import numpy.ma as ma
sys.path.append('../util_code/')
import misc
import data_prep_pyn as dpp
import place_field_analysis as pf
    


def regress_out_one_covar_from_fr(X,covar_df):
    '''
    X: nfields x ntrials
    covar_df: nfields x ntrials
    '''
    regressors = covar_df.T
    y = X.T
    # center 
    regressors_centered = regressors-regressors.mean(axis=0).values[None,:]
    y_centered = y-y.mean(axis=0).values[None,:]
    # mask nan
    regressors_centered_ma = ma.masked_array(regressors_centered, mask=np.isnan(regressors_centered))
    # do the least square
    col_wise_inner_product_nan_xx=np.ma.diag(np.ma.inner(regressors_centered_ma.T,regressors_centered_ma.T)).data
    col_wise_inner_product_nan_xy=np.ma.diag(np.ma.inner(regressors_centered_ma.T,y.T)).data
    w=1 / col_wise_inner_product_nan_xx * col_wise_inner_product_nan_xy
    # residual
    
    resid = y_centered - w * regressors_centered
    return w, resid.T

def renormalize(resid):
    '''
    resid: nfields x ntrials
    '''
    resid_norm = (resid -resid.min(axis=1).values[:,None]) / (resid.max(axis=1) - resid.min(axis=1)).values[:,None]
    return resid_norm
    

import trial_correlates as tc
def get_resid_onetrialtype(place_field_res,detection,trial_type,covar_by_trial_pos,iscommonfield=True,fr_key='fr_mean',covar_key='mean',do_renormalize=True):
    '''
    one trialtype, one field detection
    all_fields: if commonfield: (uid x field index) x [start end peak ...]; if not commonfield, (uid x field index x trial) x [start end peak...]
    covar_by_trial_pos: result of tc.get_speed_by_trial_pos, but .loc[trial_type]
    '''
    X = place_field_res[detection]['params'][trial_type].loc[fr_key]
    if iscommonfield:
        all_fields = place_field_res[detection]['all_fields'][trial_type]
        res_l = tc.get_covar_within_field_by_trial_commonfield(all_fields,covar_by_trial_pos)
    else:
        all_fields = place_field_res[detection]['params'][trial_type].loc[['start','end','peak']]
        all_fields = all_fields.stack().unstack(level=0)
        res_l = tc.get_covar_within_field_by_trial_seperatefield(all_fields,covar_by_trial_pos)
    
    w,resid=regress_out_one_covar_from_fr(X,res_l.loc[covar_key])
    if do_renormalize:
        resid = renormalize(resid)
    resid_res = {'resid':resid,'w':w,'covar_within_field':res_l,'X':X}
    return resid_res

def get_resid_alltrialtype(place_field_res,speed_by_trial_pos,fr_key='fr_mean',covar_key='mean',do_renormalize=True):
    resid_res_d = {}
    # for detection in place_field_res.keys():
    for detection in ['avg','trial_filter']:
        iscommonfield = False if 'trial' in detection else True
        resid_res_d[detection] = {}
        for trial_type in place_field_res[detection]['params'].keys():
            covar_by_trial_pos = speed_by_trial_pos.loc[trial_type]
            resid_res = get_resid_onetrialtype(place_field_res,detection,trial_type,covar_by_trial_pos,iscommonfield=iscommonfield,fr_key=fr_key,covar_key=covar_key,do_renormalize=do_renormalize)
            resid_res_d[detection][trial_type] = resid_res

    return resid_res_d

def turn_resid_res_into_place_field_res(resid_res_d,fr_key = 'fr_mean'):
    pfr_resid = {}
    for key, resid_res in resid_res_d.items():
        pfr_resid[key]={}
        pfr_resid[key]['params']={}
        pfr_resid[key]['params'] = {k:pd.concat({fr_key:resid_res[k]['resid']}) for k in resid_res.keys()}
    return pfr_resid

def regress_out_speed_get_residual_wrapper(data_dir_full,dosave=True,force_reload=False,load_only=False,
                                            speed_thresh=1,speed_key='v',
                                            save_fn='place_field_res_speed_residual.p',
                                            nbins=100,fr_key='fr_mean'
                                                ):
    
    save_dir =misc.get_or_create_subdir(data_dir_full,'py_data')
    save_fn, res = misc.get_res(save_dir,save_fn,force_reload)
    if res is not None:
        return res    
    
    prep_res = dpp.load_spk_beh_df(data_dir_full,force_reload=False,extra_load={})
    spk_beh_df=prep_res['spk_beh_df']
    _,spk_beh_df=dpp.group_into_trialtype(spk_beh_df)
    cell_cols_d = prep_res['cell_cols_d']                          
    cell_cols = cell_cols_d['pyr']
    spk_beh_df,lin_bins = dpp.add_lin_binned(spk_beh_df,nbins=nbins)

    place_field_res =pf.field_detection_both_avg_trial_wrapper(data_dir_full, dosave=True,force_reload=False,nbins = 100, 
                                        save_fn = 'place_field_avg_and_trial_vthresh.p', 
                                        shuffle_fn='fr_map_null_trialtype_vthresh.p',
                                        smth_in_bin=2.5, speed_thresh=1.,
                                        load_only=True,
                                        )
    if place_field_res is None:
        print('No place field res. Do that first!')
        return

    speed_by_trial_pos=tc.get_speed_by_trial_pos(spk_beh_df,trial_type_key='trial_type',speed_thresh=speed_thresh,speed_key=speed_key)
    resid_res_d = get_resid_alltrialtype(place_field_res,speed_by_trial_pos,fr_key='fr_mean',covar_key='mean',do_renormalize=True)
    pfr_resid = turn_resid_res_into_place_field_res(resid_res_d,fr_key = fr_key)
    res_to_save_data = pfr_resid
    misc.save_res(save_fn,res_to_save_data,dosave)
    return res_to_save_data


