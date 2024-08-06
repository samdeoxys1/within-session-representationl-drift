'''
get relevant variables (behavioral metric, lfp, e) for each trial
'''
import sys,copy,pdb,importlib,os,itertools,pickle
import numpy as np
import pandas as pd
import scipy
import data_prep_pyn as dpp
sys.path.append('/mnt/home/szheng/projects/seq_detection2/code')
import preprocess as prep
import misc


def get_variable_statistics_per_trial(beh_df,key='speed',agg_dict={}):
    '''
    applicable for both speed related and theta!
    '''
    agg_dict_ = {f'{key}_std':np.std,f'{key}_mean':np.mean,f'{key}_cv':lambda x:x.std()/x.mean()}
    agg_dict_.update(agg_dict)
    if 'trial_type' not in beh_df.columns:
        _,beh_df=dpp.group_into_trialtype(beh_df)

    stats= beh_df.groupby(['trial_type','trial'])[key].agg(list(agg_dict_.values()))
    stats.columns = agg_dict_.keys()
    return stats

def get_time_related_per_trial(beh_df,speed_thresh=1,speed_key='v'):
    '''
    get per trial: trial duration, idle time, idle percentage
    '''
    if 'trial_type' not in beh_df.columns:
        _,beh_df=dpp.group_into_trialtype(beh_df)
    dt_df= beh_df.groupby(['trial_type','trial']).apply(lambda x:np.median(np.diff(x.index))) # dts are different!!
    beh_df['idle_in_trial'] = beh_df[speed_key] < speed_thresh
    beh_gpb = beh_df.groupby(['trial_type','trial'])
    trial_duration= beh_gpb['lin'].count() * dt_df
    idle_time = beh_gpb.apply(lambda x:x['idle_in_trial'].sum()) * dt_df
    idle_time_ratio = beh_gpb.apply(lambda x:x['idle_in_trial'].mean())
    time_related = pd.concat([trial_duration,idle_time,idle_time_ratio],axis=1)
    time_related.columns=['trial_dur','idle_dur','idle_ratio']
    return time_related

def get_firing_related_per_trial(spk_beh_df,cell_cols_d,speed_thresh=1,speed_key='v'):
    '''
    get average firing rate, fraction of activate cell, and the e-i ratio for these
    [TO DO]: a version with only spikes during running 
    '''
    if 'trial_type' not in spk_beh_df.columns:
        _,spk_beh_df=dpp.group_into_trialtype(spk_beh_df)
    spk_beh_df = spk_beh_df.loc[spk_beh_df[speed_key]>speed_thresh]
    spk_beh_gpb = spk_beh_df.groupby(['trial_type','trial'])
    res={}
    # for key,cols in cell_cols_d.items():
    for key in ['pyr','int']: # all is not important
        cols = cell_cols_d[key]
        res[f'mean_fr_{key}']=spk_beh_gpb[cols].mean().mean(axis=1) # spike/fr averaged across bins and neurons
        res[f'frac_active_{key}']=(spk_beh_gpb[cols].sum() > 0).mean(axis=1) # fraction of active cell
    res['mean_fr_ratio']=res['mean_fr_pyr']/res['mean_fr_int'] # fr e-i ratio
    res[f'frac_active_ratio']=res[f'frac_active_pyr'] / res[f'frac_active_int'] # fraction active e-i ratio
    firing_related= pd.DataFrame(res)
    return firing_related

def get_reward_related_per_trial(beh_df):
    if 'trial_type' not in beh_df.columns:
        _,beh_df=dpp.group_into_trialtype(beh_df)
    sub_beh_df=beh_df.query('task=="alternation"')
    trial_type_per_trial = sub_beh_df.groupby('trial')['trial_type'].apply(lambda x:x.unique()[0])
    correct_per_trial = sub_beh_df.groupby('trial')['correct'].apply(lambda x:x.unique()[0])
    correct_prev_trial = correct_per_trial.shift(1)
    
    # reward_related = pd.DataFrame([trial_type_per_trial,correct_per_trial,correct_prev_trial],columns=['trial_type','correct','correct_prev'])
    reward_related = pd.concat([trial_type_per_trial,correct_per_trial,correct_prev_trial],axis=1)
    reward_related.columns=['trial_type','correct','correct_prev']
    reward_related_df=[]
    for k, val in reward_related.groupby('trial_type'):
        val.index=pd.MultiIndex.from_product([[k],val.index])
        reward_related_df.append(val)
    
    reward_related_df=pd.concat(reward_related_df,axis=0)
    reward_related_df=reward_related_df.drop('trial_type',axis=1)
    return reward_related_df
    
def get_ripples_related_per_trial(ripples,beh_df,speed_thresh=1,speed_key='speed'):
    ripple_times=pd.DataFrame(ripples['timestamps'],columns=['start','end'])
    if 'trial_type' not in beh_df.columns:
        _,beh_df=dpp.group_into_trialtype(beh_df)
    
    gpb=beh_df.groupby('trial_type')
    ripples_related={}
    for key, val in gpb:
        ripples_related[key]={}
        trial_intervals=val.groupby('trial').apply(lambda x:(x.index[0],x.index[-1])) # get trial begin end series for one trialtype
        trial_intervals_index =pd.IntervalIndex.from_tuples(trial_intervals)
        trial_assignment_ripples=pd.cut(ripple_times['start'],trial_intervals_index)
        trial_assignment_ripples=trial_assignment_ripples.cat.rename_categories(trial_intervals.index)
        ripple_count = pd.value_counts(trial_assignment_ripples)
        n_idle_bin_in_trial=val.groupby('trial').apply(lambda x:(x[speed_key]<speed_thresh).sum())
        
        dt =np.median(np.diff(val.index))
        idle_duration = dt * n_idle_bin_in_trial
        ripple_rate = ripple_count / idle_duration 
        ripples_related[key] = pd.concat([ripple_count,ripple_rate],axis=1)
        ripples_related[key].columns = ['ripple_count','ripple_rate']
        ripples_related[key] = ripples_related[key].sort_index()
        
    ripples_related = pd.concat(ripples_related)
    return ripples_related


def get_trial_correlates(spk_beh_df,mat_to_return,cell_cols_d,cols=None,speed_key='v',speed_thresh=1):
    '''
    trial_correlates: df: (trialtype x ntrials) x cols
    '''
    if cols is None:
        cols = ['v_mean','v_cv','idle_ratio','mean_fr_pyr','frac_active_pyr','correct','correct_prev']
    
    speed_stats = get_variable_statistics_per_trial(spk_beh_df,key=speed_key)
    time_related=get_time_related_per_trial(spk_beh_df,speed_key=speed_key)
    firing_related=get_firing_related_per_trial(spk_beh_df,cell_cols_d,speed_key=speed_key,speed_thresh=speed_thresh)
    reward_related=get_reward_related_per_trial(spk_beh_df)
    trial_correlates = pd.concat([speed_stats,time_related,firing_related,reward_related],axis=1)
    try:
        ripples=mat_to_return['ripples']
        ripples_related=get_ripples_related_per_trial(ripples,spk_beh_df)
        has_ripples=True
        ripple_cols = ['ripple_count','ripple_rate']
        cols.extend(ripple_cols)
        for rc in ripple_cols:
            trial_correlates[rc] = ripples_related[rc].values
        # trial_correlates = pd.concat([trial_correlates,ripples_related.reset_index(drop=True)],axis=1)
    except:
        has_ripples = False
        pass
    trial_correlates = trial_correlates[cols]
    return trial_correlates

def get_trial_correlates_wrapper(data_dir_full,dosave=True,save_fn='trial_correlates.p',save_dir='trial_correlates',force_reload=False,**kwargs):
    save_dir = misc.get_or_create_subdir(data_dir_full,'py_data',save_dir)
    fn_full,res=misc.get_res(save_dir,save_fn,force_reload)
    if res is not None:
        return res
    mat_to_return=prep.load_stuff(data_dir_full)
    
    prep_res = dpp.load_spk_beh_df(data_dir_full,force_reload=False,extra_load={})
    spk_beh_df=prep_res['spk_beh_df']
    _,spk_beh_df=dpp.group_into_trialtype(spk_beh_df)
    cell_cols_d = prep_res['cell_cols_d']
    trial_cor = get_trial_correlates(spk_beh_df,mat_to_return,cell_cols_d,cols=None,**kwargs)
    misc.save_res(fn_full,trial_cor,dosave=dosave)
    return trial_cor



import scipy
import scipy.cluster.hierarchy as sch

def cluster_corr(corr_array, inplace=False):
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly 
    correlated variables are next to eachother 
    
    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix 
        
    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max()/2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, 
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)
    
    if not inplace:
        corr_array = corr_array.copy()
    
    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :], idx
    return corr_array[idx, :][:, idx], idx


    
def get_speed_by_trial_pos(spk_beh_df,trial_type_key='trial_type',speed_thresh=1,speed_key='v'):
    '''

    res: series: (task x turntype x trial x pos)
    '''
    assert 'lin_binned' in spk_beh_df.columns
    spk_beh_df = spk_beh_df.loc[spk_beh_df[speed_key] > speed_thresh]
    gpb = spk_beh_df.groupby(trial_type_key)
    res_d={}
    for key,val in gpb:
        gpbtrial=val.groupby(['trial'])
        across_trial = {}
        for tt,(kk,valval) in enumerate(gpbtrial):
            across_trial[tt]=valval.groupby('lin_binned')[speed_key].mean()
        res = pd.concat(across_trial,axis=0)
        res = res.unstack().interpolate(limit_direction='both',axis=1).stack()  # fill in the nan; due to truncation, or perhaps camera sample issue?  
        res_d[key] = res
        # res=val.groupby(['trial','lin_binned'])[speed_key].mean() # to index within trialtype
        # res_d[key] = res
    res_d = pd.concat(res_d,axis=0)
    return res_d

def get_covar_within_field_by_trial_commonfield(all_fields,covar_by_trial_pos):
    '''
    all_fields, 
    covar_by_trial_pos: see output of get_speed_by_trial_pos; subselected one trialtype
    '''
    mean_l = {}
    peak_l = {}
    res_l = {}
    gpb=covar_by_trial_pos.groupby(level=0)
    for (uid,field_ind),row in all_fields.iterrows():

        mean_trial=gpb.apply(lambda x:x.loc[slice(None),row['start']:row['end']].mean())
        mean_l[uid,field_ind]=mean_trial
        peak_trial = gpb.apply(lambda x:x.loc[slice(None),row['peak']].iloc[0] if row['peak'] in x.index.get_level_values(1) else np.nan) # perhaps a trial might not have covar value at that location (due to speed threshold eg)
        peak_l[uid,field_ind] = peak_trial
    mean_l = pd.concat(mean_l,axis=1).T
    peak_l = pd.concat(peak_l,axis=1).T
    res_l['mean'] = mean_l
    res_l['peak'] = peak_l
    res_l=pd.concat(res_l,axis=0)
    return res_l

def get_covar_within_field_by_trial_seperatefield(all_fields_trial,covar_by_trial_pos):
    mean_l = {}
    peak_l = {}
    res_l = {}
    for (uid,field_ind,trial_ind),row in all_fields_trial.iterrows():
        mean = covar_by_trial_pos.loc[trial_ind,row['start']:row['end']].mean()
        peak = covar_by_trial_pos.loc[trial_ind,row['peak']]
        mean_l[(uid,field_ind,trial_ind)]=mean
        peak_l[(uid,field_ind,trial_ind)]=peak
    mean_l = pd.Series(mean_l)
    peak_l = pd.Series(peak_l)
    res_l['mean'] = mean_l
    res_l['peak'] = peak_l
    res_l=pd.concat(res_l,axis=0)
    res_l = res_l.unstack(level=-1) # ([mean,peak] x uid x field id) x ntrials
    return res_l

def get_resid_onetrialtype(all_fields,covar_by_trial_pos,iscommonfield=True,fr_key='fr_mean',covar_key='mean'):
    '''
    one trialtype, one field detection
    all_fields: if commonfield: (uid x field index) x [start end peak ...]; if not commonfield, (uid x field index x trial) x [start end peak...]
    covar_by_trial_pos: result of get_speed_by_trial_pos, but .loc[trial_type]
    '''


    if iscommonfield:
        res_l = get_covar_within_field_by_trial_commonfield(all_fields,covar_by_trial_pos)
    else:
        res_l = get_covar_within_field_by_trial_seperatefield(all_fields,covar_by_trial_pos)
    
    
