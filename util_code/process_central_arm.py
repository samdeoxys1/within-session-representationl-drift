# dealing with the central arm, splitter cells, etc.

import numpy as np
import scipy
import pandas as pd
import copy, pdb, sys, os
import data_prep_pyn as dpp
import change_point_analysis as cpa

# find left right similar fields
def find_left_right_similar_fields(all_fields_both_trialtypes,similar_thresh_in_bin=5,loc_key='peak'):
    left_right_similar_fields = {}
    index_other = []
    # task_ind = 0
    trial_type_ind = 0
    trial_type_to_compare_ind = 1
    # similar_thresh_in_bin = 5 # careful since different maze default binsize might be different
    for (ni,fi),row in all_fields_both_trialtypes.loc[trial_type_ind].iterrows():
        if ni in all_fields_both_trialtypes.loc[trial_type_to_compare_ind].index:
            dist = np.abs(row[loc_key] - all_fields_both_trialtypes.loc[trial_type_to_compare_ind].loc[ni][loc_key] )
            if dist.min() < similar_thresh_in_bin:
                left_right_similar_fields[ni,fi] = row
                fi_other = dist.astype(int).idxmin()
                index_other.append([ni,fi_other])
    index_other = np.array(index_other)
    left_right_similar_fields = pd.DataFrame(left_right_similar_fields).T
    left_right_similar_fields['other_field_index'] = index_other[:,1]
    return left_right_similar_fields

# combine left right similar fields' FR 
def combine_left_right_simlar_fields(pf_fr,selected_fields,index_within_to_trial_index_df):
    '''
    pf_fr: df: (trialtype x fields) x ntrials_within_trialtype; concat from pf_res, index into task type already
    index_within_to_trial_index_df: (trialtype x index within trialtype): index within all ; need [!!] need to be .loc[task_ind] first, after getting it from pa.concat_fr_map_trial_df_d(fr_map_trial_df_d,spk_beh_df) 
    
    ===
    mat: df: fields x alltrials
    '''
    index_corresponding_trialtype_index=0 # index in left_right_similar_fields, which trialtype is it based on
    other_corresponding_trialtype_index=1 # other_field_index in left_right_similar_fields, which trialtype is it based on
    # task_index=0

    ntrials_tot = index_within_to_trial_index_df.max()+1 # trials always 0 indexed
    nfields = selected_fields.shape[0]
    
    mat = np.zeros((nfields,ntrials_tot)) * np.nan

    # for (nn,ff),row in selected_fields.iterrows():
    field_inds_onetrialtype=selected_fields.index
    field_inds_othertrialtype = selected_fields.reset_index(level=1).set_index('other_field_index',append=True).index
    # pdb.set_trace()
    pf_fr_onetrialtype = pf_fr.loc[index_corresponding_trialtype_index].loc[field_inds_onetrialtype].dropna(axis=1,how='all')
    index_within_to_trial_index_df_onett = index_within_to_trial_index_df.loc[index_corresponding_trialtype_index]
    common_ind = pf_fr_onetrialtype.columns.intersection(index_within_to_trial_index_df_onett.index)
    mat[:,index_within_to_trial_index_df_onett.loc[common_ind].values] = pf_fr_onetrialtype[common_ind].values
    
    pf_fr_othertrialtype = pf_fr.loc[other_corresponding_trialtype_index].loc[field_inds_othertrialtype].dropna(axis=1,how='all')
    pf_fr_othertrialtype_col = pf_fr_othertrialtype.columns
    # pdb.set_trace()
    index_within_to_trial_index_df_sub=index_within_to_trial_index_df.loc[other_corresponding_trialtype_index]
    available_index_within=index_within_to_trial_index_df_sub.index.intersection(pf_fr_othertrialtype_col)
    mat[:,index_within_to_trial_index_df_sub.loc[available_index_within].values] = pf_fr_othertrialtype[available_index_within].values
    mat = pd.DataFrame(mat,index=field_inds_onetrialtype)
    mat = mat.dropna(axis=1,how='all')
    
    return mat

# get fields on the central arm
def get_central_fields(all_fields_all_trialtype,central_arm_bounds_cm = np.array([0,74]),
    bin_width = 2.2,pos_key = 'com',within_central_ratio_thresh=0.7):
    '''
    all_fields_all_trialtype: df: (task, trialtype, fields) x [start, end, com, peak, fr_peak, fr_mean, ...]
    '''
    central_arm_bounds_bin = (central_arm_bounds_cm // bin_width).astype(int)
    # check peak/com in center
    pos_in_central = all_fields_all_trialtype[pos_key] <= central_arm_bounds_cm[1]
    
    # check large fraction in center
    width_within_bound=np.minimum(all_fields_all_trialtype['end'],central_arm_bounds_bin[1]) - np.maximum(all_fields_all_trialtype['start'],central_arm_bounds_bin[0])
    width_within_bound.loc[width_within_bound<0] = 0
    width = all_fields_all_trialtype['end'] - all_fields_all_trialtype['start']
    width_within_bound_ratio = width_within_bound / width
    central_field_ma = (width_within_bound_ratio > within_central_ratio_thresh) & pos_in_central
    central_fields_all_trialtype  =all_fields_all_trialtype.loc[central_field_ma]
    return central_fields_all_trialtype


# get splitter cells
def get_left_right_intensity_difference(mat,index_within_to_trial_index_df):
    '''
    mat:nfields x ntrials_tot, df
    index_within_to_trial_index_df: series, (trialtype, trial_index_within_trialtype): trial_index_within_all_trials; caution similar to combine_left_right_simlar_fields
    '''
    trialtype_0_inds = list(set(index_within_to_trial_index_df.loc[0].values).intersection(set(mat.columns)))
    trialtype_1_inds = list(set(index_within_to_trial_index_df.loc[1].values).intersection(set(mat.columns)))
    ttest_res = scipy.stats.ttest_ind(mat[trialtype_0_inds].T,mat[trialtype_1_inds].T)
    ttest_res = pd.DataFrame(ttest_res,index=['t','pval']).T
    ttest_res.index=  mat.index
    return ttest_res
    
# combine things
def combine_changes_df_side_central(changes_df_central,changes_df_both_trialtype,nonsplitter_fields,index_within_to_trial_index_df):
    '''
    if there's already changes_df computed seperately for two trial types, subselect the necessary ones (side arm, central dissimilar center, central splitter),
    concat into index within whole session, then concat with the central arm changes_df
    
    index_within_to_trial_index_df: series, (trialtype, trial_index_within_trialtype): trial_index_within_all_trials; caution similar to combine_left_right_simlar_fields
    '''
    
    inds0 = changes_df_both_trialtype.loc[0].index.difference(nonsplitter_fields.index)
    changes0 = changes_df_both_trialtype.loc[0].loc[inds0].dropna(axis=1)
    changes0_recol = copy.copy(changes0)

    index_within_sub = index_within_to_trial_index_df.loc[0]
    c0col=changes0.columns
    avail_ind_within=index_within_sub.index.intersection(c0col)
    changes0_recol = changes0_recol[avail_ind_within]
    changes0_recol.columns = index_within_sub.loc[avail_ind_within]
    
    splitter_index_in_tt1 = nonsplitter_fields.reset_index(level=1).set_index('other_field_index',append=True).index
    inds1 = changes_df_both_trialtype.loc[1].index.difference(splitter_index_in_tt1)
    changes1 = changes_df_both_trialtype.loc[1].loc[inds1].dropna(axis=1).dropna(axis=1)
    changes1_recol = copy.copy(changes1)

    index_within_sub = index_within_to_trial_index_df.loc[1]
    c1col=changes1.columns
    avail_ind_within=index_within_sub.index.intersection(c1col)
    changes1_recol = changes1_recol[avail_ind_within]
    changes1_recol.columns = index_within_sub.loc[avail_ind_within]

    # changes1_recol.columns = index_within_to_trial_index_df.loc[1].loc[changes1.columns]
    changes_df_combined_d ={'both':changes_df_central,0:changes0,1:changes1}
    changes_df_combined = pd.concat({'both':changes_df_central,0:changes0_recol,1:changes1_recol},axis=0)
    return changes_df_combined,changes_df_combined_d

def combine_field_loc(pf_res,nonsplitter_fields,index_within_to_trial_index_df,loc_key='peak',task_ind = 0):
    '''
    pf_res: result from place_field_analysis
    nonsplitter_fields: df: nfields x [infos], shared fields
    index_within_to_trial_index_df: from dpp.get_index_within_to_trial_index_df
    ===
    pf_loc_combined: ((both,0,1) x fields) x alltrials
    '''
    
    pf_loc_both_tt = pd.concat(pf_res['avg']['params'],axis=0).loc[task_ind,slice(None),loc_key] # consider move this outside of the function
    pf_allfield_both_tt =pd.concat(pf_res['avg']['all_fields'],axis=0).loc[task_ind]
    nonsplitter_fields_ind_in_both_tt_d = {0:nonsplitter_fields.index,1:nonsplitter_fields.reset_index(level=1).set_index('other_field_index',append=True).index}
    
    nfields = pf_loc_both_tt.shape[0] - nonsplitter_fields.shape[0]
    if task_ind is not None:
        index_within_to_trial_index_df = index_within_to_trial_index_df.loc[task_ind]
    
    ntrials = len(index_within_to_trial_index_df)
    
    shared_inds = nonsplitter_fields_ind_in_both_tt_d[0]
    sep_inds_d = {}
    for tt in [0,1]:
        sep_inds_d[tt] = pf_loc_both_tt.loc[tt].index.difference(nonsplitter_fields_ind_in_both_tt_d[tt]) # importance, use the inds within that trialtype even for the shared fields
    
    all_inds = pd.concat({'both':pd.DataFrame(np.asarray(shared_inds)),0:pd.DataFrame(np.asarray(sep_inds_d[0])),1:pd.DataFrame(np.asarray(sep_inds_d[1]))})
    
    all_inds = all_inds.reset_index(level=0)
    # all_inds = np.concatenate([shared_inds,sep_inds_d[0],sep_inds_d[1]])
    all_inds = pd.MultiIndex.from_frame(all_inds)
    
    # pf_loc_combined = pd.DataFrame(np.zeros((nfields, ntrials)),index=all_inds)
    pf_loc_combined ={}
    nfields_central = shared_inds.shape[0]
    pf_loc_central_combined = pd.DataFrame(np.nan * np.zeros((nfields_central,ntrials)),index=shared_inds)
    pf_loc_combined['both'] = pf_loc_central_combined

    pf_all_field_combined = {}

    for tt in [0,1]:        
        inds = np.concatenate([nonsplitter_fields_ind_in_both_tt_d[tt],sep_inds_d[tt]])
        pf_loc_one_tt_recol = pf_loc_both_tt.loc[tt].loc[inds].dropna(axis=1,how='all')
        
        # pf_loc_one_tt_central_recol = pf_loc_both_tt.loc[tt].loc[nonsplitter_fields_ind_in_both_tt_d[tt]].dropna(axis=1,how='all')
        
        cols_tt = index_within_to_trial_index_df.loc[tt]
        cols_tt = cols_tt.loc[pf_loc_one_tt_recol.columns] # deal with bad trial issue; subselect index_within
        
        pf_loc_central_combined.loc[shared_inds,cols_tt] = pf_loc_both_tt.loc[tt].loc[nonsplitter_fields_ind_in_both_tt_d[tt]].dropna(axis=1,how='all').values


        pf_loc_one_tt_side_recol = pf_loc_both_tt.loc[tt].loc[sep_inds_d[tt]].dropna(axis=1,how='all').interpolate(method='linear',axis=1)
        
        pf_loc_one_tt_side_recol.columns = cols_tt.values #same as above, bad trial correction #index_within_to_trial_index_df.loc[tt].values
        
        
        pf_loc_combined[tt] = pf_loc_one_tt_side_recol
    # pf_loc_central_combined = pd.concat(pf_loc_central_combined,join='inner') # inner join for central arms
    pf_loc_central_combined = pf_loc_central_combined.interpolate(method='linear',axis=1)

    pf_loc_combined = pd.concat(pf_loc_combined,axis=0,join='outer')
    for k,val in pf_loc_combined.groupby(level=0): #k: 'both', 0, 1; val: df: field x trials
        if k=='both': # if a field is present in both, then its index in 0 is used
            k_orig=0
        else:
            k_orig = k
        
        pf_all_field_combined[k] = pf_allfield_both_tt.loc[k_orig].loc[val.droplevel(0).index]
    
    pf_all_field_combined = pd.concat(pf_all_field_combined,axis=0)
    
    return pf_loc_combined,pf_all_field_combined

def combine_pf_res(pf_params,all_fields,beh_df=None,task_l=None,corners_df=None,index_within_to_trial_index_df=None,**kwargs):
    if beh_df is not None:
        speed_key = 'directed_locomotion'
        speed_thresh = 0.5
        corners_d,xy_sampled_d,segment_d=dpp.find_tmaze_turns(beh_df,speed_key=speed_key,speed_thresh=speed_thresh)
        corners_df = pd.concat(corners_d)
        task_l = beh_df.groupby('task_index')['task'].first()
        index_within_to_trial_index_df = dpp.index_within_to_trial_index(beh_df)

    kwargs_ = {'similar_thresh_in_bin':5,'loc_key':'peak','fr_key':'fr_peak'}
    kwargs_.update(kwargs)
    
    task_ind_l = all_fields.index.get_level_values(0).unique()
    all_fields_recombined_alltask = {}
    pf_par_recombined_alltask = {}
    for task_ind in task_ind_l:
        task = task_l[task_ind]
        if task_ind in corners_df.index.get_level_values(0):
            central_arm_bounds = np.array([0,corners_df.loc[0]['lin'].loc[(slice(None),1)].mean()]) 

        if 'alternation' in task.lower():
            all_fields_one = all_fields.loc[task_ind]
            pf_params_one = pf_params.loc[task_ind]
            pf_fr_one = pf_params_one.loc[(slice(None),kwargs_['fr_key']),:].droplevel(1) # drop the fr_key level

            # get fields similar in location in left and right trials
            left_right_similar_fields=find_left_right_similar_fields(all_fields_one,similar_thresh_in_bin=kwargs_['similar_thresh_in_bin'],loc_key=kwargs_['loc_key'])

            # among those fields, get the fields on the central arm
            central_fields_all_trialtype = get_central_fields(all_fields_one,central_arm_bounds_cm = central_arm_bounds)
            central_lrsim_inds = left_right_similar_fields.index.intersection(central_fields_all_trialtype.loc[0].index)
            central_lrsim_fields = left_right_similar_fields.loc[central_lrsim_inds]

            noncentral_fields_all_trialtype_inds = [x for x in all_fields_one.index if x not in central_fields_all_trialtype.index]
            noncentral_fields_all_trialtype = all_fields_one.loc[noncentral_fields_all_trialtype_inds]

            # combine trial types for the central left_right location similar cells
            pf_fr_trialtype_combined = combine_left_right_simlar_fields(pf_fr_one,selected_fields=central_lrsim_fields, index_within_to_trial_index_df=index_within_to_trial_index_df.loc[task_ind])

            # test the left right firing rate differences for the l-r location similar cells
            lr_fr_diff = get_left_right_intensity_difference(pf_fr_trialtype_combined, index_within_to_trial_index_df.loc[task_ind])
            alpha = 0.05
            nonsplitter_inds = lr_fr_diff.loc[lr_fr_diff['pval'] >= alpha].index
            splitter_inds = lr_fr_diff.loc[lr_fr_diff['pval'] < alpha].index
            splitter_fields = central_lrsim_fields.loc[splitter_inds] 
            nonsplitter_fields = central_lrsim_fields.loc[nonsplitter_inds] # this will be NEEDED!

            # prepare nonsplitter fields to be added
            nonsplitter_fields_both = pd.concat({0:nonsplitter_fields.drop('other_field_index',axis=1),
            1:nonsplitter_fields.droplevel(1).set_index('other_field_index',append=True)},axis=0) # don't use, just use the index
            nonsplitter_fields_both_ind = nonsplitter_fields_both.index
            nonsplitter_fields_added_index = nonsplitter_fields.assign(trial_type='both')
            nonsplitter_fields_added_index=nonsplitter_fields_added_index.set_index('trial_type',append=True).swaplevel(0,2).swaplevel(1,2)

            # get fields that should be seperated by trialtype
            trialtype_seperate_fields_ind = all_fields_one.index.difference(nonsplitter_fields_both_ind)

            # combine all fields
            all_fields_one_recombined = pd.concat([all_fields_one.loc[trialtype_seperate_fields_ind],nonsplitter_fields_added_index],axis=0)
            all_fields_recombined_alltask[task_ind] = all_fields_one_recombined

            # combine pf_params
            pf_params_gpb = pf_params_one.groupby(level=1)
            pf_par_recombined_d = {}
            for par_key,pf_par in pf_params_gpb:
                pf_par = pf_par.droplevel(1)

                # combine pf_params for nonsplitter
                nonsplitter_pf_param_combined = combine_left_right_simlar_fields(pf_par,nonsplitter_fields,index_within_to_trial_index_df.loc[task_ind])

                # combine pf_params for trialtype seperate fields
                index_within_to_trial_index_df_onetask = index_within_to_trial_index_df.loc[task_ind]
                gpb = pf_par.loc[trialtype_seperate_fields_ind].groupby(level=0)
                pf_par_trialtype_seperate_fields = {}
                for tt,val in gpb:
                    val = val.loc[tt]
                    val = val.dropna(axis=1,how='all')
                    cols = val.columns
                    index_within_to_trial_index_df_onett = index_within_to_trial_index_df_onetask.loc[tt]
                    common_index_within = cols.intersection(index_within_to_trial_index_df_onett.index)
                    val = val[common_index_within]
                    val.columns = index_within_to_trial_index_df_onett.loc[common_index_within]
                    pf_par_trialtype_seperate_fields[tt] = val
                pf_par_trialtype_seperate_fields = pd.concat(pf_par_trialtype_seperate_fields)
                pf_par_trialtype_seperate_fields = pf_par_trialtype_seperate_fields.sort_index(axis=1)

                nonsplitter_pf_param_combined_index_added=pd.concat({'both':nonsplitter_pf_param_combined})
                pf_par_recombined = pd.concat([pf_par_trialtype_seperate_fields,nonsplitter_pf_param_combined_index_added],axis=0)
                pf_par_recombined_d[par_key] = pf_par_recombined
            
            pf_par_recombined_d = pd.concat(pf_par_recombined_d)
            pf_par_recombined_alltask[task_ind] = pf_par_recombined_d
        elif 'linear' in task.lower(): 
            all_fields_one = all_fields.loc[task_ind]
            pf_params_one = pf_params.loc[task_ind].swaplevel(0,1)
            
            pf_params_one_tt_l =[]
            for tt in [0,1]:
                ind_map = index_within_to_trial_index_df.loc[task_ind,tt]
                pf_params_one_tt =pf_params_one.loc[(slice(None),[tt]),:].dropna(axis=1,how='all')
                cols_intersect = pf_params_one_tt.columns.intersection(ind_map.index)
                # pdb.set_trace()
                pf_params_one_tt = pf_params_one_tt.loc[:,cols_intersect]
                pf_params_one_tt.columns = ind_map.loc[pf_params_one_tt.columns]
                pf_params_one_tt_l.append(pf_params_one_tt)
            pf_params_one_tt_l = pd.concat(pf_params_one_tt_l,axis=0).sort_index(axis=1)
            
            all_fields_recombined_alltask[task_ind] = all_fields_one
            pf_par_recombined_alltask[task_ind] = pf_params_one_tt_l
        else:
            raise NotImplementedError
            
    all_fields_recombined_alltask = pd.concat(all_fields_recombined_alltask)
    pf_par_recombined_alltask = pd.concat(pf_par_recombined_alltask)
    pf_par_recombined_alltask = pf_par_recombined_alltask.swaplevel(0,1)
    return pf_par_recombined_alltask, all_fields_recombined_alltask
    

    
# get switch time related
def get_trial_pos_info(spk_beh_df,speed_key = 'v',speed_key_to_keep='speed_gauss',speed_thresh = 1,n_lin_bins=100,task_ind=0):
    '''
    get the look up table for some beh_var at eac trial & pos
    
    trial_pos_info: df: (trial, lin_binned) x [time, v, task_index, visitedArm, correct]
    '''
    spk_beh_df = spk_beh_df.query('task_index==@task_ind')
    if 'lin_binned' not in spk_beh_df.columns:
        spk_beh_df,_=dpp.add_lin_binned(spk_beh_df,nbins=n_lin_bins)
    spk_beh_df_withtime = spk_beh_df.reset_index().rename(columns={'Time (s)':'time'})
    spk_beh_df_withtime = spk_beh_df_withtime.query(f'{speed_key}>@speed_thresh')
    trial_pos_info = spk_beh_df_withtime.groupby(['trial','lin_binned']).median()[['time',speed_key_to_keep,'task_index','visitedArm','correct']]
    return trial_pos_info

def get_field_loc_per_trial(pf_res, task_ind,tt_ind,loc_key='peak'):
    '''
    just a wrapper for convenience; content might subject to change
    
    field_loc: nfields x ntrials_within_one_tt
    '''
    field_loc = pf_res['avg']['params'][task_ind,tt_ind].loc[loc_key]
    field_loc=field_loc.interpolate(method='linear',axis=1) # interpolate to get rid of the nan
    return field_loc


def get_all_switch_times_combined(pos_to_time_func_per_trial,pf_loc_combined,changes_df_combined,speed_key='speed_gauss'):
    '''
    for combined changes_df: nfields x ntrials_tot; shared fields have all trials, seperate fields only have part of the trials
    '''
    all_sw_times_v = []
    
    for trial_index in changes_df_combined.columns:
        if trial_index in pos_to_time_func_per_trial.keys():

            sw = (changes_df_combined.loc[:,trial_index] ==1) | (changes_df_combined.loc[:,trial_index] ==-1) # rulling out np.nan!!!
            nonzero_inds_in_sw = sw.to_numpy().nonzero()
            sw_nonzero_vals = changes_df_combined.loc[:,trial_index].values[nonzero_inds_in_sw]
            sw_fields = changes_df_combined.index[nonzero_inds_in_sw[0]]
        #     sw_fields_pos = all_fields_onett[pos_key].loc[sw_fields]
            sw_fields_pos = pf_loc_combined[trial_index].loc[sw_fields]
        
            interp_func_res = pos_to_time_func_per_trial[trial_index](sw_fields_pos.values.astype(float)).T

            if interp_func_res.shape[0]>0:
                
                interp_func_res = pd.DataFrame(interp_func_res,index=sw_fields,columns=['time',speed_key])
        #         interp_func_res.index.name = ('uid','field_index')
                interp_func_res['field_pos'] = sw_fields_pos
                interp_func_res = interp_func_res.reset_index().rename(columns={'level_0':'trialtype','level_1':'uid','level_2':'field_index'})
                # interp_func_res['index_within_trialtype'] = trial_within
                interp_func_res['trial_index'] = trial_index
                
                interp_func_res['switch']=sw_nonzero_vals

                all_sw_times_v.append(interp_func_res)
    all_sw_times_v = pd.concat(all_sw_times_v,axis=0)
    return all_sw_times_v
    
def get_all_place_field_times_combined(pos_to_time_func_per_trial,pf_loc_combined):
    '''
    similar to get_all_switch_times_combined,
    get timestamps and v for when the animal is at a place field at all trials, instead of only at the switch trials
    
    all_field_times_v: (n_fields x ntrials)  x [trialtype, uid, field_index, time, v, trial_index]
    '''
    all_field_times_v =[]
    
    for trial, func in pos_to_time_func_per_trial.items():
        
        all_fields_time_v_one_trial_df = pd.DataFrame(func(pf_loc_combined[trial]).T,columns=['time','v'],index=pf_loc_combined[trial].index) 
        all_fields_time_v_one_trial_df['field_pos']=pf_loc_combined[trial]
        all_fields_time_v_one_trial_df = all_fields_time_v_one_trial_df.reset_index().rename(columns={'level_0':'trialtype','level_1':'uid','level_2':'field_index'}) # nfields x [time, v]
        all_fields_time_v_one_trial_df['trial_index'] = trial
        all_field_times_v.append(all_fields_time_v_one_trial_df)
    
    all_field_times_v = pd.concat(all_field_times_v,axis=0)
    all_field_times_v = all_field_times_v.dropna(axis=0)
    return all_field_times_v


def get_all_switch_times(pos_to_time_func_per_trial,field_loc,changes_df_onett,index_within_to_trial_index_df=None,task_ind=0,tt_ind=0):
    '''
    this is for trial type seperated changes_df

    pos_to_time_func_per_trial, from get_pos_to_time_func_per_trial
    field_loc, from get_field_loc_per_trial
    changes_df_onett, nfields x ntrials_within
    index_within_to_trial_index_df, from dpp.index_within_to_trial_index
    '''
    all_sw_times_v = []
    for trial_within in changes_df_onett.columns:
        if index_within_to_trial_index_df is not None:
            trial_index = index_within_to_trial_index_df.loc[task_ind,tt_ind].loc[trial_within]
        else:
            trial_index = trial_within
        sw = changes_df_onett.loc[:,trial_within] !=0
        sw_fields = changes_df_onett.index[sw.to_numpy().nonzero()[0]]
    #     sw_fields_pos = all_fields_onett[pos_key].loc[sw_fields]
        sw_fields_pos = field_loc[trial_within].loc[sw_fields]
        
        interp_func_res = pos_to_time_func_per_trial[trial_index](sw_fields_pos.values.astype(float)).T
        
        if interp_func_res.shape[0]>0:
            interp_func_res = pd.DataFrame(interp_func_res,index=sw_fields,columns=['time','v'])
    #         interp_func_res.index.name = ('uid','field_index')
            interp_func_res['field_pos'] = sw_fields_pos
            interp_func_res = interp_func_res.reset_index().rename(columns={'level_0':'uid','level_1':'field_index'})
            interp_func_res['index_within_trialtype'] = trial_within
            interp_func_res['trial_index'] = trial_index

            all_sw_times_v.append(interp_func_res)
    all_sw_times_v = pd.concat(all_sw_times_v,axis=0)
    all_sw_times_v['task_ind'] =task_ind
    all_sw_times_v['trial_type_ind'] = tt_ind
    
    return all_sw_times_v
from scipy.interpolate import interp1d
def get_pos_to_time_func_per_trial(trial_pos_info,speed_key='speed_gauss'):
    '''
    trial_pos_info: from get_trial_pos_info
    ===
    pos_to_time_func_per_trial:{trial: func (pos -> time and v)}
    
    '''
    pos_to_time_func_per_trial = {}
    for trial, val in trial_pos_info.groupby(level=0):
        pos_to_time_func_per_trial[trial] = interp1d(val.loc[trial].index,val[['time',speed_key]].T,fill_value='extrapolate')
    return pos_to_time_func_per_trial

def get_sw_times_pairwise_diff(all_sw_times_v,bins=10):
    sw_times_all =  all_sw_times_v['time'].values
    sw_times_all_diff = sw_times_all[:,None] - sw_times_all[None,:]
    diff_triu = sw_times_all_diff[np.triu_indices_from(sw_times_all_diff,k=1)]
    
    pmf,bin_edges = np.histogram(diff_triu,bins,density=False) 
    pmf  = pmf/ len(diff_triu)
    cdf = np.cumsum(pmf)
    return cdf, pmf, bin_edges

def gen_circular_shuffle_changes_df_combined(changes_df_combined_d,index_within_to_trial_index_df,nrepeats=100):
    '''
    circularly shuffle: 
    central nonsplitter fields across all trials; side specific fields within trialtype
    recombine 

    changes_df_combined_d: {'both':, 0:, 1:}, each val: changes_df: fields x trials
    index_within_to_trial_index_df: (trialtype, index within trialtype): trial index
    '''
    changes_df_shuffle_d = {}
    for k,val in changes_df_combined_d.items():
        changes_df_shuffle = cpa.gen_circular_shuffle(val, nrepeats=nrepeats)
        changes_df_shuffle_d[k] = changes_df_shuffle
    changes_df_combined_shuffle_l = []
    for i in range(nrepeats):
        for tt in [0,1]:
            
            cd_cols = changes_df_shuffle_d[tt][i].columns 
            changes_df_shuffle_d[tt][i].columns = index_within_to_trial_index_df.loc[tt].loc[cd_cols] # bad trial fix
        changes_df_combined_shuffle = pd.concat({kk:vv[i] for kk,vv in changes_df_shuffle_d.items()})
        changes_df_combined_shuffle_l.append(changes_df_combined_shuffle)
    return changes_df_combined_shuffle_l         


def divide_central_fields_splitter_gather_params(all_fields,pf_fr,index_within_to_trial_index_df,task_ind=0,**kwargs):
    '''
    all_fields: from eg. pf_res['avg']['all_fields'].loc[task_ind], (trial_ind x neuron x field) x [start, end, com, peak, fr_peak, fr_mean]
    pf_fr: from eg. 
            fr_key = 'fr_peak'
            pf_fr = pd.concat(pf_res['avg']['params'],axis=0).loc[(slice(None),slice(None),fr_key),:]
            index=pf_fr.index.droplevel(2)
            pf_fr.index=index
            pf_fr = pf_fr.loc[task_ind]
    index_within_to_trial_index_df: 
    
    '''
    kwargs_ = {'similar_thresh_in_bin':5,'loc_key':'peak'}
    kwargs_.update(kwargs)

    # get fields similar in location in left and right trials
    left_right_similar_fields=find_left_right_similar_fields(all_fields,similar_thresh_in_bin=kwargs_['similar_thresh_in_bin'],loc_key=kwargs_['loc_key'])
    
    # among those fields, get the fields on the central arm
    central_fields_all_trialtype = get_central_fields(all_fields)
    central_lrsim_inds = left_right_similar_fields.index.intersection(central_fields_all_trialtype.loc[0].index)
    central_lrsim_fields = left_right_similar_fields.loc[central_lrsim_inds]

    # combine trial types for the central left_right location similar cells
    pf_fr_trialtype_combined = combine_left_right_simlar_fields(pf_fr,selected_fields=central_lrsim_fields, index_within_to_trial_index_df=index_within_to_trial_index_df.loc[task_ind])

    # test the left right firing rate differences for the l-r location similar cells
    lr_fr_diff = get_left_right_intensity_difference(pf_fr_trialtype_combined, index_within_to_trial_index_df.loc[task_ind])
    alpha = 0.05
    nonsplitter_inds = lr_fr_diff.loc[lr_fr_diff['pval'] >= alpha].index
    splitter_inds = lr_fr_diff.loc[lr_fr_diff['pval'] < alpha].index
    splitter_fields = central_lrsim_fields.loc[splitter_inds] 
    nonsplitter_fields = central_lrsim_fields.loc[nonsplitter_inds] # this will be NEEDED!

    pf_fr_trialtype_combined_nonsplitter = pf_fr_trialtype_combined.loc[nonsplitter_inds]
    pf_fr_trialtype_combined_splitter = pf_fr_trialtype_combined.loc[splitter_inds]
    
    return splitter_fields,nonsplitter_fields, pf_fr_trialtype_combined_splitter, pf_fr_trialtype_combined_nonsplitter