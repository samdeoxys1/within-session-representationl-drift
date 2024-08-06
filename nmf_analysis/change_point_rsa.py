import numpy as np
import pandas as pd
import scipy
import change_point_analysis as cpa

def reduce_get_triu_flatten(mat,mask = None):
    '''
    reduce multi field to one field using the max (similarity) value
    then get the upper tri elements, flattened
    mask: see get_onoff_outer_reduced_flatten
    '''
    mat_reduced = mat.groupby(level=0).max().groupby(level=0,axis=1).max() # reduce multi field to one field using the max (similarity) value
    neurons_left = mat_reduced.index
    mat_val = mat_reduced.values
    
    mat_triu_inds = np.triu_indices(mat_val.shape[0],1)
    flattened = mat_val[mat_triu_inds]

    if mask is None:
        mask = pd.DataFrame(np.zeros_like(mat.values),index=mat.index,columns=mat.columns)
    
    # if mask is not None:
    mask_reduced = mask.groupby(level=0).max().groupby(level=0,axis=1).max() 
    mask_flattened =  mask_reduced.values[mat_triu_inds]
    flattened_masked = flattened[np.logical_not(mask_flattened)]
    return flattened_masked,neurons_left,flattened,mask_flattened
    
    
def get_onoff_outer_reduced_flatten(changes_df,mask=None):
    '''
    mask: mask out pairs that are too close, 1 to mask out 0 to keep; first try conservative: if two fields are two close, then the two neurons are eliminated
    '''
    _,_,_,_,_,onoff_outer = cpa.get_shared_onoff(changes_df,return_outer=True)
    onoff_outer_reduced_flatten_d = {}
    flattened_no_mask_d = {} #no mask
    for k in onoff_outer.keys():
#         outer_reduced_one = onoff_outer[k].groupby(level=0).max().groupby(level=0,axis=1).max() # reduce multi fields in the same neuron using OR

#         neurons_left = outer_reduced_one.index # should be the same across on off
#         triu_inds = np.triu_indices(neurons_left.shape[0],1)
#         onoff_outer_reduced_flatten=outer_reduced_one.values[triu_inds]
        
        # onoff_outer_reduced_flatten,neurons_left = reduce_get_triu_flatten(onoff_outer[k],mask = mask)
        onoff_outer_reduced_flatten_masked,neurons_left,flattened,mask_flattened = reduce_get_triu_flatten(onoff_outer[k],mask = mask)
        
        onoff_outer_reduced_flatten_d[k] = onoff_outer_reduced_flatten_masked
        flattened_no_mask_d[k] = flattened
    return onoff_outer_reduced_flatten_d,neurons_left,flattened_no_mask_d,mask_flattened
    
def get_pairwise_field_overlap(all_field_bounds):
    '''
    get pariwise overlap of place fields
    
    all_field_bounds: df; (neuron x field index) x ['start','end']
    '''
    mat = np.zeros((all_field_bounds.shape[0],all_field_bounds.shape[0]))
    for iii,(ii,row) in enumerate(all_field_bounds.iterrows()):
        for jjj,(jj,col) in enumerate(all_field_bounds.iterrows()):
            if (row['start'] <= col['end']) and (col['start'] <= row['end']):
                mat[iii,jjj]=np.minimum(row['end'],col['end']) - np.maximum(row['start'],col['start'])
    
    pairwise_field_overlap = pd.DataFrame(mat,index=all_field_bounds.index,columns=all_field_bounds.index)
    width = (all_field_bounds['end']-all_field_bounds['start']).values[:,None]
    width_sum_pair = width + width.T
    pairwise_field_overlap_ratio_width_sum = pairwise_field_overlap / width_sum_pair
    return pairwise_field_overlap, pairwise_field_overlap_ratio_width_sum

# def process_pairwise_field_regressors():
#     ''' 
#     process all pairwise field matrices, turn into a flattened vector of the upper triag, after reducing fields to neuron
#     '''

import copy
def binned_ttest(reg_mat,key_to_bin,nbins=5,target_key='sim_diff',to_test_key='coswitch_on_cpd'):
    reg_mat_sub = copy.copy(reg_mat[[key_to_bin,to_test_key,target_key]])
    key_to_bin_binned = key_to_bin+'_binned'
    reg_mat_sub[key_to_bin_binned],bins = pd.cut(reg_mat_sub[key_to_bin],nbins,retbins=True,labels=False)
    gpb = reg_mat_sub.groupby([key_to_bin_binned])
    
    ttest_res_d =  {}
    for ol, val in gpb:
        ttest_res_d[ol] = scipy.stats.ttest_ind(val.loc[val[to_test_key]==True][target_key],val.loc[val[to_test_key]==False][target_key]) 
    ttest_res_d = pd.DataFrame(ttest_res_d,index=['t','p_val']).T
    ttest_res_d['bin_start'] = bins[:-1]

    gpb = reg_mat_sub.groupby([to_test_key,key_to_bin_binned])
    target_grouped_summary = gpb[target_key].agg(mean='mean',sem='sem')
    target_grouped_summary = target_grouped_summary.unstack()

    return ttest_res_d, target_grouped_summary