'''
different ways to get pairwise similarities, including ripple cofiring, theta correlation, monosynaptic, coswitching etc.
'''
import numpy as np
import scipy
import matplotlib.pyplot as plt
import sys,os,copy,pdb,importlib
from importlib import reload
import pandas as pd
import misc
import itertools
from itertools import product
sys.path.append('/mnt/home/szheng/projects/nmf_analysis')
sys.path.append('/mnt/home/szheng/projects/util_code')
sys.path.append('/mnt/home/szheng/projects/seq_detection2/code')

import data_prep_pyn as dpp
import place_cell_analysis as pa
import place_field_analysis as pf
# import switch_analysis_one_session as saos
import change_point_analysis_central_arm_seperate as cpacas
import process_central_arm as pca 
import plot_helper as ph
import preprocess as prep
reload(ph)

from scipy.spatial.distance import pdist, squareform

def get_all_ripple_pairwise_sim(cell_metrics,ripple_events,mergepoints):
    '''

    ====
    ripple_sim_d: df: (window x key x epoch x n_neuron) x n_neuron; window: ripple_only or extended; key: count_in_interval, etc.; epoch: 0,1,2,for pre beh post
    '''
    res_all_epochs_d, ripple_time_ints_epochs_d = prep.get_spike_count_rate_participation_in_ripple_all(cell_metrics,
                                                                                                    ripple_events,
                                                                                                    mergepoints,
                                                                                                   )
    # sim_all_window = {}
    sim_all_d = {}
    nepochs = mergepoints.timestamps.shape[0]
    for window,val in res_all_epochs_d.items():
        # sim_one_window_all_key = {}
        for key,valval in val.items():
            # sim_one_key_all_epochs = {}
            for e in range(nepochs):

                mat = valval[e]
                if ('count' in key) or ('rate' in key):
                    sim = mat.T.corr()
                elif ('participation' in key):
                    sim = squareform(1-pdist(mat,metric='jaccard'))
                    sim = pd.DataFrame(sim,index=mat.index,columns=mat.index)
                
                sim_all_d[(window,key,e)]=sim
            # sim_one_window_all_key[key] = sim_one_key_all_epochs
        # sim_all_window[window] = sim_one_window_all_key
    ripple_sim_d = pd.concat(sim_all_d,axis=0)
    ripple_sim_d.index = ripple_sim_d.index.set_names(['window','ripple_sim_type','epoch','uid'])
    return ripple_sim_d




def get_beh_corr(spk_beh_df,cell_cols=None,window=3,window_size=0.1,speed_key='v',speed_thresh=1.):
    def corr_one_val(val):
        if window_size is not None:
            if 'time' in val.columns:
                dt = np.median(np.diff(val['time']))
            else:
                dt = np.median(np.diff(val.index))
            window = int(window_size / dt)
            val_binned = val[cell_cols].rolling(window).sum()[::window]
            corr = val_binned[cell_cols].corr()
            return corr


    spk_beh_df = spk_beh_df.loc[spk_beh_df[speed_key]>=speed_thresh]
    corr_d = {}
    if cell_cols is None:
        cell_cols =[c for c in spk_beh_df.columns if (isinstance(c,int) or isinstance(c,float))]
    # if len(groupby_keys)>0:
    gpb = spk_beh_df.groupby('trial_type')
    for k,val in gpb:
        corr = corr_one_val(val)
        corr_d[k]=corr
    gpb = spk_beh_df.groupby('task_index')
    for k,val in gpb:
        corr = corr_one_val(val)
        corr_d[(k,'both')]=corr # assuming trial_type = (task_index, trialtype_index), no further levels
    corr_d = pd.concat(corr_d,axis=0)
    corr_d.index = corr_d.index.set_names(['task_index','trialtype_index','uid'])
    return corr_d 

def get_field_overlap_pair(pf_all_field_combined,uid=None):
    
    st=pf_all_field_combined['start'].values
    ed = pf_all_field_combined['end'].values
    width = (pf_all_field_combined['end'] - pf_all_field_combined['start']).values
    overlap = np.minimum.outer(ed,ed) - np.maximum.outer(st,st)
    overlap[overlap<0]=0
    width_sum = np.add.outer(width,width)
    overlap_ratio = overlap * 2 / width_sum

    triu_ind=np.triu_indices_from(overlap_ratio,0)
    overlap_ratio_flatten=overlap_ratio[triu_ind]
    n_field = pf_all_field_combined.shape[0]
    ind_pair_tensor = np.zeros((n_field,n_field,2))
    if uid is None:
        uid = pf_all_field_combined.index.get_level_values(1) # subject to change
    ind_pair_tensor[:,:,0] = np.tile(uid[:,None],[1,n_field])
    ind_pair_tensor[:,:,1] = np.tile(uid[None,:],[n_field,1])
    ind_pair_tensor_flatten = ind_pair_tensor[triu_ind]

    overlap_ratio_df = np.concatenate([ind_pair_tensor_flatten,overlap_ratio_flatten[:,None]],axis=1)
    overlap_ratio_df = pd.DataFrame(overlap_ratio_df,columns=['uid_0','uid_1','overlap_ratio'])
    overlap_ratio_max_df = overlap_ratio_df.groupby(['uid_0','uid_1'])['overlap_ratio'].max()
    overlap_ratio_max_df_unstack=overlap_ratio_max_df.unstack()
    overlap_ratio_max_df_unstack = overlap_ratio_max_df_unstack.fillna(0)
    overlap_ratio_max_df_final = np.maximum(overlap_ratio_max_df_unstack,overlap_ratio_max_df_unstack.T) 
    overlap_ratio = overlap_ratio_max_df_final
    return overlap_ratio




def get_diff_min_uid_pair(value,uid):
    '''
    df: n_fields x [...,key,'uid',...]
    eg get_diff_min_uid_pair(pf_all_field_combined['peak'],pf_all_field_combined.index.get_level_values(1))
    '''
    time_diff = value.values[:,None] - value.values[None,:]
    triu_ind=np.triu_indices_from(time_diff,0)
    time_diff_flatten=np.abs(time_diff[triu_ind])

    n_sw = value.shape[0]
    ind_pair_tensor = np.zeros((n_sw,n_sw,2))
    ind_pair_tensor[:,:,0] = np.tile(uid.values[:,None],[1,n_sw])
    ind_pair_tensor[:,:,1] = np.tile(uid.values[None,:],[n_sw,1])
    ind_pair_tensor_flatten = ind_pair_tensor[triu_ind]

    time_diff_df = np.concatenate([ind_pair_tensor_flatten,time_diff_flatten[:,None]],axis=1)
    time_diff_df = pd.DataFrame(time_diff_df,columns=['uid_0','uid_1','time_diff'])
    time_diff_min_df = time_diff_df.groupby(['uid_0','uid_1'])['time_diff'].min()
    time_diff_min_df_unstack=time_diff_min_df.unstack()
    time_diff_min_df_unstack = time_diff_min_df_unstack.fillna(np.inf)
    time_diff_min_df_final = np.minimum(time_diff_min_df_unstack,time_diff_min_df_unstack.T) # min abs time diff between the switches between two neurons
    time_diff = time_diff_min_df_final
    return time_diff

def get_sim(value,uid,win_l=[2,5,10,60],decay_rate_l=[0.001,0.003,0.005,0.007,0.01,0.05]):
    pass
def get_sw_sim(all_sw_info,diff_key='time',win_l=[1,10,30,60],decay_rate_l=[0.001,0.003,0.005,0.007,0.01,0.05]):
    '''
    get switch similarities

    ======
    sw_sim_allonoff: [onoff,'within_',decay...] x n_neuron; 
        onoff: 1/-1/both; [NB!] both means one neuron ON the other OFF and vice versa, take the min
        within_: whether two neurons have at least one switches within that window
        decay: pass through an exponential decay, the smaller the decay rate, the less distance matters, i.e. everything becomes similar      
    diff_key: can be time, trial_index, field_pos, etc, to allow for different types of window for counting co-switching 
    
    time_diff: min abs time diff between the switches between two neurons; for both, the diagonal items will not be 0!! And some terms will be INF if it never happen that one neuron ON and the other OFF. careful in downstream analysis

    '''
    sim_d_allonoff = {}
    time_diff_allonoff = {}
    for onoff in [1,-1,'both']:
        sim_d = {}
        if onoff=='both':
            all_sw_info_onoff = all_sw_info
            on = all_sw_info.loc[all_sw_info['switch']==1]
            on_inds=pd.MultiIndex.from_frame(on[['trialtype','uid','field_index']])
            n_on = on.shape[0]
            off = all_sw_info.loc[all_sw_info['switch']==-1]
            n_off = off.shape[0]
            off_inds=pd.MultiIndex.from_frame(off[['trialtype','uid','field_index']])
            time_diff = np.abs(on[diff_key].values[:,None] - off[diff_key].values[None,:])
            time_diff_flatten = np.ravel(time_diff)
            ind_pair_tensor = np.zeros((n_on,n_off,2))
            ind_pair_tensor[:,:,0] = np.tile(on['uid'].values[:,None],[1,n_off])
            ind_pair_tensor[:,:,1] = np.tile(off['uid'].values[None,:],[n_on,1])
            ind_pair_tensor_flatten = ind_pair_tensor.reshape(-1,2)


        else:
            all_sw_info_onoff = all_sw_info.loc[all_sw_info['switch']==onoff]

            time_diff = all_sw_info_onoff[diff_key].values[:,None] - all_sw_info_onoff[diff_key].values[None,:]
            triu_ind=np.triu_indices_from(time_diff,0)
            time_diff_flatten=np.abs(time_diff[triu_ind])

            n_sw = all_sw_info_onoff.shape[0]
            ind_pair_tensor = np.zeros((n_sw,n_sw,2))
            ind_pair_tensor[:,:,0] = np.tile(all_sw_info_onoff['uid'].values[:,None],[1,n_sw])
            ind_pair_tensor[:,:,1] = np.tile(all_sw_info_onoff['uid'].values[None,:],[n_sw,1])
            ind_pair_tensor_flatten = ind_pair_tensor[triu_ind]

        time_diff_df = np.concatenate([ind_pair_tensor_flatten,time_diff_flatten[:,None]],axis=1)
        time_diff_df = pd.DataFrame(time_diff_df,columns=['uid_0','uid_1','time_diff'])
        time_diff_min_df = time_diff_df.groupby(['uid_0','uid_1'])['time_diff'].min()
        time_diff_min_df_unstack=time_diff_min_df.unstack()
        if onoff=='both': # need to symmetrize to complete the df
            ind_union = time_diff_min_df_unstack.index.union(time_diff_min_df_unstack.columns)
            time_diff_min_df_unstack=time_diff_min_df_unstack.reindex(index=ind_union,columns=ind_union)
        time_diff_min_df_unstack = time_diff_min_df_unstack.fillna(np.inf)
        time_diff_min_df_final = np.minimum(time_diff_min_df_unstack,time_diff_min_df_unstack.T) # min abs time diff between the switches between two neurons
        time_diff = time_diff_min_df_final
        
        for win in win_l:
            whether_coswitch_within_win = time_diff_min_df_final <= win
            sim_d[f'within_{win}'] = whether_coswitch_within_win.astype(float)
        for decay_rate in decay_rate_l:
            exp_sim =exponential_decay(time_diff_min_df_final,decay_rate=decay_rate)
            sim_d[decay_rate] = exp_sim.astype(float)
        sw_sim_d = pd.concat(sim_d,axis=0)
        sim_d_allonoff[onoff] = sw_sim_d
        time_diff_allonoff[onoff] = time_diff
    sw_sim_allonoff = pd.concat(sim_d_allonoff,axis=0)
    time_diff_allonoff = pd.concat(time_diff_allonoff,axis=0)
    return sw_sim_allonoff, time_diff_allonoff # time_diff need to seperate, because it's distance

def exponential_decay(distance, decay_rate=1.0):
    return np.exp(-decay_rate * distance)

# ====test==== #


def label_difference_grouped_by_coswitch(label_sim_one,co_sw_sim_one):
    label_sim_one = label_sim_one.dropna(axis=0,how='all').dropna(axis=1,how='all')
    co_sw_sim_one = co_sw_sim_one.dropna(axis=0,how='all').dropna(axis=1,how='all')
    c1 = label_sim_one.index
    c2 = co_sw_sim_one.index
    common_ind = c1.intersection(c2)
    label=label_sim_one.loc[common_ind,common_ind].values
    cosw=co_sw_sim_one.loc[common_ind,common_ind].values
    label_flatten=np.ravel(label[np.triu_indices_from(label,1)])
    cosw_flatten=np.ravel(cosw[np.triu_indices_from(cosw,1)])
    diff = np.mean(label_flatten[cosw_flatten==1]) - np.mean(label_flatten[cosw_flatten==0])
    
#     diff = scipy.stats.pearsonr(label_flatten,cosw_flatten)[0]
    return diff, label_flatten, cosw_flatten

# shuffle
def shuffle_test_label_switch_diff_plot(label_sim_one,co_sw_sim_one,sw_sim_allonoff_shuffle,onoff=1,sw_key='within_1',
                                        fig=None,ax=None,doplot=False
                                       ):
    diff_data,label_flatten,cosw_flatten = label_difference_grouped_by_coswitch(label_sim_one,co_sw_sim_one)
    diff_data_sh_l = []
    for ss in sw_sim_allonoff_shuffle:
        cosw_shuffle_one=ss.loc[onoff,sw_key]
        diff_data_sh,label_flatten,cosw_flatten_sh=label_difference_grouped_by_coswitch(label_sim_one,cosw_shuffle_one)
        diff_data_sh_l.append(diff_data_sh)
    diff_data_sh_l = np.array(diff_data_sh_l)
    pval = 1-(diff_data > diff_data_sh_l).mean()
    to_return = [diff_data,diff_data_sh_l, pval]
    if doplot:
        fig,ax=ph.plot_shuffle_data_dist_with_thresh(diff_data_sh_l,diff_data,fig=fig,ax=ax)
        to_return.extend([fig,ax])
    return tuple(to_return)
        

    
#============================interneuron similarity============================#
def get_int_con_sim_oneti(glm_res_df ,type='inh_jaccard'):
    '''
    glm_res_df: tt x uid x field_id
    '''
    if type=='inh_jaccard':
        glm_res_per_uid = glm_res_df.groupby(level=(0,1)).mean()
        int_that_inh=(glm_res_per_uid['coef'] < 0) & (glm_res_per_uid['p'] < 0.05)
        int_that_inh_sim = 1-squareform(pdist(int_that_inh,metric='jaccard'))
        int_that_inh_sim = pd.DataFrame(int_that_inh_sim,index=glm_res_per_uid.index,columns=glm_res_per_uid.index)
        int_con_sim = int_that_inh_sim
    elif type=='all_corr':
        coef = glm_res_df['coef']
        coef_per_uid = coef.groupby(level=(0,1)).mean()
        glmcoef_sim = np.corrcoef(coef_per_uid)
        glmcoef_sim = pd.DataFrame(glmcoef_sim,index=coef_per_uid.index,columns=coef_per_uid.index)
        int_con_sim = glmcoef_sim
    return int_con_sim

def get_overlap_and_co_sw_not_co_sw_inds(all_sw_with_inh_change_one,sw,tt,dist_thresh = 10,trial_dist_thresh=0):
    all_sw_with_inh_change_one_onoff = all_sw_with_inh_change_one.query('switch==@sw&(trialtype==@tt|trialtype=="both")')
    uid_l = all_sw_with_inh_change_one_onoff['uid'].values
    field_dist = all_sw_with_inh_change_one_onoff['field_pos'].values[:,None] - all_sw_with_inh_change_one_onoff['field_pos'].values[None,:]
    field_dist=np.abs(field_dist)
    
    field_dist_binary =field_dist < dist_thresh

    trialtype_match = all_sw_with_inh_change_one_onoff['trialtype'].values[:,None] == all_sw_with_inh_change_one_onoff['trialtype'].values[None,:]
    trialtype_match[(all_sw_with_inh_change_one_onoff['trialtype']=='both').values] = True
    trialtype_match[:,(all_sw_with_inh_change_one_onoff['trialtype']=='both').values] = True

    field_dist_binary = np.logical_and(field_dist_binary,trialtype_match)


    field_dist_binary = np.triu(field_dist_binary,1)
    ind1,ind2=np.nonzero(field_dist_binary)
    overlap_and_sw_ind = np.stack([uid_l[ind1],uid_l[ind2]]).T

    trial_ind_dist =all_sw_with_inh_change_one_onoff['trial_index'].values[:,None] - all_sw_with_inh_change_one_onoff['trial_index'].values[None,:]
    
    trial_ind_dist_binary = np.abs(trial_ind_dist) <=trial_dist_thresh
    trial_ind_dist_binary = np.triu(trial_ind_dist_binary,1)

    overlap_and_co_sw = np.logical_and(field_dist_binary,  trial_ind_dist_binary)
    overla_and_sw_not_co = np.logical_and(field_dist_binary,  ~trial_ind_dist_binary)

    ind1,ind2=np.nonzero(overlap_and_co_sw)
    overlap_and_co_sw_ind = np.stack([uid_l[ind1],uid_l[ind2]]).T

    ind1,ind2=np.nonzero(overla_and_sw_not_co)
    overlap_and_sw_not_co_ind = np.stack([uid_l[ind1],uid_l[ind2]]).T


    return overlap_and_co_sw_ind,overlap_and_sw_not_co_ind
    # overlap_and_co_sw_ind = 



def get_int_con_group_by_overlap_and_cosw(int_con_sim,all_sw_with_inh_change_one,sw=1,dist_thresh = 10,trial_dist_thresh=0):
    has_both = 'both' in int_con_sim.index.get_level_values(0)
    # sw_sim_allonoff, time_diff_allonoff = pwa.get_sw_sim(all_sw_with_inh_change_one,diff_key='field_pos',win_l=[5],decay_rate_l=[])
    int_con_sim_onett_grouped_bothtt={}
    for tt in [0,1]:
        if has_both:
            tt_l = [tt,'both']
        else:
            tt_l = tt

        overlap_and_cosw_ind,overlap_and_sw_not_co_ind=get_overlap_and_co_sw_not_co_sw_inds(all_sw_with_inh_change_one,sw,tt,dist_thresh = dist_thresh,trial_dist_thresh=trial_dist_thresh)
        # sw_sim_onett_peruid = sw_sim_allonoff.loc[sw,'within_5'].dropna(axis=1,how='all')
        
        int_con_sim_onett=int_con_sim.loc[(tt_l,slice(None)),(tt_l,slice(None))]
        int_con_sim_onett=int_con_sim_onett.groupby(level=1,axis=0).mean().groupby(level=1,axis=1).mean()
        
        subdf = intersect_then_index(int_con_sim_onett,overlap_and_cosw_ind[:,0],axis=0)
        int_con_sim_onett_overlap_and_cosw = intersect_then_index(subdf,overlap_and_cosw_ind[:,1],axis=1)

        subdf = intersect_then_index(int_con_sim_onett,overlap_and_sw_not_co_ind[:,0],axis=0)
        int_con_sim_onett_overlap_and_sw_not_co = intersect_then_index(subdf,overlap_and_sw_not_co_ind[:,1],axis=1)

        int_con_sim_onett_grouped=pd.concat({'no_cosw':int_con_sim_onett_overlap_and_sw_not_co.unstack(),
            'cosw':int_con_sim_onett_overlap_and_cosw.unstack()}).reset_index(level=0)
        int_con_sim_onett_grouped.columns=['cosw','int_con_sim']
        int_con_sim_onett_grouped_bothtt[tt] = int_con_sim_onett_grouped
    int_con_sim_onett_grouped_bothtt = pd.concat(int_con_sim_onett_grouped_bothtt,axis=0)
    return int_con_sim_onett_grouped_bothtt

def intersect_then_index(df,ind,axis=0):
    if axis==0:
        ind_ = df.index.intersection(ind)
        subdf=df.loc[ind_]
    else:
        ind_ = df.columns.intersection(ind)
        subdf = df.loc[:,ind_]
    return subdf
        


#========DISTANCE (deprecated)=====#
# import numpy.linalg as LA

# def get_pdist_geo_corr(sim_all_d):
#     '''
#     sim_all_d: df: (... x n_neuron) x n_neuron
#     '''
#     nlevels = len(sim_all_d.index.levels)
#     sim_all_d_gpb =sim_all_d.groupby(level=(list(range(nlevels-1))))
#     corr_d = {}
#     geo_d = {}
#     pval_d = {}
#     for ii,(k, val) in enumerate(sim_all_d_gpb):
#         val = val.loc[k]
#         for jj,(k2,val2) in enumerate(sim_all_d_gpb):
#             if (jj>ii):
#                 val2 = val2.loc[k2]
#                 corr_d[k,k2],pval_d[k,k2] = corr_distance(val,val2)
#                 geo_d[k,k2] = geodesic_distance(val,val2)
#     res={'corr':pd.Series(corr_d),'geo':pd.Series(geo_d)}
#     res = pd.DataFrame(res)
#     return res


# def geodesic_distance(FC1,FC2,eig_thresh=1e-8,epsilon = 1e-8,norm_s=False):
#     '''
#     dist = sqrt(trace(log^2(M)))
#     M = Q_1^{-1/2}*Q_2*Q_1^{-1/2}
#     '''
#     # compute Q_1^{-1/2} via eigen value decmposition
#     if isinstance(FC1,pd.DataFrame):
#         FC1 = FC1.dropna(axis=0,how='all').dropna(axis=1,how='all')
#         FC2 = FC2.dropna(axis=0,how='all').dropna(axis=1,how='all')
#         common_inds = FC1.index.intersection(FC2.index)
#         FC1 = FC1.loc[common_inds,common_inds]
#         FC2 = FC2.loc[common_inds,common_inds]
    
#     u, s, _ = scipy.linalg.svd(FC1, full_matrices=True)

#     u2, s2, _ = scipy.linalg.svd(FC2, full_matrices=True)
#     if norm_s:
#         s = s / np.linalg.norm(s)
#         s2 = s2 / np.linalg.norm(s2)
    
#     # ## lift very small eigen values
#     # for ii, s_ii in enumerate(s):
#     #     if s_ii < eig_thresh:
#     #         s[ii] = eig_thresh
#     ma = (s > eig_thresh) & (s2 > eig_thresh)


#     '''
#     since FC1 is in S+, u = v, u^{-1} = u'
#     FC1 = usu^(-1)
#     FC1^{1/2} = u[s^{1/2}]u'
#     FC1^{-1/2} = u[s^{-1/2}]u'
#     '''
    
    
    
#     FC1_mod = u[:,ma] @ np.diag(s[ma]**(-1/2)) @ np.transpose(u[:,ma])
#     M = FC1_mod @ u2[:,ma] @ np.diag(s2[ma]) @ u2[:,ma].T @ FC1_mod

#     '''
#     trace = sum of eigenvalues;
#     np.logm might have round errors,
#     implement using svd instead
#     '''
# #     _, s, _ = LA.svd(M, full_matrices=True)
    
#     M_regularized = M + np.eye(M.shape[0]) * epsilon
#     _, s, _ = scipy.linalg.svd(M_regularized, full_matrices=True)
#     s = s[ma]

#     return np.sqrt(np.sum(np.log(s)**2))

# def corr_distance(fc1,fc2,type='pearson'):
#     if isinstance(fc1,pd.DataFrame):
#         fc1 = fc1.dropna(axis=0,how='all').dropna(axis=1,how='all')
#         fc2 = fc2.dropna(axis=0,how='all').dropna(axis=1,how='all')
#         common_inds = fc1.index.intersection(fc2.index)
#         fc1 = fc1.loc[common_inds,common_inds].values
#         fc2 = fc2.loc[common_inds,common_inds].values
        
#     corr_func = scipy.stats.pearsonr if type=='pearson' else scipy.stats.spearmanr
#     triu_inds=np.triu_indices_from(fc1,1)
#     r,pval = corr_func(fc1[triu_inds],fc2[triu_inds])
#     return 1-r,pval

# def dropna_intersect_ind_flatten(df1,df2):
#     df1 = df1.dropna(axis=1,how='all').dropna(axis=0,how='all')
#     df2 = df2.dropna(axis=1,how='all').dropna(axis=0,how='all')
#     common_ind = df1.index.intersection(df2.index)
#     dfv1 = df1.loc[common_ind,common_ind]
#     triu_inds= np.triu_indices_from(dfv1,1)
#     dfv1_flatten = dfv1.values[triu_inds]
#     dfv2 = df2.loc[common_ind,common_ind]
#     dfv2_flatten = dfv2.values[triu_inds]
#     return dfv1,dfv2,dfv1_flatten,dfv2_flatten


# def get_sw_label_distance(sw_sim_allonoff,label_sim_all):
#     sw_label_distance_df_onoff = {}
#     for onoff,sw_sim_d in sw_sim_allonoff.groupby(level=0):
#         sw_sim_d=sw_sim_d.loc[onoff]
#         sw_label_distance_df = []
#         for k,sw_sim in sw_sim_d.groupby(level=0):
#             for kk,label_sim in label_sim_all:
#                 geo = geodesic_distance(label_sim.loc[kk],sw_sim.loc[k])
#                 corr,pval = corr_distance(label_sim.loc[kk],sw_sim.loc[k])
#                 kk_names = label_sim.index.names[:-1]
#                 #         sw_label_distance_d[(k,kk)]=pd.Series([geo,corr],index=['geo','corr'])
#                 distance_one=pd.Series([geo,corr,k,*kk],index=['geo','corr','sw_sim_type',*kk_names])
#                 sw_label_distance_df.append(distance_one)
#         sw_label_distance_df = pd.DataFrame(sw_label_distance_df)
#         sw_label_distance_df_onoff[onoff] = sw_label_distance_df
#     sw_label_distance_df_onoff = pd.concat(sw_label_distance_df_onoff,axis=0)
#     return sw_label_distance_df_onoff