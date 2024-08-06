import numpy as np
import scipy
import pandas as pd
import copy,os,sys
import matplotlib.pyplot as plt
import seaborn
from pathlib import Path
import seaborn as sns

import plot_helper
from plot_helper import *

from scipy.ndimage import gaussian_filter1d
import scipy.interpolate
from scipy.interpolate import interp1d

sys.path.append(str(Path(__file__).resolve().parent))

import data_prep_new as dpn
import data_prep_pyn as dpp



###=== firing map====####
def get_fr_map_trial_onegroup(df,cell_columns,gauss_width=2.5,order=['smooth','divide','average'],bin_size=2.2,n_lin_bins=100,**kwargs):
    '''

    return:
    if 'average' in order: fr_map_final: dataframe, ncells x nbins
    if not: np array, ncells x nbins x ntrials
    '''
    columns = cell_columns
    if 'lin_binned' not in df:
        # df['lin_binned'] = pd.cut(df['lin'],bins=n_lin_bins,labels=False,retbins=False)
        df,_ = dpp.add_lin_binned(df,bin_size=bin_size,nbins=n_lin_bins)
        # n_lin_bins  = len(df['lin_binned'].unique())
    n_lin_bins  = int(df['lin_binned'].max()) # assuming its from 0 to n_lin_bins-1

    try:
        dt = np.median(np.diff(df['time']))
    except:
        dt = np.median(np.diff(df.index))
    
    counts_l = {}
    occupancy_l = {}
    if 'speed_thresh' in kwargs.keys():
        # mask = kwargs['mask']
        speed_thresh = kwargs['speed_thresh']
    else:
        speed_thresh = 5
    # mask = np.sqrt(df['vel_x']**2+df['vel_y']**2) > speed_thresh # by default only keep time points with speed above 5
    if 'speed_key' in kwargs.keys(): 
        speed_key = kwargs['speed_key']
    else:
        if (df['speed'] > speed_thresh).mean() < 0.001:
            speed_key = 'v'
        else:
            speed_key = 'speed'
    mask = df[speed_key] > speed_thresh # by default only keep time points with speed above 5
    df = df.loc[mask]

    # groupby trial, get histcount for each trial
    for tt,val in df.groupby('trial'):
        # import pdb
        # pdb.set_trace()
        counts = val.groupby('lin_binned')[columns].sum()
        counts = counts.reindex(range(n_lin_bins)) #making sure all position bins are included!
        counts_l[tt]=counts
        occupancy = val.groupby('lin_binned').count().iloc[:,0].rename()
        occupancy = occupancy.reindex(range(n_lin_bins))
        occupancy_l[tt]=occupancy
    
    counts_l = pd.concat(counts_l.values(),axis=1,keys=counts_l.keys()) # col multi indexed: trial, unit; index: lin_binned; 
    # counts_l = counts_l.swaplevel(axis=1) # now col: unit, trial
    counts_l = counts_l.fillna(0).astype(float)
    occupancy_l = pd.concat(occupancy_l.values(),axis=1,keys=occupancy_l.keys()).fillna(0) # n_lin_binned x n_trials

    ncells = len(cell_columns)
    nbins = counts_l.shape[0]
    counts_l_values = counts_l.values.reshape(nbins,-1,ncells) # npos x ntrials x ncells
    counts_l_values = counts_l_values.swapaxes(0,-1).swapaxes(1,-1) # ncells x npos x ntrials
    occupancy_l_values = occupancy_l.values #npos x ntrials
    bin_columns = occupancy_l.index
    # pdb.set_trace()
    fr_map_trial=np.divide(counts_l_values , occupancy_l_values*dt, where=occupancy_l_values!=0, out=np.zeros_like(counts_l_values)) # ncells x npos x ntrials

    spk_counts_final = copy.copy(counts_l_values)
    pos_counts_final = copy.copy(occupancy_l_values)
    fr_map_final = copy.copy(fr_map_trial)

    # because different trials might have different available position bins, need some reordering
    bin_columns_sorted_ind = np.argsort(bin_columns)
    bin_columns = bin_columns[bin_columns_sorted_ind]
    spk_counts_final = spk_counts_final[:,bin_columns_sorted_ind]
    spk_counts_pre_smth = copy.copy(spk_counts_final)
    pos_counts_final = pos_counts_final[bin_columns_sorted_ind]
    fr_map_trial = fr_map_trial[:,bin_columns_sorted_ind]


    for operation in order:
        if operation=='smooth':
            spk_counts_final = gaussian_filter1d(spk_counts_final,gauss_width,axis=1)
            pos_counts_final = gaussian_filter1d(pos_counts_final,gauss_width,axis=0)
            fr_map_final= gaussian_filter1d(fr_map_final,gauss_width,axis=1)
        elif operation=='divide':
    #         fr_map_final = np.divide(spk_counts_final, (pos_counts_final * dt), where=pos_counts_final!=0)
            fr_map_final = np.divide(spk_counts_final, pos_counts_final*dt, where=pos_counts_final!=0, out=np.zeros_like(spk_counts_final))
    #             fr_map_final = spk_counts_final/ (pos_counts_final * dt)
        elif operation == 'average':
            spk_counts_final = spk_counts_final.mean(axis=-1,keepdims=True)
            pos_counts_final = pos_counts_final.mean(axis=-1,keepdims=True)
            fr_map_final = fr_map_final.mean(axis=-1,keepdims=True)
    try:        
        fr_map_final = np.squeeze(fr_map_final)
        fr_map_final=pd.DataFrame(fr_map_final,index=cell_columns,columns=bin_columns)
    except:
        print('cant format into a dataframe') # when not averaging
    # if 'average' in order:
    return fr_map_final, spk_counts_final, pos_counts_final *dt  # want to return the occupancy in same unit as fr, sec
    # else: # if no average, return the pre smth spike counts for the poisson test
    #     return fr_map_final, spk_counts_pre_smth, pos_counts_final*dt 

def get_fr_map_trial(df,cell_columns,trialtype_key='visitedArm',**kwargs):
    '''
    trialtype_key for subdividing trialtype before computing the tuning curve; 
    eg for alternating it's visitedArm; for linear it's direction
    '''
    gpb = df.groupby([trialtype_key])
    fr_map_final_dict = {}
    for key, val in gpb:
        fr_map_final_dict[key] = get_fr_map_trial_onegroup(val,cell_columns,**kwargs)

    return fr_map_final_dict

def get_fr_map_trial_multi_task(spk_beh_df,cell_columns,trialtype_key_dict = {'alternation':'visitedArm','linearMaze':'direction'}, \
                        **kwargs):
    fr_map_task_dict = {}
    task_index_to_task_name = dpp.get_task_index_to_task_name(spk_beh_df)
    for k, val in spk_beh_df.groupby('task_index'):
        task_type = task_index_to_task_name[k]
        fr_map_turn_dict = get_fr_map_trial(val,cell_columns,trialtype_key=trialtype_key_dict[task_type],**kwargs)
        fr_map_task_dict[k] ={k:v[0] for k,v in fr_map_turn_dict.items()}
    return fr_map_task_dict

def concat_fr_map_trial_df_d(fr_map_trial_df_d,spk_beh_df):
    '''
    for fr_map, concatenate within a task, different trial types, in the correct order of trials
    fr_map_trial_df_d: multiindex (task, trialtype, neuron, field) x trial 
    
    turn into: (task, neuron, field) x trial, and {task:{trialtype:trialindex}}
    '''
    # trial_index_to_index_within_df = pd.concat(dpp.trial_index_to_index_within_trialtype(spk_beh_df),axis=0)
    trial_index_to_index_within_df = dpp.trial_index_to_index_within_trialtype(spk_beh_df)
    index_within_to_trial_index_df = trial_index_to_index_within_df.reset_index(level=2).set_index('index',append=True)['trial_ind'].astype(int)

    gpb = fr_map_trial_df_d.groupby(level=0) #task index
    fr_map_concat = {}
    for k,val in gpb:
        ntrials = trial_index_to_index_within_df.loc[k].shape[0]
        nfields=val.loc[k,0].shape[0]
        mat = np.zeros((nfields,ntrials)) * np.nan
        for kk,valval in val.loc[k].groupby(level=0): # trialtype 
            valval = valval.dropna(axis=1)
            mat[:,index_within_to_trial_index_df.loc[k,kk].loc[valval.columns].astype(int)] = valval.values
        mat = pd.DataFrame(mat,index=valval.loc[kk].index)
        fr_map_concat[k]=mat
    fr_map_concat = pd.concat(fr_map_concat,axis=0)
    
    return fr_map_concat,index_within_to_trial_index_df

#===== spatial information======#
def get_bits_per_spike(spk_beh_df,cell_cols=None,gauss_width = 2.5,speed_key = 'directed_locomotion',speed_thresh = 0.5,
                       trialtype_key='trial_type',
                    ):
    '''
    sum_i p_i*(l_i/l)*log2(l_i/l)
    si_d: series: (task_ind,tt,uid)
    '''
    eps = 1e-10
    fr_map_dict=get_fr_map_trial(spk_beh_df,cell_cols,gauss_width=gauss_width,trialtype_key=trialtype_key,speed_key=speed_key,speed_thresh=speed_thresh,order=['smooth','divide','average'])
    si_d = {}
    for k in fr_map_dict.keys():
        occu = fr_map_dict[k][-1]
        occu_p = np.squeeze(occu / occu.sum())
        fr_map = fr_map_dict[k][0]

        fr_mean = np.average(fr_map,weights=occu_p,axis=1)

        div = (fr_map / fr_mean[:,None])
        si = np.average(div*np.log2(div+eps),weights=occu_p,axis=1)
        si = pd.Series(si,fr_map.index)
        si_d[k] = si
    si_d = pd.concat(si_d,axis=0)
    
    return si_d
    
def get_si_one_field(fr_map,occu):
    eps = 1e-10
    occu_p = np.squeeze(occu / occu.sum())
    # pdb.set_trace()
    common_pos=fr_map.columns.intersection(occu.index)
    fr_map_v = fr_map.loc[:,common_pos].values
    fr_map_v[fr_map_v<0] = 0. # rectify to positive
    fr_mean = np.average(fr_map_v,weights=occu_p,axis=1)
    div = (fr_map_v / fr_mean[:,None])
    si = np.average(div*np.log2(div+eps),weights=occu_p,axis=1)
    si = pd.Series(si,fr_map.index)
    return si

def get_si_from_frmap_and_occu(frmap,occu_map):
    si_all = {}
    for kk, row in occu_map.iterrows():
        frmap_one=frmap.loc[kk].dropna(axis=1)
        row = row.dropna()
        si_one = get_si_one_field(frmap_one,row)
        si_all[kk] = si_one
    si_all = pd.concat(si_all,axis=0)
    return si_all

def get_fr_cv(fr_map_trial_df_all,level_l = [0,1,2,3,4]):
    '''
    fr_map_trial_df_all: (region x exp x day x isnovel x uid x pos) x trial
    '''
    # rectify negative:
    fr_map_trial_df_all_v = fr_map_trial_df_all.values
    fr_map_trial_df_all_v[fr_map_trial_df_all_v<0] = 0.
    fr_map_trial_df_all = pd.DataFrame(fr_map_trial_df_all_v,index=fr_map_trial_df_all.index,columns=fr_map_trial_df_all.columns)
    

    gpb = fr_map_trial_df_all.groupby(level=level_l)
    mean_over_pos = gpb.mean() # position out
    mean_fr = mean_over_pos.mean(axis=1)
    fr_cv = mean_over_pos.std(axis=1) / mean_fr

    return fr_cv, mean_fr


#============Representational Similarity Analysis=========#
def get_pop_rep_corr(sess_name,data_dir,data_dir_full=None,toplot=True,corr=True,n_pos_bins=100):
    '''
    if corr==True: do correlation; else: uncentered covariance
    '''
    # animal_name = sess_name.split('_')[0]


    cell_metrics,behavior,spike_times,uid,fr,cell_type,mergepoints,behav_timestamps,position,\
            rReward,lReward,endDelay,startPoint,visitedArm \
            = dpn.load_sess(sess_name, data_dir=data_dir,data_dir_full=data_dir_full)

    # spike_times,uid,fr,cell_type,mergepoints,behav_timestamps,position,\
    #                 rReward,lReward,endDelay,startPoint,visitedArm \
    #     = dpn.load_sess(sess_name, data_dir=data_dir)

    df_dict, pos_bins_dict,cell_cols_dict = dpn.get_fr_beh_df(spike_times,uid,behav_timestamps,cell_type,position,visitedArm,startPoint,n_pos_bins=n_pos_bins)

    fr_map_final=get_fr_map_trial(df_dict['pyr'],cell_columns=cell_cols_dict['pyr'], n_lin_bins=n_pos_bins)

    fr_map_final_ = {key:val[0].T for key,val in fr_map_final.items()}
    ncells = len(cell_cols_dict['pyr'])
    rep_mat = pd.concat(fr_map_final_.values(),keys=fr_map_final_.keys()).T # ncells x (choice,pos) 
    if corr:
        rep_mat_corr = rep_mat.corr()
    else:
        rep_mat_corr = rep_mat.T.dot(rep_mat)

    
    bin_turn1 = (75>pos_bins_dict['lin']).sum()-1
    bin_turn2 = (112.5>pos_bins_dict['lin']).sum()-1
    bin_turn3 = (187.5>pos_bins_dict['lin']).sum()-1

    ticks_onearm = np.array([0,bin_turn1,bin_turn2,bin_turn3])
    ticks_both = np.concatenate([ticks_onearm,ticks_onearm+n_pos_bins])

    from itertools import product
    tick_labels = list(product(['arm0','arm1'], ['beg','t1','t2','t3']))

    if toplot:
        fig,ax=plt.subplots()
        ax=sns.heatmap(rep_mat_corr,ax=ax)
        ax.set_title(f'{sess_name}, ncells={ncells}')


        ax.set_xticks(ticks_both)
        ax.set_xticklabels(tick_labels)
        ax.set_yticks(ticks_both)
        ax.set_yticklabels(tick_labels)
    
        return fig,ax,rep_mat_corr,(tick_labels,ticks_both)
    else:
        return rep_mat_corr,(tick_labels,ticks_both)


def get_stuff_by_trial(trial_markers,time_stamps,spk_times_all,*args,mask=None):
    '''
    turn trial concatenated data into having a by trial structure
    all variables share a common unit in time (s)
    time_stamps serve as a time measure against which to break things down into trials, can be behav_timestamps
    
    trial_markers: ntrials x 2, 

    spk_times_all: ncells x nspks, list

    '''
#     if mask is not None:
#         pos,time_stamps,spk_times_all
        
#     pos_trial = []
    args_trial = [[] for arg in args] 
    time_stamps_trial = []
    spk_times_trial_all = [[] for cc in range(len(spk_times_all))] 
    for tt,tm in enumerate(trial_markers):
        pos_inds_mask = (time_stamps >=tm[0]) & (time_stamps <tm[1])
        
        # all args that share the same time_stamps
        for kk,arg in enumerate(args): 
#             args_trial[kk]=np.append(args_trial[kk],arg[pos_inds_mask])
            args_trial[kk].append(arg[pos_inds_mask])
        time_stamps_trial.append(time_stamps[pos_inds_mask])
        
        for cc,spk_times in enumerate(spk_times_all):
            spk_times_mask = (spk_times>=tm[0])&(spk_times<time_stamps[pos_inds_mask][-1]) #because sampling freq differ between behav and spike, if exclude end point in behav, there could still be spike times exceeding the behavior timestamp, ccausing problem in interpolation
            spk_times_trial_all[cc].append(spk_times[spk_times_mask])
    args_trial = [np.array(arg_trial,dtype=object) for arg_trial in args_trial] 
#     return np.array(pos_trial,dtype=object), np.array(time_stamps_trial,dtype=object), np.array(spk_times_trial_all,dtype=object)
    return np.array(time_stamps_trial,dtype=object), np.array(spk_times_trial_all,dtype=object), args_trial

def get_spk_triggered_positions(pos_trial, time_stamps_trial, spk_times_trial_all,speedmask=None,speed_trial=None,\
    return_spk_triggered_spe=False, return_speed_masked_spk_times=False):
    '''
    get spike triggered positions; through interpolation
    
    pos_trial, ntrials x ntimeswithin trial; list
    time_stamps_trial, ntrials x ntimeswithintrial; list
    spk_times_trial_all, ncells x ntrials x nspikes within trial; list

    speedmask: either None, or a number
    
    return: depending on the flags:
    spk_triggered_positions_trial_all
    (spk_triggered_positions_trial_all, spk_times_masked)
    (spk_triggered_positions_trial_all, spk_triggered_spe_trial_all)
    (spk_triggered_positions_trial_all, spk_times_masked, spk_triggered_spe_trial_all)
    
    '''
    ntrials = len(pos_trial)
    ncells = len(spk_times_trial_all)
    spk_triggered_positions_trial_all = []
    spk_triggered_spe_trial_all = []
    spk_times_masked = []
    if speedmask is not None:
        if speed_trial is None:
            raise NameError('no speed_trial')
        else:
            mask_by_speed = True
    else:
        mask_by_speed = False
    for cc in range(ncells): # cell loop
        spk_times_trial = spk_times_trial_all[cc]
        spk_times_masked_trial = []
        spk_triggered_positions_trial = []
        spk_triggered_spe_trial = []
        for tt in range(ntrials): # trial loop
            f = interp1d(time_stamps_trial[tt],pos_trial[tt],axis=0)

            spk_times = spk_times_trial[tt]

            if mask_by_speed:
                spe=speed_trial[tt]
                f_spe = interp1d(time_stamps_trial[tt],spe,axis=0)
                spk_triggered_spe = f_spe(spk_times)
                spk_speed_mask = spk_triggered_spe > speedmask
                spk_times = spk_times[spk_speed_mask]
                spk_triggered_spe_trial.append(spk_triggered_spe)
                spk_times_masked_trial.append(spk_times)

            spk_triggered_pos = f(spk_times)
            spk_triggered_positions_trial.append(spk_triggered_pos)

        if mask_by_speed:
            spk_times_masked.append(np.array(spk_times_masked_trial,dtype=object))

        spk_triggered_spe_trial_all.append(np.array(spk_triggered_spe_trial,dtype=object))

        spk_triggered_positions_trial_all.append(np.array(spk_triggered_positions_trial,dtype=object))

    to_return = [spk_triggered_positions_trial_all]

    if return_speed_masked_spk_times:
        to_return.append(spk_times_masked)

    if return_spk_triggered_spe:
        try:
            to_return.append(spk_triggered_spe_trial_all)
        except:
            print('spk triggered spe not computed')
    if len(to_return)==1:
        to_return = to_return[0] # to make it consistent with the default to_return as before

    return to_return

    
def plot_spikes_on_lin(spk_triggered_positions_trial,**kwargs):
    '''
    spk_triggered_positions_trial: ntrials x ntimepoints, list, for one neuron
    one element of spk_triggered_positions_trial_all
    '''
    scatter_kwargs={'color':'r','s':0.5}
    if 'scatter_kwargs' in kwargs.keys():
        scatter_kwargs.update(kwargs['scatter_kwargs'])
    if 'figsize' in kwargs.keys():
        figsize=kwargs['figsize']
    else:
        figsize=(6,4)
    if 'ax' not in kwargs.keys():
        fig,ax = plt.subplots(figsize=figsize)
    else:
        fig=kwargs['fig']
        ax=kwargs['ax']
    
    ## new implementation: concatenate all (trial_index, spike_position) pairs; easier to filter by spike_position across all trials and assign different colors
    trial_spk_pair_l=[]
    for (trial,spk_trial) in enumerate(spk_triggered_positions_trial):
        trial_col=np.ones(len(spk_trial))*trial
        trial_spk_pair=np.concatenate([spk_trial[:,None],trial_col[:,None]],axis=1)
        trial_spk_pair_l.append(trial_spk_pair)
    trial_spk_pair_l=np.concatenate(trial_spk_pair_l,axis=0)

    ax.scatter(trial_spk_pair_l[:,0],trial_spk_pair_l[:,1],**scatter_kwargs)
    ## old implementation: loop over trial
    # for tt,spk_triggered_positions in enumerate(spk_triggered_positions_trial):
    #     ys=np.ones_like(spk_triggered_positions) * (tt+1)
    #     ax.scatter(spk_triggered_positions,ys,**scatter_kwargs)

    return fig,ax


def get_spk_triggered_positions_speedmasked_directly(sess_name,data_dir_full=None):
    DATABASE_LOC = '/mnt/home/szheng/ceph/ad/database.csv'
    db = pd.read_csv(DATABASE_LOC,index_col=[0,1])
    if data_dir_full is None:
        data_dir_full = db.query(f'sess_name=="{sess_name}"').loc[:,'data_dir_full'].iloc[0]
    cell_metric,behavior,spike_times,uid,fr,cell_type,mergepoints,behav_timestamps,position,\
                rReward,lReward,endDelay,startPoint,visitedArm \
    = dpn.load_sess(sess_name=sess_name, data_dir=None, data_dir_full=data_dir_full)

    pos_2d = np.stack([behavior['position']['x'],behavior['position']['y']],axis=1)
    dt = behav_timestamps[2]-behav_timestamps[1]
    speed,_ = dpn.smooth_get_speed(pos_2d,dt,sigma=30)
    trial_ints = behavior['trials']['startPoint']
    mask=np.isnan(trial_ints)
    mask = ~np.logical_or(mask[:,0],mask[:,1])
    trial_ints = trial_ints[mask]
    choice=behavior['trials']['visitedArm'][mask]
    pos=behavior['position']['lin']
    trial_markers =trial_ints
    spk_times_all=cell_metric['spikes']['times']
    time_stamps_trial, spk_times_trial_all, [pos_trial,speed_trial]=\
    get_stuff_by_trial(trial_markers,behav_timestamps,spk_times_all,pos,speed)
    spk_triggered_positions_trial_all_speedmasked = get_spk_triggered_positions(pos_trial,time_stamps_trial,spk_times_trial_all,speedmask=5,speed_trial=speed_trial)
    return spk_triggered_positions_trial_all_speedmasked

###====place field test===###
def circular_shift_one(times,shift,bounds):
    times = times + shift
    exceeded_mask = times > bounds[1]
    times[exceeded_mask] = bounds[0] + times[exceeded_mask] - bounds[1]
    return times
def circular_shift_spike_times(spike_times,shift_l,bounds):
    # shift_l: iterable; trying to make shifts different for different neurons
    spkt_shifted = np.array([circular_shift_one(times,shift,bounds) for (times,shift) in zip(spike_times,shift_l)],dtype=object)
    return spkt_shifted

def get_speed_masked_spike_times(spike_times,trial_markers,time_stamps,behavior,sigma=30,speedmask=1):
    pos_2d = np.stack([behavior['position']['x'],behavior['position']['y']],axis=1)
    dt = time_stamps[2]-time_stamps[1]
    speed,_ = dpn.smooth_get_speed(pos_2d,dt,sigma=sigma)
    time_stamps_trial, spk_times_trial_all, [pos_trial,speed_trial]=get_stuff_by_trial(trial_markers,time_stamps,spike_times,pos_2d,speed)
    _,spike_times_masked=get_spk_triggered_positions(pos_trial, time_stamps_trial, spk_times_trial_all,speedmask=speedmask,speed_trial=speed_trial,\
        return_spk_triggered_spe=False, return_speed_masked_spk_times=True)
    return spike_times_masked

def get_shuffle_fr_map(sess_name,data_dir_full,N_shifts=100,dosave=False,speedmask=5):
    
    # shift the whole spike train; perhaps should only do behavior time? but perhaps shouldn't matter
    
    cell_metric,behavior,spike_times,uid,fr,cell_type,mergepoints,behav_timestamps,position,\
                rReward,lReward,endDelay,startPoint,visitedArm \
    = dpn.load_sess(sess_name=sess_name, data_dir=None, data_dir_full=data_dir_full)

    # get the behavior section
    behav_st_ed = np.array([[behav_timestamps[0],behav_timestamps[-1]]]) # 1x2, beg and end of the behavior section
    
    spike_times = get_speed_masked_spike_times(spike_times,behav_st_ed,behav_timestamps,behavior,sigma=30,speedmask=speedmask)
    
    spike_times = np.array([a[0] for a in spike_times],dtype=object)

    min_shift_sec = 5.0
    max_shift_sec = behav_st_ed[0,1] - min_shift_sec
    n_neurons = len(spike_times)
    shifts_mat = min_shift_sec + np.random.rand(N_shifts,n_neurons) * (max_shift_sec- min_shift_sec - min_shift_sec) #-2 min_shift_sec because it's circular

    n_pos_bins=100

    shifted_fr_map_l_dict = {0:[],1:[]}
    for shift_l in shifts_mat:
        spike_times_shifted = circular_shift_spike_times(spike_times,shift_l,behav_st_ed[0])


        df_dict, pos_bins_dict,cell_cols_dict = dpn.get_fr_beh_df(spike_times_shifted,uid,behav_timestamps,cell_type,position,visitedArm,startPoint,n_pos_bins=n_pos_bins)

        df_all = pd.concat([df_dict['pyr'],df_dict['int']],axis=1)
        df_all = df_all.loc[:,~df_all.columns.duplicated()]
        cols_all = cell_cols_dict['pyr'] + cell_cols_dict['int']

        fr_map_dict = get_fr_map_trial(df_all,cols_all,gauss_width=2.5,order=['smooth','divide','average'],n_lin_bins=n_pos_bins,speed_thresh=speedmask)
        shifted_fr_map_l_dict[0].append(fr_map_dict[0][0])
        shifted_fr_map_l_dict[1].append(fr_map_dict[1][0])
    res={}
    for k,val in shifted_fr_map_l_dict.items():
        res[str(k)]=np.stack([v.values for v in val])
    unit_names = shifted_fr_map_l_dict[0][0].index
    
    if dosave:
        save_data_dir = os.path.join(data_dir_full,'py_data')
        if not os.path.exists(save_data_dir):
            os.mkdir(save_data_dir)
            print(f'{save_data_dir} made!')
        fn_full = os.path.join(save_data_dir,'shifted_fr_map.npz')
        np.savez(fn_full,**res,unit_names=unit_names)
        print(f'{fn_full} saved!')

    return res,unit_names

# stimulation modulation
def get_self_similarity_across_stim(fr_map_d):
    '''
    fr_map_d: dict / pd.df, keys: (task_ind,trial_type_ind, stim,...): df: n_neuron x n_pos

    self_similarity_d: df: 
    '''
    if isinstance(fr_map_d,dict):
        fr_map_d_concat = pd.concat(fr_map_d)
    else:
        fr_map_d_concat = fr_map_d

    all_corr = fr_map_d_concat.T.corr() # (groupping conditions x neuron) x (groupping conditions x neuron)
    gpb = all_corr.groupby(level=(0,1))
    self_similarity_d = {}
    for k,val in gpb:
        diags = np.diag(val.loc[(*k,0,slice(None)),(*k,1,slice(None))]) # stim = 0 or 1 for a trial
        self_similarity = pd.Series(diags,index=val.loc[(*k,0),:].index)
        self_similarity_d[k] = self_similarity
    self_similarity_d = pd.concat(self_similarity_d,axis=1)
    return self_similarity_d