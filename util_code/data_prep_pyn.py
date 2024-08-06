# preprocess data into dataframes with fr and behavior
# aim to accommodate different people's idiosyncracies in the raw data

import numpy as np
import scipy,os,copy,pickle,sys,itertools
from importlib import reload
import pandas as pd
from scipy.interpolate import interp1d

from scipy.ndimage import gaussian_filter1d
from plot_helper import *
import pynapple as nap
import data_prep_new as dpn
reload(dpn)
sys.path.append('/mnt/home/szheng/projects/seq_detection2/code')
sys.path.append('/mnt/home/szheng/projects/place_variability/code')
import preprocess as prep # this one is from seq_detection2
# import pykalman

import one_euro_filter as oef

import place_cell_analysis as pa
import behavior_analysis as ba

TRIALTYPE_KEY_DICT = {'alternation':'visitedArm','linearMaze':'direction'}
get_task_index_to_task_name = lambda df:dict(df.groupby('task_index')['task'].first()) # get a dict, keys are task indices, values are task names

def index_one_tsd_using_another_t(tsd,t):
    '''
    t: array like of the timestamps, eg behavior['timestamps']
    tsd: tsd, or df with timestamps as the index
    ====
    ilocs: the numerical index within tsd corresponding to the timestamps queried by t
    '''
    ilocs = tsd.index.get_indexer(t,method='nearest')
    return ilocs


########speed using butterworth#######
from scipy.signal import butter, filtfilt
# Define the low-pass Butterworth filter
def butter_lowpass(cutoff, fs, order=1):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# Apply the forward and reverse Butterworth filter (zero-phase filtering)
def butter_lowpass_filtfilt(data, cutoff, fs, order=1):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def get_speed_butter(xy,dt=0.033,order=1,cutoff=1.):
    '''
    use butterworth filter to get speed
    '''
    fs = 1/dt
    v_xy_butter = []
    for i in range(2):
        v_one_raw = np.gradient(xy[:,i]) / dt
        v_one_butter = butter_lowpass_filtfilt(v_one_raw,cutoff,fs,order=order)
        v_xy_butter.append(v_one_butter)
    v_xy_butter = np.stack(v_xy_butter,axis=1)
    # speed_xy_butter =np.sqrt(v_xy_butter[:,0]**2 + v_xy_butter[:,1]**2)
    return v_xy_butter

def get_speed_gaussian(xy,dt=0.033,win_size=0.3):
    nwin = int(win_size/dt)
    v_xy_gauss = []
    for i in range(2):
        v_one_raw = np.gradient(xy[:,i]) / dt
        v_one_gauss = gaussian_filter1d(v_one_raw,nwin)
        v_xy_gauss.append(v_one_gauss)
    v_xy_gauss = np.stack(v_xy_gauss,axis=1)
    
    return v_xy_gauss

# from pykalman import KalmanFilter
# def get_speed_kalman(xy,dt=0.033,order=2,transition_cov_scale=0.1):
#     '''
#     xy: n_times x 2
#     xy_smth: n_times x 4(order==1)/ 6(order==2), for x,y,vx,vy,ax,ay
#     '''
#     # order =2
#     # dt =np.median(np.diff(beh_df.query("task_index==0").index)) 
#     if isinstance(xy,pd.DataFrame):
#         xy = xy.values
#     R = np.eye(2) * 0.0001
#     if order==1:
#         transition_matrix = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
#         Q = transition_cov_scale * np.eye(transition_matrix.shape[0])    
#         ob_mat = np.array([[1,0,0,0],[0,1,0,0]])
#         initial_state_mean = np.concatenate([xy[0,:],[0,0]])
#     elif order==2:
#         transition_matrix = np.array([[1,0,dt,0,1/2*dt**2,0],[0,1,0,dt,0,1/2*dt**2],[0,0,1,0,dt,0],[0,0,0,1,0,dt],[0,0,0,0,1,0],[0,0,0,0,0,1]])
#         Q = transition_cov_scale * np.eye(transition_matrix.shape[0])    
#         ob_mat = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0]])
#         initial_state_mean = np.concatenate([xy[0,:],[0,0,0,0]])
        
#     kf = KalmanFilter(transition_matrices=transition_matrix,initial_state_mean=initial_state_mean,
#                     observation_matrices=ob_mat,transition_covariance=Q,observation_covariance=R,
#                     n_dim_obs=2)
#     xy_smth,_ = kf.filter(xy)

#     return xy_smth

def get_multi_maze_behavior_df(behavior,transition_cov_scale=0.1,**kwargs):
    '''
    behavior: loaded from the matlab struct: Behavior.mat
    if multiple mazes, then return a dict of df
    first based on Roman's behavior struct; if sth is missing then add

    kwargs: like sessionPulses=sessionPulses, 
    extra loaded structs, need to write specific ways to incorporate them into the df 
    '''
    
    
    pddf = pd.DataFrame
    pos = np.stack(list(behavior['position'].values()),axis=1)
    beh_df = nap.TsdFrame(t = behavior['timestamps'], 
                d = pos,
    #              x = behavior['position']['x'],
    #              y = behavior['position']['y'],
                columns = behavior['position'].keys()
                )
    beh_df['trial']=behavior['masks']['trials'] - 1 # to make trials 0 indexed; originally 1 indexed

    # assign task !! 
    ## (alternatively this could be assigned using behavior.masks.recording)
    time_support = beh_df.time_support
    try:
        iter(behavior['events']['recordings']) # using iter to check if there are more than one tasks
        task_junctures = [*list(behavior['events']['recordings']),time_support.loc[0,'end']]
        multi_task=True
    except TypeError as te:
        task_junctures = [behavior['events']['recordings'],time_support.loc[0,'end']]
        multi_task=False
    # dt = np.median(np.diff(beh_df.index)) # have to use median!!! Mean can be biased by big jumps when there are multiple recordings
    intervals = [nap.IntervalSet(start=[task_junctures[i]],end=[task_junctures[i+1]]) for i in range(len(task_junctures)-1)]  # -dt to make trial duration consistent with the mask
            
    task_index_l = []
    task_l = []
    for ii,interval in enumerate(intervals):
        
        subdf = pd.DataFrame(beh_df.restrict(interval)) # get rid of tsdframe...
        subdf['task'] = behavior['description'][ii] if multi_task else behavior['description']
        subdf['task_index'] = ii
        task_index_l.extend(subdf['task_index'].values)
        # pdb.set_trace()
        task_l.extend(subdf['task'].values)
    beh_df['task_index'] = task_index_l
    beh_df['task'] = task_l
    beh_df = pd.DataFrame(beh_df) # get rid of tsdframe

    # add in smoothed speed

    gpb = beh_df.groupby('task_index')
    task_index_to_task_name=get_task_index_to_task_name(beh_df)
    for k,val in gpb:
        dt = np.median(np.diff(val.index)) # sneaky!! different tasks might have different dt!!
        if 'linear' in task_index_to_task_name[k]:
            pos = val.loc[:,'lin']
            is2d = False
            val['v_raw'] = np.gradient(val['lin']) / dt
            val['speed_raw'] = np.abs(val['v_raw'])
            beh_df.loc[val.index,'v_raw'] = val['v_raw']
            beh_df.loc[val.index,'speed_raw'] = val['speed_raw']
            if val.loc[:,['x','y']].max().max()<1:
                beh_df.loc[val.index,'x'] = val['x'] * 100
                beh_df.loc[val.index,'y'] = val['y'] * 100
        else:
            
            pos = val.loc[:,['x','y']]
            is2d=True
            # quicka and dirty fix !!! Sometimes the x y of maze super small, need to magnify
            if pos.max().max()<1:
                beh_df.loc[val.index,'x'] = val['x'] * 100
                beh_df.loc[val.index,'y'] = val['y'] * 100
                pos = pos * 100
            beh_df.loc[val.index,'vx_raw'] = np.gradient(beh_df.loc[val.index,'x']) / dt
            beh_df.loc[val.index,'vy_raw'] = np.gradient(beh_df.loc[val.index,'y']) /dt
            beh_df.loc[val.index,'speed_raw'] = np.sqrt(beh_df.loc[val.index,'vx_raw']**2 + beh_df.loc[val.index,'vy_raw']**2)
        
        
        # speed,vel_2d = dpn.smooth_get_speed(pos,dt,sigma=30,is2d=is2d)
        # beh_df.loc[val.index,'speed'] = speed
        # beh_df.loc[val.index,'vel_x'] = vel_2d[:,0]
        # beh_df.loc[val.index,'vel_y'] = vel_2d[:,1]

        if not is2d:
            xy = np.stack([pos.values,np.zeros_like(pos.values)],axis=1)
        else:
            xy = pos.values
        # xy_smth = get_speed_kalman(xy,dt=dt,order=2,transition_cov_scale=transition_cov_scale)
        timestamps = val.index

        min_cutoff = kwargs.get('min_cutoff',0.05)
        beta = kwargs.get('beta',0.2)
        d_cutoff = kwargs.get('d_cutoff',1.)

        # xy_smth, v_oef = oef.get_speed_one_euro_filter(xy,timestamps,min_cutoff=min_cutoff,beta=beta,d_cutoff=d_cutoff)
        v_xy = get_speed_gaussian(xy,dt=dt,win_size=0.3)

        if is2d:
            # speed_kalman = np.sqrt(xy_smth[:,2]**2+xy_smth[:,3]**2)
            # beh_df.loc[val.index,'speed_kalman'] = speed_kalman
            # beh_df.loc[val.index,'vx_kalman'] = xy_smth[:,2]
            # beh_df.loc[val.index,'vy_kalman'] = xy_smth[:,3]
            # beh_df.loc[val.index,'ax_kalman'] = xy_smth[:,4]
            # beh_df.loc[val.index,'ay_kalman'] = xy_smth[:,5]
            speed_gauss = np.sqrt(v_xy[:,0]**2+v_xy[:,1]**2)
            beh_df.loc[val.index,'vx_gauss'] = v_xy[:,0]
            beh_df.loc[val.index,'vy_gauss'] = v_xy[:,1]
            beh_df.loc[val.index,'speed_gauss'] = speed_gauss

        else:
            # speed_kalman = np.abs(xy_smth[:,2])
            # v_kalman = xy_smth[:,2]
            # beh_df.loc[val.index,'speed_kalman'] = speed_kalman
            # beh_df.loc[val.index,'v_kalman'] = v_kalman
            
            ## lin for oef
            speed_gauss = np.abs(v_xy[:,0])
            beh_df.loc[val.index,'speed_gauss'] = speed_gauss
            beh_df.loc[val.index,'v_gauss'] = v_xy[:,0]


    # specific behaviors, like turn, correctness, etc. Maze dependent
    #### actually arm field does not seem the right field
    # if 'arm' in behavior['masks'].keys(): # depending on the fact that alternation task would have a arm mask 
    #     sub_df = beh_df.query('task=="alternation"')
    #     ntimepoints = sub_df.shape[0]
    #     beh_df.loc[sub_df.index,'visitedArm'] = behavior['masks']['arm'][:ntimepoints] # to make sure the length is consistent
    # visitedArm!
    if 'visitedArm' in behavior['trials'].keys():
        sub_df = beh_df.query('task=="alternation"')
        info_key = 'visitedArm'
        sub_df = add_trial_specific_info(sub_df,info_key,behavior['trials']['visitedArm'],recordings_l=behavior['trials']['recordings'])
        beh_df.loc[sub_df.index,info_key] = sub_df[info_key]
        # gpb = sub_df.groupby('trial')
        # for k,val in gpb:
        #     if np.isnan(k):
        #         beh_df.loc[val.index,'visitedArm'] = np.nan
        #     else:
        #         beh_df.loc[val.index,'visitedArm'] = behavior['trials']['visitedArm'][int(k)]  # trial index should be 0-indexed!
            
    if 'direction' in behavior['masks'].keys(): # depending on all tasks having a direction mask
        beh_df['direction'] = behavior['masks']['direction']
        
    # assign correct:
    gpb = beh_df.groupby('task_index')
    correct_by_trial=  behavior['trials']['choice']
    pointer_in_reward= 0 
    for k,val in gpb:
        # ntrials_within_task = pddf(val['trial']).nunique(dropna=False).values[0]
        ntrials_within_task = pddf(val['trial']).nunique(dropna=True).values[0] # assuming when nan is in the trial_mask, it doesn't correspond to anything in choice
        correct_slice = correct_by_trial[pointer_in_reward:pointer_in_reward+ntrials_within_task]
        val_gpb = val.groupby('trial')
        
        for kk, valval in val_gpb:
            # if np.isnan(kk):
            #     valval['correct'] = np.nan
    #         valval['correct'] = correct_slice[int(kk)] 
            beh_df.loc[valval.index,'correct'] = correct_slice[int(kk)]
        
    # add in the v that roman computed using Kalman filter
    if 'position_trcat' in behavior['trials'].keys():
        # promote the position_trcat to list such that for loop can be applied
        if not isinstance(behavior['trials']['position_trcat'],np.ndarray):
            position_trcat_l=[behavior['trials']['position_trcat']]
        else:
            position_trcat_l = behavior['trials']['position_trcat']
        beh_df['v'] = pd.Series([],dtype=np.float64)
        for ptr in position_trcat_l:
            ilocs = index_one_tsd_using_another_t(beh_df,ptr['timestamps'])
            beh_df.iloc[ilocs,-1] = ptr['v']
    
    ########## get theta phase and amplitude
    if 'filtered' in kwargs.keys():
        filtered = kwargs['filtered']
        beh_df = add_lfp_filtered(beh_df,filtered,lfp_band='theta')

    ########## for ipshita's data, i.e. with stim during some trials
    if 'sessionPulses' in kwargs.keys():
        sessionPulses = kwargs['sessionPulses']
        info_val_l = [val['stim'] for val in sessionPulses.values()]
        beh_df = add_trial_specific_info(beh_df,'stim_trial',info_val_l)
    
    _,beh_df = group_into_trialtype(beh_df)
    
    
    ########## add signed v 
    if "alternation" in beh_df['task'].values:
        
        beh_df_sub = beh_df.query('task=="alternation"')
        beh_df_sub = get_v_kalman_and_aligned(beh_df_sub)
        # beh_df.loc[beh_df_sub.index,'v_kalman'] = beh_df_sub['v_kalman']
        # beh_df.loc[beh_df_sub.index,'v_kalman_aligned'] = beh_df_sub['v_kalman_aligned']
        beh_df.loc[beh_df_sub.index,'v_gauss'] = beh_df_sub['v_gauss']
        beh_df.loc[beh_df_sub.index,'segment'] = beh_df_sub['segment']

    if 'linearMaze' in beh_df['task'].values:
        beh_df_sub = beh_df.query('task.str.contains("linear")')
        beh_df_sub = get_v_kalman_linear(beh_df_sub)
        # beh_df.loc[beh_df_sub.index,'v_kalman'] = beh_df_sub['v_kalman']
        # beh_df.loc[beh_df_sub.index,'v_kalman_aligned'] = beh_df_sub['v_kalman_aligned']
        beh_df.loc[beh_df_sub.index,'v_gauss'] = beh_df_sub['v_gauss']

    beh_df = beh_df.reset_index().rename({'Time (s)':'time'},axis=1)
    
    ################ add behavior labels (highly subject to modifications)###################

    beh_df = ba.detect_offtrack_event(beh_df,find_turns_kws={},off_track_thresh = 4.,
            on_track_thresh = 1,edges_merge_time=0.4,st_ed_dist_thresh = 20.)


    beh_df = ba.detect_speed_related_event(beh_df,exclude_key_l=['off_track_event'],
                            speed_key='speed_gauss',speed_thresh=1.,
                            compare_type='<=',
                            edges_merge_time = None,
                            event_key = 'pause_event',
                            )

    beh_df = ba.detect_speed_related_event(beh_df,exclude_key_l=['off_track_event','pause_event'],
                            speed_key='v_gauss',speed_thresh=1.,
                            compare_type='>',
                            edges_merge_time = None,
                            event_key = 'directed_locomotion',
                            )

    beh_df = ba.detect_speed_related_event(beh_df,exclude_key_l=['off_track_event','pause_event','directed_locomotion'],
                            speed_key='speed_gauss',speed_thresh=[1,5],
                            compare_type='between',
                            edges_merge_time = None,
                            event_key = 'low_speed',
                            )


    return beh_df        

def add_lfp_filtered(df,filtered,lfp_band='theta'):
    '''
    add lfp information to spk_beh_df
    careful about multiple recordings!!! NEED to recheck if things work. 
    '''
    for lfp_key in ['phase','amp','data']:
        func = scipy.interpolate.interp1d(filtered['timestamps'],filtered[lfp_key])
        interp_result = df.groupby('task_index').apply(lambda x:func(x.index))
        info_key = lfp_band +'_'+lfp_key
        for task_ind,val in interp_result.iteritems():
            df.loc[df['task_index']==task_ind,info_key] = val
    return df

        

def get_spike_trains(cell_metrics):
    spike_trains = nap.TsGroup({int(uid): nap.Ts(t = np.array(times)) for uid, times in zip(cell_metrics['UID'],cell_metrics['spikes']['times'])})
    ispyr = np.array(['pyr' in ct.lower() for ct in cell_metrics['putativeCellType']])
    isint = np.array(['int' in ct.lower() for ct in cell_metrics['putativeCellType']])
    spike_trains.set_info(celltype=np.array(cell_metrics['putativeCellType']),ispyr=ispyr,isint=isint)
    UID = cell_metrics['UID']
    cell_type_mask_d = {'pyr':ispyr,'int':'isint','all':np.ones(len(UID),dtype=bool)}
    cell_cols_d = {'pyr':UID[ispyr],'int':UID[isint],'all':UID}
    return spike_trains, cell_type_mask_d, cell_cols_d

def get_binned_spk(spike_trains,bins):
    '''
    if bins is int could just use .count from nap, but we need exact edges
    ===
    binned_spk: nbins x nneurons
    '''
    binned_spk = np.array([np.histogram(spk.times(),bins=bins)[0] for spk in spike_trains.values()]).T
    return binned_spk

def get_spike_behavior_df(spike_trains,beh_df):
    '''
    using results from get_multi_maze_behavior_df() and get_spike_trains()
    '''
    # dt = np.median(np.diff(beh_df.index))
    spike_trains_l = []
    for k,val in beh_df.groupby('task_index'):
        # ep = nap.IntervalSet(start=[val.index[0]-dt/2],end=[val.index[-1] + dt/2]) # because after the .count, the index will be the centers of each bin.
        timestamps = val['time'].values
        # dt = np.median(np.diff(val.index))
        dt = np.median(np.diff(timestamps)) # now index no longer time
        edges = pd.Series(timestamps).rolling(2).mean().values # create edges at the centers of the behavior.timestamps, with the extra start and end
        edges[0] = timestamps[0]-dt/2
        edges = np.append(edges,timestamps[-1] + dt/2)
        # spike_trains_binned = spike_trains.count(dt,ep=ep)
        spike_trains_binned = get_binned_spk(spike_trains,edges)
        spike_trains_binned = pd.DataFrame(spike_trains_binned,index=val.index,columns=spike_trains.keys())
        spike_trains_l.append(spike_trains_binned)
    spike_trains_df = pd.concat(spike_trains_l,axis=0)
    spk_beh_df = pd.merge_asof(spike_trains_df,beh_df,left_index=True,right_index=True,direction='nearest')

    spk_beh_df = spk_beh_df.query('trial.notna()') # trial==nan usually the initial junk time, can be safely discarded
    return spk_beh_df

def add_trial_specific_info(df,info_key,info_val_l,recordings_l=None):
    '''
    add trial specific info to spk_beh_df
    df: spk_beh_df or beh_df, ntimes x nfeatures
    info_val_l: list of arrays, each element of the list containing (ntrials,) for the corresponding recording session

    if recordings_l is not None, then:
        info_val_l: n_tot_trials, , i.e. concatenated
        recordings_l: n_tot_trials, from behavior['trials]['recordings'], index of 
        which recording the current trial belongs to
        info_val_l will be transformed into a list of arrays like above
    '''
    gpb = df.groupby(['task_index','trial'])
    if recordings_l is not None:
        info_val_l_new =[]
        for c in np.unique(recordings_l):
            ma = recordings_l==c
            info_val_l_new.append(info_val_l[ma])
    else:
        info_val_l_new = info_val_l
    for k,val in gpb:
        df.loc[val.index,info_key] = info_val_l_new[int(k[0])][int(k[1])]
    return df

def prep_roman_data(behavior,cell_metrics,**kwargs):
    '''
    kwargs go to get_multi_maze_behavior_df
    '''
    
    beh_df = get_multi_maze_behavior_df(behavior,**kwargs)
    spike_trains,cell_type_mask_d, cell_cols_d = get_spike_trains(cell_metrics)
    spk_beh_df = get_spike_behavior_df(spike_trains,beh_df)
    task_index_to_task_name = get_task_index_to_task_name(spk_beh_df)
    res = {'beh_df':beh_df,'spike_trains':spike_trains,
        'cell_type_mask_d':cell_type_mask_d,
        'cell_cols_d':cell_cols_d,
        'spk_beh_df':spk_beh_df,
        'task_index_to_task_name':task_index_to_task_name
    }
    return res

def prep_ipshita_data(behavior,cell_metrics,**kwargs):
    beh_df = get_multi_maze_behavior_df(behavior,**kwargs)
    spike_trains,cell_type_mask_d, cell_cols_d = get_spike_trains(cell_metrics)
    cell_cols_d = get_cell_cols_d_with_brainregion(cell_metrics)
    spk_beh_df = get_spike_behavior_df(spike_trains,beh_df)
    sessionPulses = kwargs['sessionPulses']
    spk_beh_df =add_stim(spk_beh_df,sessionPulses,stim_key='stim')
    task_index_to_task_name = get_task_index_to_task_name(spk_beh_df)
    res = {'beh_df':beh_df,'spike_trains':spike_trains,
        'cell_type_mask_d':cell_type_mask_d,
        'cell_cols_d':cell_cols_d,
        'spk_beh_df':spk_beh_df,
        'task_index_to_task_name':task_index_to_task_name
    }
    return res


    


def group_into_trialtype(beh_df,additional_groupers=None):
    '''
    divide input data into trialtypes (each trialtype uses one ratemap, eg an arm of a t maze or a direction of a linear maze)
    beh_df: can be beh_df or spk_beh_df or any other version that contains the trial and trialtype information in time
    notice this will exclude the waiting part!!!!! So to do ripple analysis need to expand this to include the non running part

    eg additional_groupers=['stim']
    '''
    beh_df = pd.DataFrame(beh_df)
    res_d = {}
    task_index_to_task_name = get_task_index_to_task_name(beh_df)
    trialtype_key_d = {}
    for task_type, beh_df_task in beh_df.groupby('task_index'):
        trialtype_key=TRIALTYPE_KEY_DICT[task_index_to_task_name[task_type]] # visitedArm or direction
        if additional_groupers is None:
            for trial_type, beh_df_task_trialtype in beh_df_task.groupby(trialtype_key):
                res_d[(task_type,trial_type)] = beh_df_task_trialtype
        else:
            for trial_type, beh_df_task_trialtype in beh_df_task.groupby([trialtype_key,*additional_groupers]):
                res_d[(task_type,*trial_type)] = beh_df_task_trialtype

    beh_df['trial_type'] = np.empty(beh_df.shape[0],dtype=object)
    for k,val in res_d.items():
        beh_df.loc[val.index,'trial_type']=[[k]]*val.shape[0]
    beh_df['trial_type'] = beh_df['trial_type'].map(lambda x:x[0] if isinstance(x,list) else x)
    return res_d,beh_df





def load_spk_beh_df(session_dir,behavior=None,cell_metrics=None,force_reload=False,extra_load={},
                load_only=True,dosave=True,
                load_type='roman'
            ):
    '''
    extra_load: feed to prep.load_stuff, like 
    sessionPulses='*SessionPulses.Events.mat', i.e. name of the struct within the struct file = partial name of the struct used for glob

    '''
    save_data_dir = os.path.join(session_dir,'py_data')
    if not os.path.exists(save_data_dir):
        os.mkdir(save_data_dir)
        print(f'{save_data_dir} made!')
    save_data_fn = os.path.join(save_data_dir,'spk_beh_df.p')
    if os.path.exists(save_data_fn) & (not force_reload):
        res = pickle.load(open(save_data_fn,'rb'))
        return res
    if load_only:
        return 

    if (behavior is None) or (cell_metrics is None):
        to_return=prep.load_stuff(session_dir,**extra_load)
        cell_metrics=to_return['cell_metrics']
        behavior=to_return['behavior']
        
        extra_loaded_result = {k:to_return[k] for k in extra_load.keys() if k in to_return.keys()}
    if load_type=='roman':
        res = prep_roman_data(behavior,cell_metrics,**extra_loaded_result)
    elif load_type=='ipshita':
        res = prep_ipshita_data(behavior,cell_metrics,**extra_loaded_result)
    if dosave:
        pickle.dump(res,open(save_data_fn,'wb'))
        print(f'{save_data_fn} saved!')
    return res




def load_fr_map(session_dir,spk_beh_df=None,force_reload=False,speed_thresh=2.):
    '''
    session_dir: same as data_dir_full in the database
    either compute fr_map, or load fr_map, both trial averaged and for each trial
    using place_cell_analysis.get_fr_map_trial_multi_task
    '''
    save_data_dir = os.path.join(session_dir,'py_data')
    if not os.path.exists(save_data_dir):
            os.mkdir(save_data_dir)
            print(f'{save_data_dir} made!')
    save_data_fn = os.path.join(save_data_dir,'fr_map.p')
    if os.path.exists(save_data_fn) & (not force_reload):
        maps = pickle.load(open(save_data_fn,'rb'))
        return maps
    else:
        if spk_beh_df is None:
            res = load_spk_beh_df(session_dir,behavior=None,cell_metrics=None,force_reload=False) # remember not to force_reload this! load_spk_beh_df might has extra arguments like sessionPulses that need not be used here
            spk_beh_df = res['spk_beh_df']
            cell_cols_d = res['cell_cols_d']

        fr_map_task_dict = pa.get_fr_map_trial_multi_task(spk_beh_df,cell_cols_d['all'],trialtype_key_dict = TRIALTYPE_KEY_DICT,speed_thresh=speed_thresh)
        fr_map_trial_task_dict = pa.get_fr_map_trial_multi_task(spk_beh_df,cell_cols_d['all'],trialtype_key_dict = TRIALTYPE_KEY_DICT,speed_thresh=speed_thresh,order=['smooth','divide'])
        maps = {'fr_map':fr_map_task_dict,'fr_map_trial':fr_map_trial_task_dict}
        pickle.dump(maps,open(save_data_fn,'wb'))
        print(f'{save_data_fn} saved!')
        return maps


####### THETA#####
def get_theta_df(filtered,do_phase_correction=True):
    '''
    do_phase_correction: if True, shift the (-pi, 0) ones to (pi, 2pi); the theta_filtered i get currently is trough (-pi) to trough (-pi)
    '''
    if do_phase_correction:
        p = filtered['phase']
        filtered['phase'][p < 0] = np.pi * 2 + p[p<0]
    theta_df = pd.DataFrame(np.array([filtered['phase'],filtered['amp']]).T,index=filtered['timestamps'],columns=['theta_phase','theta_amp'])
    
    return theta_df

def merge_theta_into_beh_df(theta_df,beh_df):
    '''
    theta_df from get_theta_df: nlfpsamples x 2
    only select the time points from beh_df, using interpolation
    '''
    beh_time_mask=(theta_df.index>=beh_df.index[0]) & (theta_df.index<=beh_df.index[-1])
    f = interp1d(theta_df.index,theta_df.values,axis=0)
    theta_df_beh = pd.DataFrame(f(beh_df.index),columns=theta_df.columns,index=beh_df.index)
    beh_df_with_theta=pd.concat([beh_df,theta_df_beh],axis=1)
    
    return beh_df_with_theta, theta_df_beh

def add_lin_binned(spk_beh_df,bin_size=None,nbins=100):
    '''
    bin_size takes precedence, usually 2.2
    '''
    # lin_binned = spk_beh_df.groupby('task_index').apply(lambda x:pd.cut(x['lin'],bins=nbins,labels=False,retbins=True)[0]).T
    # bins = spk_beh_df.groupby('task_index').apply(lambda x:pd.cut(x['lin'],bins=nbins,labels=False,retbins=True)[1]).T
    bins_d = {}
    lin_binned_l = []
    for key, val in spk_beh_df.groupby('task_index'):
        if bin_size is not None:
            nbins = int(np.floor((val['lin'].max() - val['lin'].min()) / bin_size))
        lin_binned, bins = pd.cut(val['lin'],bins=nbins,labels=False,retbins=True)
        lin_binned_l.append(lin_binned)
        bins_d[key] = bins
    lin_binned_l = pd.concat(lin_binned_l)
    bins = bins_d
    
    # spk_beh_df['lin_binned'] = lin_binned.values
    spk_beh_df['lin_binned'] = lin_binned_l.values
    return spk_beh_df,bins
    

####### BRAIN REGION######
def get_cell_cols_d_with_brainregion(cell_metrics):
    '''
    get cell_cols_d, keys: (brain_region, cell_type), cell_type: 'pyr' pr 'int'
    '''
    region_mask={}
    cell_type_possible = ['pyr','int']
    cell_cols_d = {}
    for br,ct in itertools.product(np.unique(cell_metrics['brainRegion']),cell_type_possible):
        region_mask = cell_metrics['brainRegion']==br
        ct_mask = np.array([ct in s.lower() for s in cell_metrics['putativeCellType']])
        cell_cols_d[br,ct] = cell_metrics['UID'][region_mask & ct_mask]
    return cell_cols_d

########STIM ###########
def add_stim(spk_beh_df,sessionPulses,stim_key='stim'):
    stim_l = []
    for key, val in sessionPulses.items():
        stim_l.append(val[stim_key])
    spk_beh_df = add_trial_specific_info(spk_beh_df,stim_key,stim_l)
    return spk_beh_df

########process trial index#######
def trial_index_to_index_within_trialtype(spk_beh_df,mask_bad_trial=True):
    '''
    {key:map_series}
    map_series: series: index: trial_index; val: index_within_trialtype
    '''
    # spk_beh_df = spk_beh_df.loc[spk_beh_df['trial'].notna()] # filter out if trial==nan
    spk_beh_df = spk_beh_df.dropna(axis=1,how='all') # drop if everything is nan
    trial_index_by_trialtype = spk_beh_df.groupby('trial_type')['trial'].unique()
    aa=spk_beh_df.loc[spk_beh_df['trial_type']==(1,0)]['trial']
    trial_index_to_index_within_trialtype_d = {}
    for key,val in trial_index_by_trialtype.iteritems():
        map_series = pd.Series(val)
        map_series = map_series.reset_index().set_index(0)['index']
        trial_index_to_index_within_trialtype_d[key] = map_series
    trial_index_to_index_within_df = pd.concat(trial_index_to_index_within_trialtype_d,axis=0)
    # pdb.set_trace()
    if trial_index_to_index_within_df.index.nlevels==3: # patch such that still work for thomas data, which does not have trialtype
        trial_index_to_index_within_df.index.names=['task_ind','tt_ind','trial_ind']
    # to get rid of bad trials for roman's data: v should be nan; in general can pass in a mask
    if 'v' in spk_beh_df.columns:
        if mask_bad_trial:
            keep_mask=spk_beh_df.groupby('trial')['v'].apply(lambda x:x.notna().any())
            keep_mask=spk_beh_df.groupby(['task_index','trial'])['v'].apply(lambda x:x.notna().any())
        
        # xx=trial_index_to_index_within_df.reset_index().set_index('trial_ind').loc[keep_mask].set_index(['task_ind','tt_ind'],append='True').iloc[:,0]

        task_ind_l = trial_index_to_index_within_df.index.get_level_values(0).unique()
        trial_index_to_index_within_df_alltasks={}
        for task_ind in task_ind_l:
            trial_index_to_index_within_df_one_task = trial_index_to_index_within_df.loc[task_ind]
            if mask_bad_trial:
                keep_mask_onetask = keep_mask.loc[task_ind]
            else:
                keep_mask_onetask = np.ones(len(trial_index_to_index_within_df_one_task),dtype=bool)
            
            xx=trial_index_to_index_within_df_one_task.reset_index().set_index('trial_ind').loc[keep_mask_onetask].set_index('tt_ind',append='True').iloc[:,0]
            xx = xx.swaplevel('trial_ind','tt_ind')
            trial_index_to_index_within_df_alltasks[task_ind] = xx
        trial_index_to_index_within_df_alltasks = pd.concat(trial_index_to_index_within_df_alltasks)
        trial_index_to_index_within_df = trial_index_to_index_within_df_alltasks
        
        # xx = xx.swaplevel('trial_ind','task_ind').swaplevel('trial_ind','tt_ind')
        # trial_index_to_index_within_df = xx
    return trial_index_to_index_within_df#trial_index_to_index_within_trialtype_d

def index_within_to_trial_index(spk_beh_df):
    '''
    inverse of trial_index_to_index_within_trialtype
    index_within_to_trial_index_df:
        series: (task_index,trialtype_index, index_within_trialtype) : trial_index
    '''
    # trial_index_to_index_within_df = pd.concat(trial_index_to_index_within_trialtype(spk_beh_df),axis=0)
    trial_index_to_index_within_df = trial_index_to_index_within_trialtype(spk_beh_df)
    # pdb.set_trace()
    if trial_index_to_index_within_df.index.nlevels==3:
        index_within_to_trial_index_df = trial_index_to_index_within_df.reset_index(level=2).set_index('index',append=True)['trial_ind'].astype(int)
    elif trial_index_to_index_within_df.index.nlevels==2:
        index_within_to_trial_index_df = trial_index_to_index_within_df.reset_index(level=1).set_index('index',append=True)[0].astype(int)
    return index_within_to_trial_index_df


#========processing tmaze structure, running directions, etc.=========#
from scipy.spatial.distance import cdist

def get_v_kalman_linear(beh_df,key_post_fix='gauss'):
    '''
    only for the linear task
    after kalman already run; correcting the sign of v based on directions
    '''
    def sign_count_return_larger_ind(x):
        sgn = np.sign(x).value_counts().sort_values(ascending=False).index[0]
        return sgn
    speed_key = f'speed_{key_post_fix}'
    v_key = f'v_{key_post_fix}'
    
    direction_to_sign = beh_df.query(f'{speed_key}>5').groupby('direction')[v_key].apply(lambda x:sign_count_return_larger_ind(x))
    gpb= beh_df.groupby('direction')
    for dir,val in gpb:
        v_kalman = val[v_key] * direction_to_sign.loc[dir]
        beh_df.loc[val.index,v_key] = v_kalman
    beh_df[f'v_{key_post_fix}_aligned'] = beh_df[v_key] # same thing for linear, different for tmaze
    return beh_df





def get_v_kalman_and_aligned(beh_df,**kwargs):
    '''
    beh_df: remember to feed in only task==alternation
    v_kalman: signed speed_kalman
    v_kalman_aligned: v projected onto the direction of each arm, currently assuming arms are vertical or horizontal but not diagonal

    no longer using the projection, just getting a sign
    '''
    kwargs_ = {'n_lin':200,'speed_key':'speed_gauss','speed_thresh':10,'key_post_fix':'gauss','dist_to_corner_thresh':5}
    kwargs_.update(kwargs)
    corners_d,xy_sampled_d,segment_d=find_tmaze_turns(beh_df,n_lin=kwargs_['n_lin'],speed_key=kwargs_['speed_key'],speed_thresh=kwargs_['speed_thresh'])
    v_dir_d = get_v_direction(corners_d,thresh = 5.) # within thresh, considered no change
    gpb = beh_df.groupby('trial_type')
    post_fix_key = kwargs_['key_post_fix']
    vx_key = f'vx_{post_fix_key}'
    vy_key = f'vy_{post_fix_key}'
    v_key = f'v_{post_fix_key}'
    speed_key = kwargs_['speed_key']
    

    for tt, val in gpb:
        v_dir = v_dir_d[tt]
        val_xy_to_sample_dist = cdist(val[['x','y']].values,xy_sampled_d[tt])
        inds = np.argmin(val_xy_to_sample_dist,axis=1)
        seg_l = segment_d[tt][inds]
        x_sign_l = v_dir.loc[seg_l,'x_sign'].values
        y_sign_l = v_dir.loc[seg_l,'y_sign'].values
        x_coord_l = corners_d[tt].loc[seg_l,'x'].values
        y_coord_l = corners_d[tt].loc[seg_l,'y'].values
        v_aligned = val[vx_key] * x_sign_l + val[vy_key] * y_sign_l # assuming sign: one +-1, the other 0
        # v_off = val['vx_kalman'] * (x_sign_l==0) + val['vy_kalman'] * (y_sign_l==0) # assuming sign: one +-1, the other 0


        # deal with close to corner cases
        dist_to_corners = cdist(val[['x','y']].values,corners_d[tt].loc[0:3,['x','y']])
        dist_to_corners_argmin = np.argmin(dist_to_corners,axis=1)
        dist_to_corners_min = np.min(dist_to_corners,axis=1)
        dist_to_corner_thresh = 5
        close_to_corner = dist_to_corners_min < dist_to_corner_thresh

        seg_l_prev = (dist_to_corners_argmin - 1) % 4
        seg_l_next = (dist_to_corners_argmin) % 4
        # pdb.set_trace()
        x_sign_l_prev = v_dir.loc[seg_l_prev,'x_sign'].values
        y_sign_l_prev = v_dir.loc[seg_l_prev,'y_sign'].values
        v_aligned_prev = val[vx_key] * x_sign_l_prev + val[vy_key] * y_sign_l_prev

        x_sign_l_next = v_dir.loc[seg_l_next,'x_sign'].values
        y_sign_l_next = v_dir.loc[seg_l_next,'y_sign'].values
        v_aligned_next = val[vx_key] * x_sign_l_next + val[vy_key] * y_sign_l_next
        aligned_close_to_corner = ((np.sign(v_aligned_prev) + np.sign(v_aligned_next)) >= 0).astype(int) # lenient: if one direction is positive, count as positive
        aligned_close_to_corner = 2*(aligned_close_to_corner - 1/2)
        aligned_close_to_corner = aligned_close_to_corner[close_to_corner]
        

        aligned = np.sign(v_aligned)
        aligned[close_to_corner] = aligned_close_to_corner
        v_kalman = val[speed_key] * aligned
        # distance off the track
        # coord_off = (val['x']-x_coord_l) * (x_sign_l==0) + (val['y']-y_coord_l) * (y_sign_l==0) # select x/y as where x/y_sign==0 as the baseline coordinate to compute deviation 

        beh_df.loc[val.index,'segment'] = seg_l
        beh_df.loc[val.index,v_key] = v_kalman
        # beh_df.loc[val.index,'v_kalman_aligned'] = v_aligned
        # beh_df.loc[val.index,'v_kalman_off'] = v_off
        # beh_df.loc[val.index,'coord_off'] = coord_off
    
    return beh_df




def kde_findpeak(x,n_x = 80,**kwargs):
    kwargs_ = {'height':0.015}
    kwargs_.update(kwargs)
    x_range = np.linspace(np.min(x),np.max(x),n_x)
    kernel = scipy.stats.gaussian_kde(x)
    pdf = kernel(x_range)
    pdf_peaks = x_range[scipy.signal.find_peaks(pdf,**kwargs_)[0]]
    return pdf_peaks,pdf
    

# def find_t_maze_stoppoints_lin(xy,lin,radius=0.5,split_on_y=True,**kwargs):
#     '''
#     find the lin corresponding to corners/turns in t-maze, assuming those are where the animals spend more times 
#     (mode in the distribution)
    
#     find modes in kde of x and y positions, draw a small ball and average the corresponding lins. 
#     special case: home cage will be ambiguous whether it's 0 or end, make it 0.
    
    
#     xy: T x 2
#     lin: T,
#     split_on_y: y coordinate split into left and right turn
#     radius: only for xy within this radius, consider the corresponding lin for the weighted average
#     '''
    
#     x_peaks,_ = kde_findpeak(xy[:,0],**kwargs)
#     y_peaks,_ = kde_findpeak(xy[:,1],**kwargs)
#     lin_d ={}
#     for xp in x_peaks:
#         for yp in y_peaks:
#             dist = np.sqrt(np.sum((xy - np.array([[xp,yp]]))**2,axis=1)) + 1e-10
#             ma =  dist <= radius 
#             counts,edges=np.histogram(lin[ma])
#             if (edges[-1]-edges[0]) > 1/2 * (np.max(lin)-np.min(lin)): # detect 0, which can be confused with the end
#                 lin_p = 0.
#             else:
#                 lin_p = np.sum((lin[ma] * (1/dist[ma])) / (np.sum(1/dist[ma])))


#             lin_d[(xp,yp)]=lin_p
#     # reduce the lin values to avoid left right duplicates
#     lin_val = np.array(list(lin_d.values()))
#     lin_val_left = []
#     for v in lin_val:
#         ma = np.abs(v - lin_val) < 5.
#         if ma.sum()>0:
#             lin_val_left.append(np.mean(lin_val[ma]))
#             lin_val = lin_val[~ma]
#     lin_val_left = np.array(lin_val_left)
#     lin_val_left = np.sort(lin_val_left)
        
            
#     return lin_d,lin_val_left




def hist_find_peaks(x,bins=30,n_peaks=2):
    x=x[np.isfinite(x)]
    count,edges=np.histogram(x,bins)
    edge_centers = (edges[:-1]+edges[1:]) / 2
    inds = np.argsort(count)
    peaks = edge_centers[inds][-n_peaks:]
    return peaks

def find_closest_ind_2d_unique(xy_probe,xy):
    '''
    same way of getting distance as find_closest_ind_2d, only take the argmin of dist
    '''
    eps=1e-9
    xy_probe_diff = xy_probe[None,:] - xy
    dist=np.linalg.norm(xy_probe_diff,axis=1)
    ind = np.argmin(dist)
    return ind


def find_closest_ind_2d(xy_probe,xy):
    eps=1e-9
    xy_probe_diff = xy_probe[None,:] - xy
    dist=np.linalg.norm(xy_probe_diff,axis=1)
    dist_inv = 1 / dist
    
    y = dist_inv
    pw = 5
    y_padded = np.pad(y, pad_width=pw, mode='reflect')

    # Find the peaks
    height=0.2
    peaks, _ = scipy.signal.find_peaks(y_padded,height=0.2)
    while len(peaks)==0:
        height = height / 2
        peaks, _ = scipy.signal.find_peaks(y_padded,height=height)

    # Remove peaks that are too close to the edges
    peaks = peaks[(peaks > (pw-1)) & (peaks < (len(y_padded) - pw))] - 1
    peaks = peaks - (pw-1)
    
    return peaks


def map_xy_to_lin(xy_probe,lin,xy):
    peaks=find_closest_ind_2d(xy_probe,xy)
    # pdb.set_trace()
    lin_probe_ind = np.unique([peaks[0],peaks[-1]]) # keep the first and last peak! in the bimodal case, i.e. home cage, 2; other cases, drop the duplicates 
    if lin_probe_ind[0] >= 5:
        lin_probe_ind = np.array([int(np.mean(lin_probe_ind))]) # if the first ind not 0, then average the peaks, as they must be close and multi peaks due to noise
    else:
        lin_probe_ind[0] = 0. # manually set the edges
    if len(lin) - (lin_probe_ind[-1]+1) <=3:
        lin_probe_ind[-1] = len(lin)-1

    lin_probe = lin[lin_probe_ind]
    
    
    return lin_probe,lin_probe_ind

def get_xy_samples_from_lin_one(val,speed_key='speed_gauss',speed_thresh=10,n_lin=200,lin_st_eps=0.001):
    val = val.loc[val[speed_key]>speed_thresh] 
    val_trial = val.groupby('trial')
    xy_sampled_median = []
    lin_vals = np.linspace(val['lin'].min()+lin_st_eps,val['lin'].max(),n_lin)
    for tr, vt in val_trial:
        lin_one = vt['lin'].values
        
        lin_to_xy_func = interp1d(lin_one,vt[['x','y']].values,axis=0,fill_value='extrapolate')
        
        xy_sampled = lin_to_xy_func(lin_vals)
        xy_sampled_median.append(xy_sampled)
    xy_sampled_median = np.array(xy_sampled_median)
    xy_sampled_median = np.nanmedian(xy_sampled_median,axis=0)
    xy_sampled = xy_sampled_median
    ma=np.isnan(xy_sampled).any(axis=1)
    xy_sampled  = xy_sampled[~ma]
    return xy_sampled,lin_vals


def find_tmaze_turns(beh_df,n_lin = 200,speed_key='speed_gauss',speed_thresh=10,filt_win=5):
    '''
    use some lin points, map to xy points using interpolation, use the peak values to get the maze boundaries
    then get corners, then map corners back to lin using shortest distance
    0 and end are both kept
    two trial types are mapped seperately

    n_lin: # bins for mapping to xy
    '''
    beh_df=beh_df.loc[beh_df['task']=='alternation']
    gpb=beh_df.groupby('trial_type')
    # lin_st_eps = 0.001 # if 0, then get nan
    corners_d = {}
    xy_sampled_d = {} # a more regularized maze cooridnates
    segment_d = {}
    lin_st_eps = 1e-3
    for tt, val in gpb:
        xy_sampled,lin_vals = get_xy_samples_from_lin_one(val,speed_key,speed_thresh,lin_st_eps=lin_st_eps)
        # if filt_win is not None:
        #     xy_sampled = scipy.ndimage.median_filter(xy_sampled,(filt_win,1))
        xy_sampled_d[tt] = xy_sampled
        segment_l = np.zeros(xy_sampled.shape[0])
        x_bounds = hist_find_peaks(xy_sampled[:,0],n_peaks=2)
        y_bounds = hist_find_peaks(xy_sampled[:,1],n_peaks=2)
        xy_corners = np.array(list(itertools.product(x_bounds,y_bounds)))
        
        
        xyc_l = []
        lc_l = []
        ind_c_l = []
        for xyc in xy_corners:
            # ind_c = find_closest_ind_2d(xyc,xy_sampled)
            lin_c,ind_c = map_xy_to_lin(xyc,lin_vals,xy_sampled) # lin_vals and xy_sampled could be different, so this function might be weird; but it's not used anyway

            # lin_corners.append(lin_c)
            # for lc in lin_c:
            for lc,ic in zip(lin_c,ind_c):
                if lc==lin_st_eps: 
                    lc = val['lin'].min() # turn it back to min value
                ind_c_l.append(ic)
                xyc_l.append(xyc)
                lc_l.append(lc)
        xyc_l = np.array(xyc_l)
        lc_l = np.array(lc_l)
        ind_c_l = np.sort(ind_c_l)
        for ii,(st,ed) in enumerate(zip(ind_c_l[:-1],ind_c_l[1:])):
            segment_l[st:ed+1] = ii # add 1 to right bounds
        segment_d[tt] = segment_l
        corners = np.concatenate([xyc_l,lc_l[:,None]],axis=1)
        corners = pd.DataFrame(corners,columns=['x','y','lin'])
        corners = corners.sort_values('lin').reset_index(drop=True)
        corners_d[tt] = corners
    
    xy_sampled_d,segment_d = fill_in_gap_xy_sample(xy_sampled_d,segment_d)
    
    return corners_d,xy_sampled_d,segment_d

# fill in the gap
def fill_in_gap_xy_sample(xy_sampled_d,segment_d,gap_z_thresh = 1):
    '''
    sometimes the samples have gaps; fill them in by first detecting gaps i.e. consecutive points whose distances 
    are bigger than gap_z_thresh std, then do linear interpolation

    [NB!] if the gap occurs on the turns, would be problematic....
    '''
    segment_d_filled = {}
    xy_sampled_d_filled = {}
    for k,xys in xy_sampled_d.items():
        xys = np.append(xys,xys[[0]],axis=0) # append the first row to the last to deal with that gap
        gaps = np.linalg.norm(np.diff(xys,axis=0),axis=1)
        gaps_z = scipy.stats.zscore(gaps)
        expected_gap = gaps[gaps_z < 1].mean()
        
        gap_ind = np.nonzero(gaps_z > gap_z_thresh )[0] 

        segment_one = segment_d[k]
        for gi in gap_ind[::-1]:
            gap_size = gaps[gi]
            npts = int(gap_size / expected_gap)
            added_x = np.linspace(xys[gi,0],xys[gi+1,0],npts+2)[1:-1]
            added_y = np.linspace(xys[gi,1],xys[gi+1,1],npts+2)[1:-1]
            added_xy = np.stack([added_x,added_y],axis=1)
            xys = np.insert(xys,gi,added_xy,axis=0)
            seg_insert = np.array([segment_d[k][gi]] * npts)
            segment_one = np.insert(segment_one,gi,seg_insert)
            
        segment_d_filled[k] = segment_one
        xys = xys[:-1] # get rid of the append
        
        xy_sampled_d_filled[k] =xys

    return xy_sampled_d_filled, segment_d_filled




# import ruptures as rpt
# def find_tmaze_turns(beh_df,n_lin = 200,speed_key='speed_kalman',speed_thresh=10):
#     '''
#     use some lin points, map to xy points using interpolation
#     then use cpd on the velocity to get the turns, append the beginning
#     get which segment each sample xy belongs to, 
#     also the signs of v within that segment 
#     '''
#     beh_df=beh_df.loc[beh_df['task']=='alternation']
#     gpb=beh_df.groupby('trial_type')
#     lin_st_eps = 0.001 # if 0, then get nan
#     corners_d = {}
#     xy_sampled_d = {} # a more regularized maze cooridnates
#     n_segments_in_a_loop = 4
#     for tt, val in gpb:
#         val = val.loc[val[speed_key]>speed_thresh]
        
#         lin_one = val['lin'].values
#         lin_to_xy_func = interp1d(lin_one,val[['x','y']].values,axis=0)
#         lin_vals = np.linspace(lin_one.min()+lin_st_eps,lin_one.max(),n_lin)
#         xy_sampled = lin_to_xy_func(lin_vals)
#         xy_sampled_d[tt] = xy_sampled
#         # based on within each segment, the moving direction will be non zero mean low variance, the non moving direction will be zero mean with high variance
#         y = np.pad(xy_sampled,pad_width=(1,0),mode='edge')
#         y = np.diff(y,axis=0)
#         algo = rpt.Dynp(model='normal',min_size=10,jump=1).fit(y) # make sure # bins within a segment >=10
#         result = algo.predict(n_segments_in_a_loop-1) # that will give 4 segments
#         result =  np.insert(result,0,0)
        
#         segment_l = np.zeros(xy_sampled.shape[0])
#         for ii,(st,ed) in enumerate(zip(result[:-1],result[1:])):
#             segment_l[st:ed] = ii

#         result[-1] = result[-1]-1 #
#         corners_xy = xy_sampled[result]
#         corners_lin = lin_vals[result]
#         corners = np.concatenate([corners_xy,corners_lin[:,None]],axis=1)
#         corners_df = pd.DataFrame(corners,columns=['x','y','lin'])
#         corners_d[tt] = corners_df
        
#     return corners_d, xy_sampled_d, segment_l
        







def get_v_direction(corners_d,thresh = 5.):
    '''
    as a function of lin, the sign of vx and vy that correspond to a positive v
    '''
    v_direction_d = {}
    for tt,corners in corners_d.items():
        v_direction_df = []
        ncorners = corners.shape[0]
        for nn in range(ncorners-1):


            x_diff = (corners['x'][nn+1] - corners['x'][nn])
            x_sign = np.sign(x_diff) * (np.abs(x_diff)>thresh)

            y_diff = (corners['y'][nn+1] - corners['y'][nn])
            y_sign = np.sign(y_diff) * (np.abs(y_diff)>thresh)
            v_direction_df.append([corners['lin'][nn],corners['lin'][nn+1],x_sign,y_sign])
        v_direction_df = pd.DataFrame(v_direction_df,columns=['lin_st','lin_end','x_sign','y_sign'])
        v_direction_d[tt] = v_direction_df
    return v_direction_d


#======processing sleep related=====================#

def add_ripple_peak_to_spike_trains(spike_trains,ripple_peak_times):
    rip_col_ind = len(spike_trains) + 1 # it's 1 indexed
    spike_trains_new_d={k:nap.Ts(val.index.values) for k,val in dict(spike_trains).items()} # this comes from version difference.
    spike_trains_new_d[rip_col_ind]=nap.Ts(ripple_peak_times)
    spike_trains_new = nap.TsGroup(spike_trains_new_d)
    return spike_trains_new,rip_col_ind

def get_nrem_firing_with_ripple_time(spike_trains, nrem_episode_intervals, ripple_peak_times, bin_size=0.1):
    '''
    bin the spikes that happen in the nrem_episode_intervals, add to the resulting dataframe info from ripple_peaks_times 

    spike_trains: nap.TsGroups
    nrem_episode_intervals: N x 2, any intervals for restricting the spike times, in the end will be concatenated
    ripple_peak_times: N_rip, 

    '''
    spike_trains_new,rip_col_ind = add_ripple_peak_to_spike_trains(spike_trains,ripple_peak_times)

    nrem = nap.IntervalSet(start=nrem_episode_intervals[:,0],end=nrem_episode_intervals[:,1])
    spk_train_nrem = spike_trains_new.restrict(nrem)
    spk_count_nrem = spk_train_nrem.count(bin_size=0.1)
    spk_count_nrem=spk_count_nrem.rename({rip_col_ind:'ripple_peak'},axis=1)
    
    return spk_count_nrem