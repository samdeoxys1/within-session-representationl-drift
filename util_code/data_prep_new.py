import numpy as np
import scipy,os,copy
import pandas as pd

from scipy.ndimage import gaussian_filter1d
from plot_helper import *
import pynapple as nap

def get_speed(pos_2d,dt,is2d=True):
    if is2d:
        vel = np.diff(pos_2d,prepend=pos_2d[[0],:],axis=0)/dt
        speed = np.sqrt(vel[:,0]**2+vel[:,1]**2)
    else:
        vel = np.diff(pos_2d,prepend=pos_2d[0]) / dt
        speed = np.sqrt(vel**2)
    return speed,vel
def smooth_get_speed(pos_2d,dt,sigma=30,is2d=True):
    pos_2d_smth = gaussian_filter1d(pos_2d,sigma=sigma,axis=0)
    speed,vel_2d = get_speed(pos_2d_smth,dt,is2d=is2d)
    return speed,vel_2d


def get_beh_df(behav_timestamps,position,visitedArm,startPoint,n_pos_bins=100):
    beg_end = (behav_timestamps[0],behav_timestamps[-1])
    # get velocity
    behav_dt = np.diff(behav_timestamps).mean()
    # vel = np.diff(position,axis=0,prepend=position[[0],:]) / behav_dt
    pos_2d=position
    speed,vel_2d=smooth_get_speed(pos_2d,behav_dt)

    df=pd.DataFrame()
    df['x'] = position[:,0]
    df['y'] = position[:,1]
    df['lin'] = position[:,2]

    df['vel_x'] = vel_2d[:,0]
    df['vel_y'] = vel_2d[:,1]
    df['vel_lin'] = vel_2d[:,2]
    df['speed'] = speed #np.sqrt(df['vel_x']**2 + df['vel_y']**2)
    df['times'] = behav_timestamps

    peak_times = startPoint[:,0]
    peak_times = np.append(peak_times,startPoint[-1,1])
    # filter df to start when trial starts
    df = df.loc[df['times'] >= peak_times[0]]
    df = df.loc[df['times'] <= peak_times[-1]]

    # for assigning a trial and visitedArm to each time bin
    condlist = [(df['times'].values <peak_times[ii+1]) & (df['times'].values >=peak_times[ii]) for ii in range(peak_times.shape[0]-1)]
    # funclist_visitedArm = visitedArm[1:] #for the bad session 1: seems to make sense? Normally shouldn't need it? Need to check more sessions
    funclist_visitedArm = visitedArm
    funclist_trial =  range(len(funclist_visitedArm))
    visitedArm_in_time=np.piecewise(df['times'].values,condlist,funclist_visitedArm)
    trial_in_time =np.piecewise(df['times'].values,condlist,funclist_trial)
    df['visitedArm'] = visitedArm_in_time
    df['trial'] = trial_in_time

    pos_bins_dict = {}
    for k in ['x','y','lin']:
        df[f'{k}_binned'],pos_bins_dict[f'{k}'] = pd.cut(df[k], bins=n_pos_bins, labels=False, retbins=True)
    return df,pos_bins_dict

#==key: get df with fr and behavior==#
def get_fr_beh_df(spike_times,uid,behav_timestamps,cell_type,position,visitedArm,startPoint,n_pos_bins=100,**kwargs):
    cell_type_dict = {'pyr':1,'int':0}
    # cell_type_dict = {'pyr':1}
    uid_dict ={}
    # beg_end = (behav_timestamps[0],behav_timestamps[-1])
    fr_mat_dict = {}
    
    kwargs_default = {'smooth':None,'normalize':None,'compare_pop_max':False,'israte':False,'bin_info':behav_timestamps}
    kwargs_default.update(kwargs)

    isempty_dict = {}
    for k,v in cell_type_dict.items():    
        inds_mask = cell_type == v # filter by cell type
        if inds_mask.sum()>0:
            isempty_dict[k] = False
            spk_times_one_cell_type = spike_times[inds_mask]
            # fr_mat_dict[k],bins = bin_spikes(spk_times_one_cell_type, bin_info=behav_timestamps,smooth=kwargs_default['smooth'], beg_end = None, normalize=kwargs_default['smooth'],compare_pop_max=kwargs_default['smooth'], israte=kwargs_default['smooth'],**kwargs)
            # import pdb
            # pdb.set_trace()
            fr_mat_dict[k],bins = bin_spikes(spk_times_one_cell_type,beg_end = None,**kwargs_default)
            uid_dict[k] = uid[inds_mask]
        else:
            isempty_dict[k] = True
            fr_mat_dict[k] = np.array([])
            uid_dict[k] = np.array([])


    times = np.concatenate([bins[:-1][:,None],bins[1:][:,None]],axis=1).mean(axis=1) # get the mean of the adjacent bins
    
    # get velocity
    behav_dt = np.diff(behav_timestamps).mean()
    # vel = np.diff(position,axis=0,prepend=position[[0],:]) / behav_dt

    # align behavior and fr
    f = scipy.interpolate.interp1d(behav_timestamps,position,axis=0)
    position_interp = f(bins[:-1]) # in principle sould use times, but then issue with edge effect being interpolated to create a bin in the middle
    # f = scipy.interpolate.interp1d(behav_timestamps,vel,axis=0)
    # vel_interp = f(bins[:-1])
    beh_df,pos_bins_dict=get_beh_df(times,position_interp,visitedArm,startPoint,n_pos_bins=n_pos_bins)

    df_dict = {}
    cell_cols_dict={}
    for key in cell_type_dict.keys():
        if isempty_dict[key]:
            df_dict[key] = pd.DataFrame([])
        else:
            fr_mat_ = fr_mat_dict[key]
            # columns = [f'unit_{i}' for i in range(fr_mat_pyr.shape[0])] 
            cell_cols_dict[key] = [f'unit_{i}' for i in uid_dict[key]] 
            df = pd.DataFrame(fr_mat_.T,columns=cell_cols_dict[key])
            # df['x'] = position_interp[:,0]
            # df['y'] = position_interp[:,1]
            # df['lin'] = position_interp[:,2]

            # df['vel_x'] = vel_interp[:,0]
            # df['vel_y'] = vel_interp[:,1]
            # df['vel_lin'] = vel_interp[:,2]
            # df['speed'] = np.sqrt(df['vel_x']**2 + df['vel_y']**2)
            df['times'] = times

            peak_times = startPoint[:,0]
            peak_times = np.append(peak_times,startPoint[-1,1])
            # filter df to start when trial starts
            df = df.loc[df['times'] >= peak_times[0]]
            df = df.loc[df['times'] <= peak_times[-1]]

            # for assigning a trial and visitedArm to each time bin
            # condlist = [(df['times'].values <peak_times[ii+1]) & (df['times'].values >=peak_times[ii]) for ii in range(peak_times.shape[0]-1)]
            # funclist_visitedArm = visitedArm[1:] #for the bad session 1: seems to make sense? Normally shouldn't need it? Need to check more sessions
            # funclist_visitedArm = visitedArm
            # funclist_trial =  range(len(funclist_visitedArm))
            # visitedArm_in_time=np.piecewise(df['times'].values,condlist,funclist_visitedArm)
            # trial_in_time =np.piecewise(df['times'].values,condlist,funclist_trial)
            # df['visitedArm'] = visitedArm_in_time
            # df['trial'] = trial_in_time

            # pos_bins_dict = {}
            # for k in ['x','y','lin']:
                # df[f'{k}_binned'],pos_bins_dict[f'{k}'] = pd.cut(df[k], bins=n_pos_bins, labels=False, retbins=True)
            df = pd.concat([df,beh_df],axis=1)
            df = df.loc[:,~df.columns.duplicated()] # repeated column
            df_dict[key] = df

    return df_dict, pos_bins_dict, cell_cols_dict

def bin_spikes(spike_times,bin_info=None, smooth=None, normalize = None, beg_end=None, compare_pop_max=True, israte=True,**kwargs):
    '''
    spike_times: list of Ncell arrays
    bin_info: can be time duration in second, or bin edges in second
    '''
    fr_mat = []
    
    if isinstance(bin_info,float):
        bin_size = bin_info
        if beg_end is None:
            min_time = np.min([np.min(t) for t in spike_times])
            max_time = np.max([np.max(t) for t in spike_times])
            beg_end = (min_time-0.001, max_time+0.001)
        bins = np.arange(beg_end[0],beg_end[1],bin_size)
    else:
        bin_size = bin_info[1] - bin_info[0]
        bins = bin_info
        beg_end = (bin_info[0],bin_info[-1])
    
    for spk in spike_times:
        count,bins = np.histogram(spk,bins=bins)
        fr_mat.append(count)
    fr_mat = np.vstack(fr_mat)

    if israte:
        fr_mat = fr_mat / bin_size
    if smooth is not None: # then it should be the gaussian sigma
        fr_mat = gaussian_filter1d(fr_mat, smooth, axis=1)
    if normalize=='z_score':
        fr_mat = (fr_mat - fr_mat.mean(axis=1,keepdims=True)) / fr_mat.std(axis=1, keepdims=True)
    elif normalize=='max':
        if compare_pop_max:
            pop_max_comparison = fr_mat.max() * 0.95 #0
        else:
            pop_max_comparison = fr_mat.max() * 0
        fr_mat = fr_mat / np.maximum(fr_mat.max(axis=1,keepdims=True), pop_max_comparison)
    
    return fr_mat,bins

def get_bins(bin_size,edge_times):
    '''
    edges_times: Nintervals x 2
    '''
    bins_all = []
    for ts in edge_times:
        bins=np.arange(ts[0],ts[1]+bin_size,bin_size)
        bins_all.append(bins)
    bins_all = np.concatenate(bins_all)
    return bins_all

def map_edge_times_to_index(edge_times,bin_size,beg,padding_bins=0):
    '''
    assume the binned spikes are continuous in time, has a begginning in time (beg)
    '''
    inds = (edge_times - beg) // bin_size
    time = []
    for ii,ind in enumerate(inds):
        nbins = int(ind[1] - ind[0])
        time_at_bins = np.linspace(edge_times[ii][0],edge_times[ii][1],nbins)
        time.append(time_at_bins)
        time.append(np.ones(padding_bins) * -1)
    time = np.concatenate(time)
        
    return inds.astype(int), time
    
    
def get_ripple_fr_mat(fr_mat,ripple_times,bin_size=0.01,min_seq_l=60/1000,begin=0,padding_time=0.6):
    '''
    fr_mat: N x Ntimebins; firing rate for all time
    ripple_times: Nripples x 2: in seconds
    bin_size: size of the bin for fr_mat, in seconds
    min_seq_l: ripples shorter than this will be filtered out; in seconds
    padding_time: the duration of added 0 in between ripples
    
    ====
    fr_mat_ripple: N x Ntimebins, firing rate for all ripple longer than min_seq_l, with 0 in between ripples
    time_vec: Ntimebins, the corresponding real world time for each bin, in second 
    '''
    padding_bins = int(padding_time / bin_size)
    filtered_ripple_times = np.diff(ripple_times).squeeze() > min_seq_l

    begin = 0
    ripple_inds,time_vec = map_edge_times_to_index(ripple_times[filtered_ripple_times,:], bin_size,begin,padding_bins=padding_bins)

    N = fr_mat.shape[0]
    fr_mat_ripple = []
    time_vec_with_fill = []
    
    for ind in ripple_inds:
        fr_mat_ripple.append(fr_mat[:,ind[0]:ind[1]])
        fr_mat_ripple.append(np.zeros((N,padding_bins)))
        
    fr_mat_ripple = np.concatenate(fr_mat_ripple,axis=1)
    
    return fr_mat_ripple,time_vec


#======#
def select_time_in_intervals(times,int_l):
    if len(int_l.shape)>1:
        times_l = []
        mask_l =[]
        for int in int_l:
            mask = (int[0] <=times)&(times<= int[1])
            times_l.append(times[mask])
            mask_l.append(mask)
        return times_l,mask_l
    else:
        mask = (int_l[0] <=times)&(times<= int_l[1])
        return times[mask], mask

def select_time_in_intervals_all(spike_times,int_l):
    times_selected_l=[]
    for times in spike_times:
        times_selected,mask=select_time_in_intervals(times,int_l)
        times_selected_l.append(times_selected)
    times_selected_l = np.array(times_selected_l,dtype=object)
    return times_selected_l,mask

def count_in_intervals(times,int_l):
    '''
    NB: assume disjoint intervals!
    First flatten, then histogram, then select 0::2
    get counts in intervals as well as between intervals, as well as rate
    '''
    edges = int_l.flatten()
    count,_=np.histogram(times,bins=edges)
    rate = count / np.diff(edges)
    count_in_interval = count[::2]
    count_between_interval = count[1::2]
    rate_in_interval = rate[::2]
    rate_between_interval = rate[1::2]
    return count_in_interval, rate_in_interval, count_between_interval, rate_between_interval

def count_in_intervals_all(spike_times,int_l):
    count_in_interval_all = []
    rate_in_interval_all = []
    count_between_interval_all = []
    rate_between_interval_all = []
    for times in spike_times:
        count_in_interval, rate_in_interval, count_between_interval, rate_between_interval = count_in_intervals(times,int_l)
        count_in_interval_all.append(count_in_interval)
        rate_in_interval_all.append(rate_in_interval)
        count_between_interval_all.append(count_between_interval)
        rate_between_interval_all.append(rate_between_interval)
    count_in_interval_all = np.array(count_in_interval_all)
    rate_in_interval_all = np.array(rate_in_interval_all)
    count_between_interval_all = np.array(count_between_interval_all)
    rate_between_interval_all = np.array(rate_between_interval_all)

    return count_in_interval_all, rate_in_interval_all, count_between_interval_all, rate_between_interval_all

# %%
#========loading all the needed variables from a sess folder=====#
def load_sess(sess_name, data_dir='/mnt/home/szheng/ceph/ad/Chronic_H2/',data_dir_full=None):
    if data_dir_full is None:
        animal = sess_name.split('_')[0]
        data_dir = os.path.join(data_dir,animal,sess_name)
    else:
        data_dir = data_dir_full # data_dir_full is used when the naming convention is different from Marisol's
    data_fn = f'{sess_name}.cell_metrics.cellinfo.mat'
    try:
        data = loadmat(os.path.join(data_dir,data_fn))
    except:
        data = mat73.loadmat(os.path.join(data_dir,data_fn))

    spike_times = np.array(data['cell_metrics']['spikes']['times'],dtype=object)
    uid = data['cell_metrics']['UID'].squeeze().astype(np.int32)
    fr = data['cell_metrics']['firingRate'].squeeze()

    cell_type=np.array([1 if 'pyr' in x.lower() else 0 for x in data['cell_metrics']['putativeCellType']],dtype=bool)
    cell_metrics = data['cell_metrics']
    del data

    data_fn = f'{sess_name}.MergePoints.events.mat'
    try:
        data = loadmat(os.path.join(data_dir,data_fn))
    except:
        data = mat73.loadmat(os.path.join(data_dir,data_fn))
    mergepoints = data['MergePoints']['timestamps']


    data_fn = f'{sess_name}.Behavior.mat'
    try:   
        data = loadmat(os.path.join(data_dir,data_fn))
    except:
        data = mat73.loadmat(os.path.join(data_dir,data_fn))
    behav_timestamps = np.squeeze(data['behavior']['timestamps'])
    x = data['behavior']['position']['x']
    y = data['behavior']['position']['y']
    lin = data['behavior']['position']['lin']
    position = np.concatenate([x[:,None],y[:,None],lin[:,None]],axis=1)

    rReward=data['behavior']['events']['rReward']
    lReward=data['behavior']['events']['lReward']

    behav_dt = np.diff(behav_timestamps).mean()

    endDelay=data['behavior']['trials']['endDelay']
    startPoint=data['behavior']['trials']['startPoint']
    visitedArm = data['behavior']['trials']['visitedArm'].squeeze()

    behavior = data['behavior']

#     bin_lin,bins_for_lin = pd.cut(lin.squeeze(),bins=100,labels=False, retbins=True)
    return cell_metrics,behavior,spike_times,uid,fr,cell_type,mergepoints,behav_timestamps,position,\
            rReward,lReward,endDelay,startPoint,visitedArm


# %%
#=========loading matlab struct into python dict============#
# %%
import scipy.io as spio
import mat73
import scipy.io.matlab as spiomat

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    try:
        data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
        return _check_keys(data)
    except:
        return mat73.loadmat(filename)



class DotDict(dict):
    # def __getattr__(self, name):
    #     return self[name]
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value
    
    def __repr__(self) -> str:
        return str(list(self.keys()))
    

def loadmat_full(filename,structname=None):
    if structname is None:
        mat = loadmat(filename)
    else:
        mat = loadmat(filename)[structname]
    mat = DotDict(mat)
    return mat

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spiomat.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
        
        elif isinstance(dict[key], np.ndarray):
            
            dict_key_res = np.zeros_like(dict[key])
            # with np.nditer([dict[key],dict_key_res],op_flags=[['readonly'], ['readwrite']]) as it:
            for ind,x in np.ndenumerate(dict_key_res): 
                orig_val = dict[key][ind]
                
                if isinstance(orig_val,scipy.io.matlab.mat_struct):
                    dict_key_res[ind] = _todict(orig_val)
                else:
                    dict_key_res[ind] = orig_val

            dict[key] = dict_key_res
        
    return dict

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spiomat.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        elif isinstance(elem,np.ndarray) and len(elem) >= 1: # used for the multi maze case; then there might be a cell array of struct that is not correctly unwrapped
            if isinstance(elem[0], spiomat.mio5_params.mat_struct):
                dict[strg] = np.array([_todict(e) for e in elem],dtype=object)
            else:
                dict[strg]  = elem
        else:
            dict[strg]  = elem
    return dict

# %%
