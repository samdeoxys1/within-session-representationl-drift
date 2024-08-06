'''
some preliminary analysis that involves place fields
'''
import numpy as np
import scipy
import pandas as pd
import sys,os,copy,itertools,pdb,pickle,tqdm
import data_prep_pyn as dpp
import pynapple as nap
sys.path.append('/mnt/home/szheng/projects/nmf_analysis')
import nmf_analysis as na
import pandarallel
import place_cell_analysis as pa
from scipy.ndimage import gaussian_filter1d

def ratemap_from_spk_beh_df_onetrialtype(spk_beh_df,cell_cols,speed_thresh=1,bin_size=2.2,nbins=100,smth_in_bin=5,speed_key=None):
    '''
    spk_beh_df: df that has neurons spike counts and behavior, in behavior timestamps
    fr_map: nneurons x nposbins
    '''
    if bin_size is not None:
        nbins = int(np.floor((spk_beh_df['lin'].max() - spk_beh_df['lin'].min()) / bin_size))
    if 'lin_binned' not in spk_beh_df.columns:
        spk_beh_df['lin_binned'],bins = pd.cut(spk_beh_df['lin'],nbins,retbins=True,labels=False)
    if speed_key is None:
        if (spk_beh_df['speed'] > speed_thresh).mean() <0.001:  # in some roman's sessions x y can't be used, so use his v for velocity estimate
            speed_key = 'v'
        else:
            speed_key= 'speed'
    # speed_ma = np.abs(spk_beh_df[speed_key]) > speed_thresh
    speed_ma = spk_beh_df[speed_key] > speed_thresh # perhaps using not the abs makes more sense? just looking at one direction for one trial in 1d task
    spk_beh_df_ma = spk_beh_df.loc[speed_ma]
    gpb = spk_beh_df_ma.groupby(['lin_binned'])
    occu = gpb['lin'].count()

    count = gpb[cell_cols].sum()
    if (smth_in_bin!=0) and (smth_in_bin is not None):
        occu_smth = gaussian_filter1d(occu.values.astype(float),sigma=smth_in_bin)
        count_smth = gaussian_filter1d(count.values.astype(float),sigma=smth_in_bin,axis=0) # float is crucial!
    else:
        occu_smth = occu.values.astype(float)
        count_smth = count.values.astype(float)
    
#     fr = count_smth.values / occu_smth.values[:,None]
    fr = count_smth / occu_smth[:,None]
    try:
        dt = np.median(np.diff(spk_beh_df['time']))
    except:
        dt = np.median(np.diff(spk_beh_df.index))
    fr = fr / dt
    fr_map = pd.DataFrame(fr,index=count.index,columns=count.columns).T
    return fr_map,count_smth.T,occu_smth

def ratemap_from_spk_beh_df_alltrialtype(spk_beh_df,cell_cols,speed_thresh=1,bin_size=2.2,nbins=100,smth_in_bin=5,speed_key=None):
    if 'trial_type' not in spk_beh_df.columns:
        _,spk_beh_df=dpp.group_into_trialtype(spk_beh_df)
    gpb = spk_beh_df.groupby('trial_type')
    
    fr_map_d,count_smth_d,occu_smth_d = {},{},{}
    for k,val in gpb:
        fr_map_d[k],count_smth_d[k],occu_smth_d[k]=ratemap_from_spk_beh_df_onetrialtype(val,cell_cols,speed_thresh=speed_thresh,bin_size=bin_size,nbins=nbins,smth_in_bin=smth_in_bin,speed_key=speed_key)
    return fr_map_d,count_smth_d,occu_smth_d
        
def test_frmap_sig(fr_map,count,occu,dt):
    '''
    using meanfr as a poisson test
    '''
    lam = fr_map.mean(axis=1)
    pvals = 1-scipy.stats.poisson(lam.values[:,None] * occu[None,:] * dt).cdf(count)
    pvals_df = pd.DataFrame(pvals,index=fr_map.index,columns=fr_map.columns)
    return pvals_df

def series_of_list_to_df(series,maxn):
    '''
    series: series of list
    maxn: new col size
    for each series value, turn it into columns of df 
    '''
    df = {}
    for k,val in series.iteritems():
        default = np.nan * np.ones(maxn)
        default[:len(val)] = np.array(val)
        df[k]=default
    df = pd.DataFrame(df).T
    return df

def get_field_boundary_for_each_field(fields,isfield_df):
    '''
    fields: series, for one neuron, nmaxfield
    #isfield: series, for one neuron, nposbins
    isfield_df: df: nneurons x nposbins
    '''
    isfield = isfield_df.loc[fields.name]
    ma=np.diff(isfield.astype(int),prepend=0,append=0)
    start=np.nonzero(ma==1)[0]
    end=np.nonzero(ma==-1)[0] - 1 # -1, because in pd loc, the right end is included
    field_bounds=[]
    for k,posbin in fields.iteritems():
        if not np.isnan(posbin):
            for st,ed in zip(start,end):
                if (posbin>=st) & (posbin<=ed):
                    field_bounds.append(np.array([st,ed,posbin]))
        else:
            field_bounds.append(np.array([np.nan,np.nan,np.nan]))
    field_bounds = pd.DataFrame(field_bounds,columns=['start','end','peak'])
    return field_bounds.T.unstack() # series: (nfieldsmax x 2) , 
                
    

# def get_field_peaks_one_trialtype(fr_map,count,occu,dt=0.033,alpha_peak=0.01,alpha_bound=0.1,min_count=3,min_rate=1):
#     '''
#     get place fields using average fr_map
#     poisson test for significance, use that to determine the boundary of field as well
#     '''
#     fr_map_peaks = fr_map.apply(lambda x:scipy.signal.find_peaks(x,height=min_rate)[0],axis=1)
#     pvals_df = test_frmap_sig(fr_map,count,occu,dt)
#     peak_isfield=(pvals_df < alpha_peak) & (count > min_count) & (fr_map > min_rate) # condition for peak
#     bound_isfield = pvals_df < alpha_bound # condition for the wider boundary of the field

#     fr_map_peaks_tested={}
#     for k,val in fr_map_peaks.iteritems():
#         fr_map_peaks_tested[k]=[xx for xx in val if peak_isfield.loc[k,xx]]
#     fr_map_peaks_tested = pd.Series(fr_map_peaks_tested) # the test is quite conservative, and prune out small peaks
#     max_nfields = fr_map_peaks_tested.apply(lambda x:len(x)).max()

#     fr_map_peaks_tested_df = series_of_list_to_df(fr_map_peaks_tested,max_nfields)
#     field_boundary_df = fr_map_peaks_tested_df.apply(get_field_boundary_for_each_field,axis=1,args=(bound_isfield,)) # nneurons x nfieldsmax
#     all_fields =field_boundary_df.stack(level=0).stack() # nfields_total,  (neurons, (start,end), nfields), no nan terms
#     res = dict(
#         fr_map_peaks_tested_df=fr_map_peaks_tested_df, # after the test; prune out some low fr field; nneurons x maxnfields
#         fr_map_peaks = fr_map_peaks, # before the test; series: nneurons, each val is a list of peaks
#         pvals_df = pvals_df, # nneurons x nposbins
#         peak_isfield = peak_isfield, # nneurons x nposbins
#         bound_isfield = bound_isfield, # nneurons x nposbins
#         field_boundary_df = field_boundary_df,
#         all_fields = all_fields
#     )
#     return res


def get_field_peaks_one_trialtype(fr_map,count,occu,dt=0.033,alpha=0.01,min_count=3,min_rate=1):
    '''
    get place fields using average fr_map
    poisson test for significance, use that to determine the boundary of field as well
    '''
    fr_map_peaks = fr_map.apply(lambda x:x.index[scipy.signal.find_peaks(x,height=min_rate)[0]],axis=1) # note some bins might be missing, so need to convert from array index to bin name
    pvals_df = test_frmap_sig(fr_map,count,occu,dt)
    isfield=(pvals_df < alpha) & (count > min_count) & (fr_map > min_rate) 


    fr_map_peaks_tested={}
    for k,val in fr_map_peaks.iteritems():
        fr_map_peaks_tested[k]=[xx for xx in val if isfield.loc[k,xx]]
    fr_map_peaks_tested = pd.Series(fr_map_peaks_tested) # the test is quite conservative, and prune out small peaks
    max_nfields = fr_map_peaks_tested.apply(lambda x:len(x)).max()

    fr_map_peaks_tested_df = series_of_list_to_df(fr_map_peaks_tested,max_nfields)
    field_boundary_df = fr_map_peaks_tested_df.apply(get_field_boundary_for_each_field,axis=1,args=(isfield,)) # nneurons x nfieldsmax
    all_fields =field_boundary_df.stack(level=0).stack() # nfields_total,  (neurons, (start,end), nfields), no nan terms
    res = dict(
        fr_map_peaks_tested_df=fr_map_peaks_tested_df, # after the test; prune out some low fr field; nneurons x maxnfields
        fr_map_peaks = fr_map_peaks, # before the test; series: nneurons, each val is a list of peaks
        pvals_df = pvals_df, # nneurons x nposbins
        isfield = isfield, # nneurons x nposbins
        field_boundary_df = field_boundary_df,
        all_fields = all_fields
    )
    return res



def get_field_params_trial(fr_map_trial,all_fields):
    '''
    fr_map_trial: df
    all_fields: series, nfields_tot,  (neurons, (start,end), nfields), no nan terms; the boundaries for each field
    '''
    fr_l = {}
    fr_peak_l = {}
    center_l = {}
    peak_l = {}
    std_l = {}

    for ind,val in all_fields.groupby(level=(0,1)):
        
        fr_field = fr_map_trial.loc[ind[0]].loc[val.loc[ind]['start']:val.loc[ind]['end']]
        fr_field_by_trial = fr_field.mean(axis=0)
        fr_l[ind[:2]]=fr_field_by_trial
        fr_peak_l[ind[:2]] = fr_field.max(axis=0)
        center_by_trial = np.sum((fr_field.values * np.array(fr_field.index)[:,None]) ,axis=0) / np.sum(fr_field,axis=0)
        center_l[ind[:2]]=center_by_trial
        peak_by_trial = fr_field.idxmax(axis=0)
        peak_l[ind[:2]] = peak_by_trial
        std_by_trial = np.sum(fr_field.values * (np.array(fr_field.index)[:,None] - center_by_trial.values[None,:])**2 ,axis=0) / np.sum(fr_field,axis=0)
        std_by_trial = np.sqrt(std_by_trial)
        std_l[ind[:2]] =std_by_trial
    res_ = {'fr_mean':fr_l,'fr_peak':fr_peak_l,'peak':peak_l,'com':center_l,'std':std_l}
    res_df = {}
    for k,val in res_.items():
        res_df[k] = pd.concat(val,axis=1).T
    res_df = pd.concat(res_df,axis=0)
    return res_df



def find_closest_neurons(fr_map_peaks_tested_df,uid=None,posbin_l=None,nselected=10,distthresh=None):
    '''
    find neurons that have close field peaks to either a neuron, or a posbin or list of posbin
    '''
    if uid is not None:
        peaks = fr_map_peaks_tested_df.loc[uid]
        peaks_notna = peaks.loc[peaks.notna()]
        if len(peaks_notna)>0:
            posbin_l = peaks_notna.values
        else:
            return {}
    dist_sorted_d={}
    selected_d = {}
    if isinstance(posbin_l,float) or isinstance(posbin_l,int):
        posbin_l = [posbin_l]
    for posbin in posbin_l:
        dist_sorted=np.abs(fr_map_peaks_tested_df - posbin).min(axis=1).sort_values()
        dist_sorted_d[posbin] = dist_sorted
        if nselected is not None:
            selected_d[posbin] = dist_sorted.index[:nselected]
        elif distthresh is not None:
            selected_d[posbin] = dist_sorted.loc[dist_sorted<distthresh].index
            
    return selected_d,dist_sorted_d

def get_trial_ep(beh_df,trial_type=None,pos_range=None,pos_key='lin_binned',trial_index_range=None,trial_index_within_trialtype_range=None):
    '''
    get the IntervalSets for a range of trial indices, each trial gives one interval
    if trial_type is not None, thne subselect beh_df
    all *range: (2,)
    pos_range: limit to when the animal is within this position range; the position variable uses pos_key, lin or lin_binned
    trial_index_range: limit to the range of trial_index (absolute)
    trial_index_within_trialtype_range: limit to the range of trial_index (relative, within this trialtype);
    only one of trial_index_range and trial_index_within_trialtype_range can be given; at least one should be
    '''
    if (trial_index_range is not None) and (trial_index_within_trialtype_range is not None):
        print('trial_index_range and trial_index_within_trialtype_range cannot both be given')
        return
    
    if (trial_index_range is None) and (trial_index_within_trialtype_range is None):
        print('trial_index_range and trial_index_within_trialtype_range cannot both be None')
        return
    
    
    if trial_type is not None:
        beh_df_sub = beh_df.loc[beh_df['trial_type']==trial_type]
    else:
        beh_df_sub = beh_df
    gpb = beh_df_sub.groupby('trial')
    start=[]
    end=[]
    for ii,(k,val) in enumerate(gpb):
        if trial_index_range is not None:
            tocompare_ind = k
            tocomapre_range=trial_index_range
        elif trial_index_within_trialtype_range is not None:
            tocompare_ind = ii
            tocomapre_range=trial_index_within_trialtype_range
            
        if (tocompare_ind >= tocomapre_range[0]) & (tocompare_ind <= tocomapre_range[1]):
            if pos_range is not None:
                ma = (val[pos_key]>=pos_range[0]) & (val[pos_key]<=pos_range[1])
                start.append(val.loc[ma].index[0])
                end.append(val.loc[ma].index[-1]) 
            else:
                start.append(val.index[0])
                end.append(val.index[-1])
            
    ep = nap.IntervalSet(start=start,end=end)
    return ep
        
    
def fr_map_trial_to_df(fr_map_trial,cell_cols):
    '''
    fr_map_trial: nneurons x nposbins x ntrials
    # convert the tensor to df; (nneurons x nposbins) x ntrials
    convert the tensor to df; (nneurons x ntrials) x nposbins
    '''
    tensor=fr_map_trial
    posbin_l = np.arange(tensor.shape[1])
    index=pd.MultiIndex.from_product([cell_cols,posbin_l])
    fr_map_trial_df = pd.DataFrame(tensor.reshape(-1,tensor.shape[-1]),index=index)
    # fr_map_trial_df= fr_map_trial_df.stack().swaplevel(1,2).unstack()
    return fr_map_trial_df

# field detection by trial, using shuffled null
def get_fr_map_shuffle(spk_beh_df,cell_cols,nrepeats=100,dosave=False,save_fn='fr_map_null_trialtype.p',speed_thresh=1,nbins=100,smth_in_bin=2.5,bin_size=2.2,speed_key='v'):
    '''
    now the shuffle is done on the basis of a trial type; 
    '''
    dd,spk_beh_df=dpp.group_into_trialtype(spk_beh_df)
    # roll_inds = np.random.randint(50,spk_beh_df.shape[0]-50,size=nrepeats)
    # beh_part=spk_beh_df[['lin','trial_type','speed']]
    fr_map_null_d_all = {k:{} for k in dd.keys()}
    for tt, sbdf in dd.items():
        roll_inds = np.random.randint(50,sbdf.shape[0]-50,size=nrepeats)
        # beh_part=sbdf[['lin','v','speed','lin_binned']] 
        beh_part=sbdf[['lin',speed_key,'lin_binned','time']]  # important to add time here, otherwise dt is messed up!!
        for n in tqdm.tqdm(range(nrepeats)):
            i = roll_inds[n]
            # pdb.set_trace()
            spk_df_roll =pd.DataFrame(np.roll(sbdf[cell_cols],i,axis=0),columns=cell_cols,index=sbdf.index)
            spk_df_roll = pd.concat([spk_df_roll,beh_part],axis=1)
            # fr_map_null_d,_,_ = ratemap_from_spk_beh_df_alltrialtype(spk_df_roll,cell_cols,speed_thresh=speed_thresh,nbins=nbins,smth_in_bin=smth_in_bin)
            fr_map_null,_,_ = ratemap_from_spk_beh_df_onetrialtype(spk_df_roll,cell_cols,speed_thresh=speed_thresh,nbins=nbins,bin_size=bin_size,smth_in_bin=smth_in_bin,speed_key=speed_key)
        # for k,val in fr_map_null_d.items():
            fr_map_null_d_all[tt][n]=fr_map_null
    fr_map_null_df_d = {}
    for k,val in fr_map_null_d_all.items():
        fr_map_null_df_d[k] = pd.concat(val)
    if dosave:
        pickle.dump(fr_map_null_df_d,open(save_fn,'wb'))
        print(f'saved at {save_fn}')
    return fr_map_null_df_d

import misc
def get_fr_map_shuffle_wrapper(data_dir_full,nrepeats=1000, dosave=True,force_reload=False,bin_size=2.2,nbins = 100, save_fn='fr_map_null_trialtype.p',speed_thresh=1.,speed_key='v'):
    
    save_dir =misc.get_or_create_subdir(data_dir_full,'py_data')
    save_fn, res = misc.get_res(save_dir,save_fn,force_reload)
    if res is not None:
        return res

    prep_res = dpp.load_spk_beh_df(data_dir_full,force_reload=False,extra_load={})
    spk_beh_df=prep_res['spk_beh_df']
    cell_cols_d = prep_res['cell_cols_d']
    # beh_df = prep_res['beh_df'].as_dataframe()
    beh_df = prep_res['beh_df']
    beh_df_d,beh_df=dpp.group_into_trialtype(beh_df)
    spike_trains = prep_res['spike_trains']

    cell_cols = cell_cols_d['pyr']
    # cell_cols = np.concatenate(list(cell_cols_d.values()))# change! such that no longer just pyr are detected. filter afterwards
    
    spk_beh_df,lin_bins = dpp.add_lin_binned(spk_beh_df,nbins=nbins,bin_size=bin_size)

    

    res = get_fr_map_shuffle(spk_beh_df,cell_cols,nrepeats=nrepeats,dosave=dosave,save_fn=save_fn,speed_key=speed_key,speed_thresh=speed_thresh)
    # misc.save_res(save_fn,res,dosave)
    return res


# def get_field_by_trial_shuffle(fr_map_trial,count_trial,occu_trial,cell_cols,nrepeats = 1000,alpha=0.05):
#     '''

#     '''
#     count_trial_df = fr_map_trial_to_df(count_trial,cell_cols)
#     fr_map_trial_df = fr_map_trial_to_df(fr_map_trial,cell_cols)
#     fr_map_trial_df  = fr_map_trial_df.unstack() # (ntrial x nposbins)


#     count_trial_long = count_trial_df.stack().swaplevel(1,2).unstack().stack()
#     occu_trial_long = occu_trial.T.reshape(-1,1)
#     nbins = occu_trial_long.shape[0]
#     roll_inds = np.random.randint(nbins,size=(nrepeats))
#     count_roll = count_trial_long.groupby(level=0).apply(lambda x:np.array([np.roll(x,i,axis=0) for i in roll_inds]))
    
    

#     columns =fr_map_trial_df.columns
#     index = pd.MultiIndex.from_product([cell_cols,np.arange(nrepeats)])   
#     count_roll=pd.DataFrame(np.concatenate(count_roll.values,axis=0),index=index,columns=columns)

#     count_roll = count_roll.groupby(level=1,axis=1).sum()

#     return count_roll

#========using coarse bin=========#

### unfinished
def get_field_peaks_one_trialtype_binned(X,min_fr = 1.,min_trials=3):
    '''
    X: fr_map_trial_df; (nneurons x nbins) x ntrials
    '''
    Xunstack=X.unstack(level=0)
    nbins = Xunstack.shape[0]
    Xunstack.loc[-1] = Xunstack.loc[1]
    Xunstack.loc[nbins] = Xunstack.loc[nbins-2]
    Xunstack = Xunstack.sort_index()
    new_bins_index= Xunstack.index
    peaks_per_trial_per_neuron = Xunstack.apply(lambda x:new_bins_index[scipy.signal.find_peaks(x,height=min_fr)[0]],axis=0)

    pass

#==========using shuffled map, get each trial, merge using clustering=====#

def detect_significant_segments(fr_map_null,fr_map_trial_df,**kwargs):
    '''
    return===
    all_fields_bounds: df: (n_neurons x n_fields) x [start, end, field_index, peak, com, trial]
    '''
    
    kwargs_ = {'p_thresh':0.95,'min_fr':1,'width_thresh_in_bin':(4,30)}
    kwargs_.update(kwargs)
    p_thresh = kwargs_['p_thresh'] 
    min_fr = kwargs_['min_fr'] 
    width_thresh_in_bin = kwargs_['width_thresh_in_bin']
    # pos_bins = fr_map_null.loc[0].columns # important!!!! THis might not be the same as range(100)
    pos_bins = fr_map_null.columns # important!!!! THis might not be the same as range(100)
    fr_map_trial_df = fr_map_trial_df.loc[(slice(None),pos_bins),:] # fr_map_trial fill in the gap; now need to get rid of them for comparing with the null
    sig_thresh_map = fr_map_null.groupby(level=1).quantile(p_thresh) # this will reorder sig_thresh_map!!! 
    
    # issig_trial = (fr_map_trial_df > sig_thresh_map.stack().values[:,None]) #& (fr_map_trial_df > min_fr)
    issig_trial = (fr_map_trial_df > sig_thresh_map.stack().loc[fr_map_trial_df.index].values[:,None]) #reorder sig_thresh_map to make it align with fr_map_trial_df  #& (fr_map_trial_df > min_fr) 

    # detect field using the significant chunks in each trial
    issig_trial_pos_cols =issig_trial.stack().unstack(level=1)
    ma = np.diff(issig_trial_pos_cols.astype(int),prepend=0,append=0)
    start_ind,start_pos=np.nonzero(ma==1)
    end_ind,end_pos=np.nonzero(ma==-1)[0],np.nonzero(ma==-1)[1] - 1 # -1, because in pd loc, the right end is included
    
    start_ind = issig_trial_pos_cols.index.values[start_ind]
    start_pos = pos_bins[start_pos]
    end_pos = pos_bins[end_pos]
    
    all_fields_bounds=pd.DataFrame(np.array([start_ind,start_pos,end_pos]).T,columns=['uid','start','end'])
    all_fields_bounds.index=pd.MultiIndex.from_tuples(all_fields_bounds['uid'])
    all_fields_bounds = all_fields_bounds.drop('uid',axis=1)

    field_index = all_fields_bounds.groupby(level=(0,1)).cumcount()
    all_fields_bounds['field_index'] = field_index

    # filter out segments too big or too small    
    width = all_fields_bounds['end']-all_fields_bounds['start']
    width_mask = (width >= width_thresh_in_bin[0]) & (width <= width_thresh_in_bin[1])

    all_fields_bounds = all_fields_bounds.loc[width_mask]
    all_fields_bounds['trial'] = all_fields_bounds.index.get_level_values(1)

    all_fields_bounds = get_peak_com_fr_from_all_fields_bounds(fr_map_trial_df,all_fields_bounds)
    all_fields_bounds = all_fields_bounds.loc[all_fields_bounds['fr_peak'] > min_fr]
    # pdb.set_trace()
    return all_fields_bounds, sig_thresh_map

import mode_cluster as mc
from importlib import reload
reload(mc)
Kde_Peak_Cluster = mc.Kde_Peak_Cluster

# cluster significant segments into fields
def cluster_field_one_neuron(fields_bounds_one_neuron,cluster_key='com',model=Kde_Peak_Cluster,model_kws={'bw_method':0.1,'allposbins':np.arange(100),'peak_dist_thresh':6}):
    '''
    for one neuron from all_fields_bounds, cluster the segments
    '''

    n_fields_new = len(fields_bounds_one_neuron['field_index'].unique())+1 # max number of fields per trial + 1

    if fields_bounds_one_neuron.shape[0]==1:
        return np.array([0])
    else:
        while n_fields_new > fields_bounds_one_neuron.shape[0]:
            n_fields_new = n_fields_new-1
        location_to_be_clustered=fields_bounds_one_neuron[cluster_key].values[:,None]
        mdl = model(n_components=n_fields_new,**model_kws)
        cl = mdl.fit_predict(location_to_be_clustered)

        cl = scipy.stats.rankdata(cl,method='dense')-1 # reindex

        return cl

def cluster_field_all_neurons(all_fields_bounds,cluster_key='com',model=Kde_Peak_Cluster,model_kws={'bw_method':0.1,'allposbins':np.arange(100),'peak_dist_thresh':6}):
    cl_all = all_fields_bounds.groupby(level=0).apply(lambda x:cluster_field_one_neuron(x,cluster_key,model,model_kws))
    all_fields_bounds['field_index_cl']=np.concatenate(cl_all.values)
    return all_fields_bounds

# interpolate across trials
def interploate_field_across_trials_one_neuron(fields_bounds_one_neuron,trial_inds,combine_key='field_index_cl'):
    '''
    within one trial, fields with the same combine_key would be combined
    '''
    # (n_trial x field_index_cl x [start,end]) x 1

    fields_bounds_one_neuron_merged=fields_bounds_one_neuron.groupby(['trial',combine_key]).apply(lambda x:pd.DataFrame({'start':[x['start'].min()],'end':[x['end'].max()]}).T) 
    
    # n_trial x ([start,end] x field_index_cl)
    fields_bounds_one_neuron_merged_pivot=fields_bounds_one_neuron_merged.unstack().droplevel(axis=1,level=0).unstack(level=1)
    
    # make sure all trials have the same indices
    fields_bounds_one_neuron_merged_pivot=fields_bounds_one_neuron_merged_pivot.reindex(trial_inds)
    
    # before interpolation, keep a copy of which segments are significant
    # df: ntrials x nfields, binary
    presence = fields_bounds_one_neuron_merged_pivot.notna()['start'] 
    presence.columns = pd.MultiIndex.from_product([['presence'],presence.columns])

    fields_bounds_one_neuron_merged_pivot_interp=fields_bounds_one_neuron_merged_pivot.interpolate(limit_direction='both').astype(int)
    fields_bounds_one_neuron_merged_pivot_interp = pd.concat([fields_bounds_one_neuron_merged_pivot_interp,presence],axis=1)
    return fields_bounds_one_neuron_merged_pivot_interp
    
def interploate_field_across_trials_all_neuron(all_fields_bounds,trial_inds,combine_key='field_index_cl'):
    '''
    return both the pivot and unpivot version
    pivot: each field is aligned across trials; unpivot: easy to loop through all fields
    '''
    gpb = all_fields_bounds.groupby(level=0)
    all_fields_bounds_merged_pivot_interp={}
    # all_fields_presence={}
    for key, val in gpb:
        fields_bounds_one_neuron_merged_pivot_interp=interploate_field_across_trials_one_neuron(val,trial_inds,combine_key=combine_key)
        all_fields_bounds_merged_pivot_interp[key]=fields_bounds_one_neuron_merged_pivot_interp
    # all_fields_bounds_merged_pivot_interp, all_fields_presence = gpb.apply(lambda x:)
    all_fields_bounds_merged_pivot_interp = pd.concat(all_fields_bounds_merged_pivot_interp)
    # all_fields_presence = pd.concat(all_fields_presence)
    # all_fields_presence=all_fields_presence.stack().unstack(level=1) # (n_neurons x nfields) x n_trials
    all_fields_presence=all_fields_bounds_merged_pivot_interp['presence'].stack().unstack(level=1)
    all_fields_bounds_merged_interp = all_fields_bounds_merged_pivot_interp.stack(level=1)
    return all_fields_bounds_merged_pivot_interp, all_fields_bounds_merged_interp, all_fields_presence

def local_presence_test(all_fields_presence,win=5,ratio_min=0.6):
    local_presence_test_from_sig=all_fields_presence.astype(int).rolling(5,axis=1).apply(lambda x:x.mean()>=0.6)
    ma = local_presence_test_from_sig.any(axis=1)
    # all_fields_bounds_merged_interp_filtered = all_fields_bounds_merged_interp.groupby(level=1).apply(lambda x:x.loc[ma.values]).reset_index(level=0,drop=True)     
    return ma



# def get_peak_and_fr_mean(all_fields_bounds_merged_interp,fr_map_trial_df):
#     fr_peak_l=[]
#     fr_mean_l=[]
#     for ii, row in all_fields_bounds_merged_interp.iterrows():
#         uid=ii[0]
#         tt=ii[1]
#         fr_peak = fr_map_trial_df.loc[(uid,row['peak']),tt] 
#         fr_mean = fr_map_trial_df.loc[uid].loc[row['start']:row['end'],tt].mean()
#         fr_peak_l.append(fr_peak)
#         fr_mean_l.append(fr_mean)
#     all_fields_bounds_merged_interp['fr_peak'] = fr_peak_l
#     all_fields_bounds_merged_interp['fr_mean'] = fr_mean_l
#     return all_fields_bounds_merged_interp


# for getting peak, com, peak fr, mean fr, by iterating over all rows
# of some kind of all_fields_bounds df
def get_peak_com_fr_from_all_fields_bounds(fr_map_trial_df,all_fields_bounds):
    '''
    all_fields_bounds: df, (n_neurons x ntrials x...) x [start,end], rows containing all fields
    '''
    peak_l = []
    com_l = []
    fr_peak_l = []
    fr_mean_l = []
    for ii,row in all_fields_bounds.iterrows():
        uid=ii[0]
        tt=ii[1]
        peak = fr_map_trial_df.loc[(uid,slice(None))][tt].loc[row['start']:row['end']].idxmax()
        peak_l.append(peak)
        
        section = fr_map_trial_df.loc[(uid,slice(None))][tt].loc[row['start']:row['end']]
        inds = section.index
        com =na.get_com(section.values[:,None],axis=0)[0]

        com = inds[int(com)] if not np.isnan(com) else np.nan
        com_l.append(com)

        fr_peak = fr_map_trial_df.loc[(uid,peak),tt] 
        fr_mean = fr_map_trial_df.loc[uid].loc[row['start']:row['end'],tt].mean()
        fr_peak_l.append(fr_peak)
        fr_mean_l.append(fr_mean)

    all_fields_bounds['peak'] = peak_l
    all_fields_bounds['com'] = com_l
    all_fields_bounds['fr_peak'] = fr_peak_l
    all_fields_bounds['fr_mean'] = fr_mean_l
    
    return all_fields_bounds



# def check_presence_one_neuron(fields_bounds_one_neuron, min_n_trial=3):
#     '''
#     based on the field_index_cl; for each field, from when it first occur to when it last occur, what proportion of trials does it occur in 
#     min_n_trial: when computing the ratio, it's the max bewteen this and the span for the field, in case the span is too small
#     '''
#     gpb = fields_bounds_one_neuron.groupby('field_index_cl')
#     presence_ratio = gpb.apply(lambda val:val.shape[0] / np.maximum((val['trial'].max() - val['trial'].min() + 1), min_n_trial))
    
#     return presence_ratio

# COMBINE EVERYTHING TOGETHER
def field_detection_by_trial(fr_map_trial_d,fr_map_null,cell_cols,**kwargs):
    # nposbins = list(fr_map_trial_d.values())[0].shape[1]
    # allposbins = np.arange(nposbins)
    # allposbins = fr_map_null.columns
    kwargs_ = {'sig_detect_kws':{},'cluster_model':Kde_Peak_Cluster, 'cluster_key':'com',
    'cluster_model_kws':{'bw_method':0.1,'allposbins':None,'peak_dist_thresh':7},
    'combine_key':'field_index_cl',
    'presence_test_win':5,
    'presence_test_ratio':0.6,
    }
    kwargs_.update(kwargs)
    sig_detect_kws = kwargs_['sig_detect_kws']
    cluster_model_kws = kwargs_['cluster_model_kws']
    cluster_model = kwargs_['cluster_model']
    cluster_key = kwargs_['cluster_key']
    combine_key = kwargs_['combine_key']
    presence_test_win = kwargs_['presence_test_win']
    presence_test_ratio = kwargs_['presence_test_ratio']
    field_params_trial_d = {}
    field_params_median_d = {}
    field_params_trial_filtered_d = {}
    field_params_filtered_median_d = {}
    for key in fr_map_trial_d.keys():
        try:
            fr_map_trial_df =fr_map_trial_to_df(fr_map_trial_d[key],cell_cols)
            all_fields_bounds,sig_thesh_map_trial = detect_significant_segments(fr_map_null[key],fr_map_trial_df,**sig_detect_kws)
            # allposbins = fr_map_trial_df.index.get_level_values(1).unique()
            allposbins = fr_map_null[key].columns.values 
            cluster_model_kws['allposbins'] = allposbins
            all_fields_bounds = cluster_field_all_neurons(all_fields_bounds,cluster_key=cluster_key,model=cluster_model,model_kws=cluster_model_kws)
            trial_inds = fr_map_trial_df.columns
            all_fields_bounds_merged_pivot_interp, all_fields_bounds_merged_interp, all_fields_presence = interploate_field_across_trials_all_neuron(all_fields_bounds,trial_inds,combine_key=combine_key)
            field_params_trial = get_peak_com_fr_from_all_fields_bounds(fr_map_trial_df,all_fields_bounds_merged_interp)
            # shifting to df:([peak_fr, com...] x n_neuron x n_field) x ntrials
            field_params_trial = field_params_trial.stack().unstack(level=1).swaplevel(0,-1).swaplevel(1,-1)
            field_params_trial_d[key] = field_params_trial
            field_params_median_d[key] = field_params_trial.median(axis=1).unstack(level=0)

            ma = local_presence_test(all_fields_presence,win=presence_test_win,ratio_min=presence_test_ratio)
            field_params_trial_filtered = field_params_trial.groupby(level=0).apply(lambda x:x.loc[ma.values].reset_index(level=0,drop=True))
            field_params_trial_filtered_d[key] = field_params_trial_filtered
            field_params_filtered_median_d[key] = field_params_trial_filtered.median(axis=1).unstack(level=0)

        except Exception as e:
            print(f"{key} doesn't work")    
            print(e)
    return field_params_trial_d, field_params_trial_filtered_d,field_params_median_d,field_params_filtered_median_d, kwargs_


####### shuffle test but for trial averaged map ######
def get_field_params_trial_wrapper(fr_map_trial_df,all_fields_bounds_avg):
    '''
    wrapper for field bounds computed from shuffle test, to convert it to the format acceptible for get_field_params_trial
    '''
    
    df=all_fields_bounds_avg.reset_index(level=1,drop=True)
    inds=pd.MultiIndex.from_arrays([df.index,df['field_index']])
    df.index = inds
    df = df[['start','end','com','peak','fr_peak','fr_mean']]
    all_fields = df.stack()
    res_df=get_field_params_trial(fr_map_trial_df,all_fields)
    return res_df, all_fields

def field_detection_from_avg(fr_map_d,fr_map_trial_d,fr_map_null,**kwargs):
    '''
    all inputs: dict, trial_type:
    fr_map_d: df n_neurons x n_posbins
    fr_map_null: df (n_shuffle x n_neurons) x n_posbins
    fr_map_trial_d: array, n_neuron x n_posbins x n_trials
    kwargs: see detect_significant_segments
    '''
    field_params_trial_avgmap_d = {}
    all_fields_d = {}
    sig_thresh_map_d = {}
    for key in fr_map_d.keys():
        try:
            
            cell_cols = fr_map_d[key].index
            fr_map_trial = fr_map_trial_d[key]
            fr_map_trial_df=fr_map_trial_to_df(fr_map_trial,cell_cols)
            all_fields_bounds_avg, sig_thresh_map=detect_significant_segments(fr_map_null[key],fr_map_d[key].stack().to_frame(),**kwargs)
            field_params_trial_avgmap, all_fields = get_field_params_trial_wrapper(fr_map_trial_df,all_fields_bounds_avg)
            field_params_trial_avgmap_d[key] = field_params_trial_avgmap
            all_fields_d[key] = all_fields.unstack() # allfields x [start,end,com,peak,fr_peak,fr_mean]
            sig_thresh_map_d[key] = sig_thresh_map
        except Exception as e:
            print(f"{key} doesn't work: {e}")

    return field_params_trial_avgmap_d, all_fields_d, sig_thresh_map_d


####### FIELD DETECTION JUST AVG #######
def field_detection_avg_wrapper(data_dir_full, dosave=True,force_reload=False,nbins = 100, bin_size=2.2,
                                        save_fn = 'place_field_avg_and_trial.p', 
                                        shuffle_fn='fr_map_null_trialtype.p',
                                        smth_in_bin=2.5, speed_thresh=1.,speed_key='v',load_only=False,
                                        shuffle_force_reload=False,
                                        ):
    # deal with force reload
    save_dir =misc.get_or_create_subdir(data_dir_full,'py_data')
    save_fn, res = misc.get_res(save_dir,save_fn,force_reload)
    if (res is not None) or load_only:
        return res
    prep_res = dpp.load_spk_beh_df(data_dir_full,force_reload=False,extra_load={})
    spk_beh_df=prep_res['spk_beh_df']
    _,spk_beh_df=dpp.group_into_trialtype(spk_beh_df)
    cell_cols_d = prep_res['cell_cols_d']
    # beh_df = prep_res['beh_df'].as_dataframe()
    beh_df = prep_res['beh_df']
    beh_df_d,beh_df=dpp.group_into_trialtype(beh_df)
    spike_trains = prep_res['spike_trains']
    cell_cols = cell_cols_d['pyr'] 
    # cell_cols = np.concatenate(list(cell_cols_d.values()))# change! such that no longer just pyr are detected. filter afterwards
    
    spk_beh_df,lin_bins = dpp.add_lin_binned(spk_beh_df,nbins=nbins,bin_size=bin_size)
    # fr maps
    fr_map_null = get_fr_map_shuffle_wrapper(data_dir_full,nrepeats=1000, dosave=True,force_reload=shuffle_force_reload,bin_size=bin_size,nbins = nbins, save_fn=shuffle_fn,speed_key=speed_key,speed_thresh=speed_thresh)
    fr_map_dict=pa.get_fr_map_trial(spk_beh_df,cell_cols,trialtype_key='trial_type',speed_thresh=speed_thresh,bin_size=bin_size,order=['smooth','divide'],speed_key=speed_key)
    fr_map_trial_d = {k:val[0] for k,val in fr_map_dict.items()}
    fr_map_d,count_d,occu_d = ratemap_from_spk_beh_df_alltrialtype(spk_beh_df,cell_cols,speed_thresh=speed_thresh,nbins=nbins,bin_size=bin_size,smth_in_bin=smth_in_bin,speed_key=speed_key)
    
    field_params_trial_avgmap_d, all_fields_d, sig_thresh_map_d = field_detection_from_avg(fr_map_d,fr_map_trial_d,fr_map_null)
    
    res_d = {}
    res_d['params'] = field_params_trial_avgmap_d
    res_d['all_fields'] = all_fields_d
    res_d['sig_thresh_map'] = sig_thresh_map_d
    
    # for tt in fr_map_trial_d.keys():
    #     try:
    #         res = {'avg':{},'trial':{},'trial_filter':{}}
    #         res['avg']['params'] = field_params_trial_avgmap_d[tt]
    #         res['avg']['all_fields'] = all_fields_d[tt]
    #         res['trial']['params'] = field_params_trial_d[tt]
    #         field_params_median_d,field_params_filtered_median_d
    #         res['trial_filter']['params'] = field_params_trial_filtered_d[tt]
    #         res_d[tt] = res
    #     except:
    #         print(f"{tt} does not work in wrapper")
    

    misc.save_res(save_fn,res_d,dosave)
    return res_d

####### FIELD DETECTION ALL WRAPPER ######
def field_detection_both_avg_trial_wrapper(data_dir_full, dosave=True,force_reload=False,nbins = 100, bin_size=2.2,
                                        save_fn = 'place_field_avg_and_trial.p', 
                                        shuffle_fn='fr_map_null_trialtype.p',
                                        smth_in_bin=2.5, speed_thresh=1.,speed_key='v',load_only=False
                                        ):
    # deal with force reload
    save_dir =misc.get_or_create_subdir(data_dir_full,'py_data')
    save_fn, res = misc.get_res(save_dir,save_fn,force_reload)
    if (res is not None) or load_only:
        return res
    prep_res = dpp.load_spk_beh_df(data_dir_full,force_reload=False,extra_load={})
    spk_beh_df=prep_res['spk_beh_df']
    _,spk_beh_df=dpp.group_into_trialtype(spk_beh_df)
    cell_cols_d = prep_res['cell_cols_d']
    # beh_df = prep_res['beh_df'].as_dataframe()
    beh_df = prep_res['beh_df']
    beh_df_d,beh_df=dpp.group_into_trialtype(beh_df)
    spike_trains = prep_res['spike_trains']
    # cell_cols = cell_cols_d['pyr'] 
    cell_cols = np.concatenate(list(cell_cols_d.values()))# change! such that no longer just pyr are detected. filter afterwards
    
    
    spk_beh_df,lin_bins = dpp.add_lin_binned(spk_beh_df,nbins=nbins,bin_size=bin_size)
    # fr maps
    fr_map_null = get_fr_map_shuffle_wrapper(data_dir_full,nrepeats=1000, dosave=True,force_reload=False,bin_size=bin_size,nbins = nbins, save_fn=shuffle_fn,speed_key=speed_key)
    fr_map_dict=pa.get_fr_map_trial(spk_beh_df,cell_cols,trialtype_key='trial_type',speed_thresh=speed_thresh,bin_size=bin_size,order=['smooth','divide'],speed_key=speed_key)
    fr_map_trial_d = {k:val[0] for k,val in fr_map_dict.items()}
    fr_map_d,count_d,occu_d = ratemap_from_spk_beh_df_alltrialtype(spk_beh_df,cell_cols,speed_thresh=speed_thresh,nbins=nbins,bin_size=bin_size,smth_in_bin=smth_in_bin,speed_key=speed_key)
    
    field_params_trial_avgmap_d, all_fields_d, sig_thresh_map_d = field_detection_from_avg(fr_map_d,fr_map_trial_d,fr_map_null)
    field_params_trial_d, field_params_trial_filtered_d,field_params_median_d,field_params_filtered_median_d, trial_detection_kwargs = field_detection_by_trial(fr_map_trial_d,fr_map_null,cell_cols)
    res_d = {'avg':{},'trial':{},'trial_filter':{}}
    res_d['avg']['params'] = field_params_trial_avgmap_d
    res_d['avg']['all_fields'] = all_fields_d
    res_d['trial']['params'] = field_params_trial_d
    res_d['trial']['all_fields'] = field_params_median_d
    res_d['trial_filter']['params'] = field_params_trial_filtered_d
    res_d['trial_filter']['all_fields'] = field_params_filtered_median_d
    
    # for tt in fr_map_trial_d.keys():
    #     try:
    #         res = {'avg':{},'trial':{},'trial_filter':{}}
    #         res['avg']['params'] = field_params_trial_avgmap_d[tt]
    #         res['avg']['all_fields'] = all_fields_d[tt]
    #         res['trial']['params'] = field_params_trial_d[tt]
    #         field_params_median_d,field_params_filtered_median_d
    #         res['trial_filter']['params'] = field_params_trial_filtered_d[tt]
    #         res_d[tt] = res
    #     except:
    #         print(f"{tt} does not work in wrapper")
    res_d['trial_detection_kwargs'] = trial_detection_kwargs

    misc.save_res(save_fn,res_d,dosave)
    return res_d

