import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import sys,os,copy,pdb,importlib,pickle
from importlib import reload
import pandas as pd

sys.path.append('/mnt/home/szheng/projects/place_variability/code')
sys.path.append('/mnt/home/szheng/projects/util_code')
import traceback
import misc
import database

def get_contiguous_chunks(series,thresh):
    ma = series > thresh
    inds=np.nonzero(np.diff(ma,prepend=0,append=0))[0]
    inds=inds.reshape(-1,2)
    inds[:,1] = inds[:,1]-1 # end -1, correct for loc, for index has to add 1
    inds=pd.DataFrame(inds,columns=['start','end'])
    return inds

def get_frmap_shuffle(X_df_field,n_min_roll = 10,nrepeats = 1000,up_quantile=0.8,low_quantile=0.1):
    '''
    mean actually median
    '''
    X_df_field_avg= X_df_field.median(axis=1)

    npos = X_df_field.shape[0]
    X_df_field_v = X_df_field.values.reshape(1,-1,order='F')
    nbins_tot = X_df_field.shape[0] * X_df_field.shape[1]
    # pdb.set_trace()
    
    tt=np.random.randint(n_min_roll,nbins_tot-n_min_roll, size=(nrepeats))
    X_v_shuffle=np.stack([np.roll(X_df_field_v,t).reshape(npos,-1,order='F') for t in tt])
    X_v_avg_shuffle=np.median(X_v_shuffle,axis=-1) # shuffle avg across trials
    X_v_avg_shuffle_mean = np.median(X_v_avg_shuffle,axis=0)
    X_v_avg_shuffle_up = np.quantile(X_v_avg_shuffle,up_quantile,axis=0)
    X_v_avg_shuffle_low = np.quantile(X_v_avg_shuffle,low_quantile,axis=0)
    return X_df_field_avg,X_v_avg_shuffle_mean,X_v_avg_shuffle_up,X_v_avg_shuffle_low

def get_window_outside(field_bounds_final,in_field_mask=None,outfield_frac_pos = 0.1,
                    outfield_frac_size_thresh = 0.05,
                        **kwargs):
    '''
    # get window outside of field, and an outfield mask based on that
    '''
    if in_field_mask is not None:
        npos = in_field_mask.shape[0]
    else:
        npos = kwargs.get('npos',99)
    outfield_size_thresh = int(npos * outfield_frac_size_thresh)
    npos_out = int(npos * outfield_frac_pos)
    n_field = field_bounds_final.shape[0]
#     width = field_bounds_final['end']-field_bounds_final['start']
#     npos_out_half_width = width//2
#     to_extend=np.minimum(npos_out_half_width,npos_out)
    to_extend=npos_out
    field_bounds_final['window_start'] = np.maximum(field_bounds_final['start'] - to_extend,0)
    field_bounds_final['window_end'] = np.minimum(field_bounds_final['end'] + to_extend,npos-1)
    if in_field_mask is not None:
        out_field_mask = pd.Series(False,index=in_field_mask.index)
    ma  = []
    for i,row in field_bounds_final.iterrows():
        if i>0:
            prev_end = field_bounds_final.loc[i-1,'end']
            window_start=field_bounds_final.loc[i,'window_start']
            if window_start < prev_end:
                window_start = prev_end
                field_bounds_final.loc[i,'window_start']=window_start
        if i< (n_field-1):
            next_start = field_bounds_final.loc[i+1,'start']
            window_end=field_bounds_final.loc[i,'window_end']
            if window_end > next_start:
                window_end = next_start
                field_bounds_final.loc[i,'window_end']=window_end
        out_of_field_size = row['window_end']-row['end'] + row['start']-row['window_start']
        if out_of_field_size >= outfield_size_thresh: # if not too small, append row, set out field mask to be True
            ma.append(True)
            if in_field_mask is not None:
                out_field_mask.loc[row['window_start']:row['start']-1] = True
                out_field_mask.loc[row['end']+1:row['window_end']] = True
        else: # otherwise, set in_field_mask to be False
            if in_field_mask is not None:
                in_field_mask.loc[row['start']:row['end']]=False
            ma.append(False)
    ma = np.array(ma)

    # for i,row in field_bounds_final.iterrows():
    #     out_field_mask.loc[row['window_start']:row['start']-1] = True
    #     out_field_mask.loc[row['end']+1:row['window_end']] = True

    # field_bounds_final = pd.concat(row_l,axis=0).reset_index(drop=True)

    # out_of_field_size=(field_bounds_final['window_end'] - field_bounds_final['end']) + (field_bounds_final['start'] - field_bounds_final['window_start'])
    # ma = out_of_field_size > outfield_size_thresh
    field_bounds_final = field_bounds_final.loc[ma]
    if in_field_mask is not None:
        return field_bounds_final, in_field_mask, out_field_mask 
    else:
        return field_bounds_final

# ============detect field=================#
# one day, one cell #
def detect_field_using_contiguous_region_with_criteria(X_df_field,baseline_quantile=0.25,max_baseline_diff_frac=0.25,
                                                       highest=None,baseline=None,
                                                       mean_or_median_across_trial='median',
                                                       frac_pooled_thresh=0.6,
                                        min_width=4,max_width=33,in_out_ratio=3.,frac_sig_transient=0.2,
                                        n_min_roll = 10,nrepeats =1000,up_quantile=0.8,low_quantile=0.1,
                                                       outfield_frac_pos = 0.1,
                                                outfield_frac_size_thresh = 0.05,do_get_window_outside=True,
                                                       **kwargs
                                       ):
    # check the active cell criterion
            
    if (X_df_field>0.0).mean().mean() <= frac_sig_transient:
        
        return

    X_df_field_avg,X_v_avg_shuffle_mean,X_v_avg_shuffle_up,X_v_avg_shuffle_low=\
        get_frmap_shuffle(X_df_field,n_min_roll = n_min_roll,nrepeats = nrepeats,up_quantile=up_quantile,low_quantile=low_quantile)
    
    # get contiguous chunks
    if mean_or_median_across_trial=='median': # more robust for single day
        X_df_field_avg= X_df_field.median(axis=1)
    elif mean_or_median_across_trial=='mean': # more sensitive for multiple days
        X_df_field_avg= X_df_field.mean(axis=1)

    npos = X_df_field_avg.shape[0]
    if highest is None: # can be inherited from all days
        highest=X_df_field_avg.max()
    if baseline is None:
        baseline=X_df_field_avg.quantile(baseline_quantile)
    thresh_local =max_baseline_diff_frac * (X_df_field_avg.max() - X_df_field_avg.quantile(baseline_quantile)) # threshold from current sessions
    thresh = max_baseline_diff_frac*(highest-baseline)
    thresh = np.maximum(thresh * frac_pooled_thresh, thresh_local) # in case one day has really large value, use the largest among local and a scale-downed version of pooled threshold
    in_field_mask = X_df_field_avg > thresh
    field_bounds_tentative = get_contiguous_chunks(X_df_field_avg,thresh)
    X_out_field = X_df_field_avg.loc[X_df_field_avg <= thresh]
    X_out_field_mean = X_out_field.mean()
    
    # discard small field and big field
    field_size = field_bounds_tentative['end']-    field_bounds_tentative['start']
    ma =  (field_size> min_width) & (field_size < max_width)
    field_bounds_tentative = field_bounds_tentative.loc[ma]
    # pdb.set_trace()
    
    # check the in field gain criterion
    ma=[]
    for i,row in field_bounds_tentative.iterrows():
        within = X_df_field_avg.loc[row['start']:row['end']].mean()
        satisfy_inout = (within / (X_out_field_mean + 1e-10)) > in_out_ratio
        if not satisfy_inout:
            in_field_mask.loc[row['start']:row['end']]=False
        ma.append(satisfy_inout)
    ma=np.array(ma)
    field_bounds_tentative_afterinout = field_bounds_tentative.loc[ma]
        
    # check peak above null
    ma=[]
    for i,row in field_bounds_tentative_afterinout.iterrows():
        st,ed=row['start'],row['end']
        satisfy_shuffle_peak = (X_df_field_avg.loc[st:ed] > X_v_avg_shuffle_up[st:ed+1]).sum()>0
        if not satisfy_shuffle_peak:
            in_field_mask.loc[row['start']:row['end']]=False
        ma.append(satisfy_shuffle_peak)
    field_bounds_final = field_bounds_tentative_afterinout.loc[ma].reset_index(drop=True)
    
#     out_field_mask = np.logical_not(in_field_mask)
    in_field_mask = pd.Series(np.zeros(X_df_field_avg.shape[0],dtype=bool))
    for i,row in field_bounds_final.iterrows():
        in_field_mask.loc[row['start']:row['end']] = True
    if do_get_window_outside:
        field_bounds_final,in_field_mask, out_field_mask  = get_window_outside(field_bounds_final,in_field_mask,outfield_frac_pos = outfield_frac_pos,
                    outfield_frac_size_thresh = outfield_frac_size_thresh)
    else:
        out_field_mask = np.logical_not(in_field_mask)
    
    return field_bounds_final,in_field_mask,out_field_mask,X_df_field_avg,thresh,X_v_avg_shuffle_up

def get_within_field_fr(X_df,field_bounds_final):
    fr_within_field_across_trial_allfield={}
    for i,row in field_bounds_final.iterrows():
        fr_within_field_across_trial=X_df.loc[row['start']:row['end']].mean(axis=0)
        fr_within_field_across_trial_allfield[i] = fr_within_field_across_trial
    fr_within_field_across_trial_allfield = pd.concat(fr_within_field_across_trial_allfield,axis=1).T
    return fr_within_field_across_trial_allfield

# ALL day, one cell #
# modified to add an output of threshold; this will break other functions, need to check
def get_field_all_day_per_cell(X_df,pool_days_for_thresh=True,**kwargs):
    gpb = X_df.groupby(level=0,axis=1)
    mean_or_median_across_trial=kwargs.get('mean_or_median_across_trial','median')
    if mean_or_median_across_trial=='median':
        X_df_med = X_df.median(axis=1)
    elif mean_or_median_across_trial=='mean':
        X_df_med = X_df.mean(axis=1)
    field_bounds_final_allday={}
    in_field_mask_allday = {}
    out_field_mask_allday = {}
    baseline_quantile = kwargs.get('baseline_quantile',0.25)
    threshold_allday = {}
    if pool_days_for_thresh:
        highest=highest=gpb.median().max(axis=1).max() # use the max of all (medians within days) 
        baseline=X_df_med.quantile(baseline_quantile)
    else:
        highest=None
        baseline=None
        
    for k,val in gpb:        
        val=val.dropna(axis=0,how='all').dropna(axis=1,how='all').fillna(method='ffill',axis=0)
            
        assert val.isna().sum().sum()==0
        try:
            #*********
            field_bounds_final,in_field_mask,out_field_mask,X_df_field_avg,thresh,X_v_avg_shuffle_up = detect_field_using_contiguous_region_with_criteria(val,highest=highest,baseline=baseline,**kwargs) 
            field_bounds_final_allday[k] = field_bounds_final
            in_field_mask_allday[k] = in_field_mask
            out_field_mask_allday[k] = out_field_mask
            threshold_allday[k] = thresh
        except:
            pass
    try:
        field_bounds_final_allday = pd.concat(field_bounds_final_allday,axis=0)
        in_field_mask_allday = pd.concat(in_field_mask_allday,axis=0)
        out_field_mask_allday = pd.concat(out_field_mask_allday,axis=0)
        threshold_allday = pd.Series(threshold_allday)
        return field_bounds_final_allday, in_field_mask_allday, out_field_mask_allday, threshold_allday
    except:
        return

# def get_field_all_day_per_cell(X_df,pool_days_for_thresh=True,**kwargs):
#     gpb = X_df.groupby(level=0,axis=1)
    
#     X_df_med = X_df.median(axis=1)
#     field_bounds_final_allday={}
#     in_field_mask_allday = {}
#     out_field_mask_allday = {}
#     baseline_quantile = kwargs.get('baseline_quantile',0.25)
#     if pool_days_for_thresh:
#         highest=highest=gpb.median().max(axis=1).max() # use the max of all (medians within days) 
#         baseline=X_df_med.quantile(baseline_quantile)
#     else:
#         highest=None
#         baseline=None
        
#     for k,val in gpb:        
#         val=val.dropna(axis=0,how='all').dropna(axis=1,how='all').fillna(method='ffill',axis=0)
            
#         assert val.isna().sum().sum()==0
#         try:
#             #*********
#             field_bounds_final,in_field_mask,out_field_mask,X_df_field_avg,thresh,X_v_avg_shuffle_up = detect_field_using_contiguous_region_with_criteria(val,highest=highest,baseline=baseline,**kwargs) 
#             field_bounds_final_allday[k] = field_bounds_final
#             in_field_mask_allday[k] = in_field_mask
#             out_field_mask_allday[k] = out_field_mask
#         except:
#             pass
#     try:
#         field_bounds_final_allday = pd.concat(field_bounds_final_allday,axis=0)
#         in_field_mask_allday = pd.concat(in_field_mask_allday,axis=0)
#         out_field_mask_allday = pd.concat(out_field_mask_allday,axis=0)
#         return field_bounds_final_allday, in_field_mask_allday, out_field_mask_allday
#     except:
#         return 

# ALL day, ALL cell #
import tqdm
def get_field_all_day_all_cell(fr_map_trial_df_all_day_sub,pool_days_for_thresh=True,**kwargs):
    '''
    remember to change frac_sig_transient to 0.1 for DG; otherwise 0.2
    '''
    gpb = fr_map_trial_df_all_day_sub.groupby(level=(0,1)) # isnovel, uid
    field_bounds_all = {}
    in_field_mask_all = {}
    out_field_mask_all = {}
    thresh_all = {}
    for (k,val) in tqdm.tqdm(gpb):
        val = val.loc[k]
        #*******
        try:
            field_bounds_final_allday, in_field_mask_allday, out_field_mask_allday,thresh_allday = get_field_all_day_per_cell(val,pool_days_for_thresh=True,**kwargs) 
            field_bounds_all[k] = field_bounds_final_allday
            in_field_mask_all[k] = in_field_mask_allday
            out_field_mask_all[k] = out_field_mask_allday
            thresh_all[k] = thresh_allday
        except:
            pass
    field_bounds_all=pd.concat(field_bounds_all,axis=0)
    in_field_mask_all = pd.concat(in_field_mask_all,axis=0)
    out_field_mask_all = pd.concat(out_field_mask_all,axis=0)
    thresh_all = pd.concat(thresh_all,axis=0)
    
    return field_bounds_all, in_field_mask_all, out_field_mask_all, thresh_all



### reuse block analysis ###
# one cell one day#
def get_within_out_diff_one_cell_one_day(X_df_compare,in_field_mask,out_field_mask=None):
    pre_day_activation = {}
#     pdb.set_trace()
    common_pos=in_field_mask.index.intersection(X_df_compare.index).intersection(out_field_mask.index) # don't miss .index!!
    X_df_compare = X_df_compare.loc[common_pos]
    in_field_mask = in_field_mask.loc[common_pos]
    out_field_mask = out_field_mask.loc[common_pos]
    pre_day_activation['within'] = X_df_compare.loc[in_field_mask].mean()
    
    if out_field_mask is None:
        out_field_mask = np.logical_not(in_field_mask)
    pre_day_activation['outside'] = X_df_compare.loc[out_field_mask].mean()
    pre_day_activation['diff'] = pre_day_activation['within'] - pre_day_activation['outside']
    eps = 1e-10
    pre_day_activation['ratio'] = pre_day_activation['within'] / (pre_day_activation['outside'] + eps)
    pre_day_activation['diff_frac'] = (pre_day_activation['within']-pre_day_activation['outside']) / (pre_day_activation['outside'] + eps)
    test_res=scipy.stats.ranksums(pre_day_activation['within'],pre_day_activation['outside'])
    pre_day_activation = pd.concat(pre_day_activation,axis=0)
    pre_day_activation.index.rename(('type','day_pre','trial'),level=(0,1,2),inplace=True)
    return pre_day_activation,test_res
# one cell one day all fields seperate #
def get_within_out_diff_one_cell_all_field_one_day(X_df_compare,field_bounds_final):
    '''
    X_df_compare: prev days
    field_bounds_final: obtained using one day
    '''
    fr_within_field_across_trial_allfield={}
    fr_outside_field_across_trial_allfield = {}
    for i,row in field_bounds_final.iterrows():
        fr_within_field_across_trial=X_df_compare.loc[row['start']:row['end']].mean(axis=0)
        fr_within_field_across_trial_allfield[i] = fr_within_field_across_trial
        
        outside_sum=X_df_compare.loc[row['window_start']:(row['start']-1)].sum(axis=0) + X_df_compare.loc[(row['end']+1):row['window_end']].sum(axis=0)
        outside_size=(row['window_end'] - row['window_start'] + 1) - (row['end'] - row['start'] +1)
        fr_outside_field_across_trial_allfield[i] = outside_sum / outside_size
        
    fr_within_field_across_trial_allfield = pd.concat(fr_within_field_across_trial_allfield,axis=1).T
    fr_outside_field_across_trial_allfield = pd.concat(fr_outside_field_across_trial_allfield,axis=1).T
    fr_field_allfield=pd.concat({'within':fr_within_field_across_trial_allfield,'outside':fr_outside_field_across_trial_allfield,
               'diff':fr_within_field_across_trial_allfield - fr_outside_field_across_trial_allfield
              },axis=1)
    
    return fr_field_allfield
def get_prev_day_silent_mask_one_cell_all_field(fr_field_allfield,prev_day,threshold,field_not_present_frac = 0.7):
    '''

    '''
    field_not_present_per_trial = fr_field_allfield.loc[:,('within',prev_day)] < threshold
    field_not_present_ma = field_not_present_per_trial.mean(axis=1) >= field_not_present_frac
    return field_not_present_ma # for one cell, one day, all field, whether previous day it's within field FR is above threshold most of the trials or not




# one cell ALL day#
def get_within_out_diff_one_cell_all_day(X_df,in_field_mask_allday=None,out_field_mask_allday=None,n_day_pre=1,**kwargs):
    '''
    n_day_pre: if 1, get activation till the day before the getting field day; if 0, get activation including the getting field day
    '''
    if in_field_mask_allday is None:
        field_bounds_final_allday, in_field_mask_allday,out_field_mask_allday = get_field_all_day_per_cell(X_df,**kwargs)
    day_l = X_df.columns.get_level_values(0).unique()
    pre_day_activation_d={}
    test_res_d={}
    for d in day_l[1:]:
        get_field_day = d
        X_df_compare = X_df.loc[:,:(get_field_day-n_day_pre)].dropna(axis=1,how='all')
        if get_field_day in in_field_mask_allday.index.get_level_values(0):
            in_field_mask = in_field_mask_allday.loc[get_field_day]
            out_field_mask = out_field_mask_allday.loc[get_field_day]
            pre_day_activation,test_res = get_within_out_diff_one_cell_one_day(X_df_compare,in_field_mask,out_field_mask=out_field_mask)
            pre_day_activation_d[d] = pre_day_activation
            
            test_res_d[d] = test_res
    pre_day_activation_d = pd.concat(pre_day_activation_d,axis=0).dropna()
    pre_day_activation_d.index.rename('get_field_day',level=0,inplace=True)
    test_res_d=pd.DataFrame(test_res_d).T.dropna(axis=0,how='all') # drop the no field days
    test_res_d.index.rename('get_field_day')
    test_res_d.columns=['stat','pval']
    return pre_day_activation_d,test_res_d

# def get_within_out_diff_one_cell_all_day_all_field(X_df,field_bounds_final_allday=None,threshold_allday=None,field_not_present_frac = 0.7,n_day_pre=1,**kwargs):
#     '''
#     n_day_pre: if 1, get activation till the day before the getting field day; if 0, get activation including the getting field day
#     '''
#     if (field_bounds_final_allday is None) or (threshold_allday is None):
#         field_bounds_final_allday, in_field_mask_allday,out_field_mask_allday,threshold_allday = get_field_all_day_per_cell(X_df,**kwargs)
#     day_l = X_df.columns.get_level_values(0).unique()
#     pre_day_activation_d = {}
#     prev_day_silent_ma_d = {}
#     for d in day_l[2:]: # only look at at least 3 days, that way can have a prevoius silent day and at least one more day before
#         get_field_day = d
#         prev_day = d-1
#         threshold = threshold_allday.loc[prev_day] # either the threshold of the previous day, or some fraction of overall; anyway the threshold used for detecting field on the prev_day
#         X_df_compare = X_df.loc[:,:(get_field_day-n_day_pre)].dropna(axis=1,how='all')
#         if get_field_day in field_bounds_final_allday.index.get_level_values(0):
#             field_bounds_final = field_bounds_final_allday.loc[get_field_day]
#             fr_field_allfield = get_within_out_diff_one_cell_all_field_one_day(X_df_compare,field_bounds_final)
#             prev_day_silent_ma = get_prev_day_silent_mask_one_cell_all_field(fr_field_allfield,prev_day,threshold,field_not_present_frac = field_not_present_frac)
#             pre_day_activation_d[d] = fr_field_allfield
#             prev_day_silent_ma_d[d] = prev_day_silent_ma
#         else:
#             pre_day_activation_d[d] = pd.DataFrame(dtype=float)
#             prev_day_silent_ma_d[d] = pd.Series(dtype=bool)
#     pre_day_activation_d = pd.concat(pre_day_activation_d,axis=0).dropna()
#     pre_day_activation_d.index.rename('get_field_day',level=0,inplace=True)
#     prev_day_silent_ma_d = pd.concat(prev_day_silent_ma_d,axis=0)
#     prev_day_silent_ma_d.index.rename('get_field_day',level=0,inplace=True)
#     return pre_day_activation_d, prev_day_silent_ma_d

def get_within_out_diff_one_cell_all_day_all_field(X_df,field_bounds_final_allday=None,threshold_allday=None,**kwargs):
    '''
    get activation for all fields on all days
    '''
    if (field_bounds_final_allday is None) or (threshold_allday is None):
        field_bounds_final_allday, in_field_mask_allday,out_field_mask_allday,threshold_allday = get_field_all_day_per_cell(X_df,**kwargs)
    day_l = X_df.columns.get_level_values(0).unique()
    all_day_activation_d = {}
    
    for d in day_l: 
        get_field_day = d
        
        X_df_compare = X_df.dropna(axis=1,how='all')
        if get_field_day in field_bounds_final_allday.index.get_level_values(0):
            field_bounds_final = field_bounds_final_allday.loc[get_field_day]
            fr_field_allfield = get_within_out_diff_one_cell_all_field_one_day(X_df_compare,field_bounds_final)
            all_day_activation_d[d] = fr_field_allfield
        else:
            all_day_activation_d[d] = pd.DataFrame(dtype=float)
            
    all_day_activation_d = pd.concat(all_day_activation_d,axis=0).dropna()
    all_day_activation_d.index.rename('get_field_day',level=0,inplace=True)
    
    return all_day_activation_d



def get_sw_ma(fr_field_allfield,day,threshold,trial_range=5,field_not_present_frac=0.8,sw='on'):
    '''
    fr_field_allfield: 1 more levels than threshold, indicating field
    sw_ma: series: isnovel x uid x get_field_day x field, need to further filter to get sw on particular day
    '''
    pdb.set_trace()
    fr_field_allfield=fr_field_allfield.loc[:,('within',day)].dropna(axis=1,how='all')
    ind_orig = fr_field_allfield.index #original index
    mat = fr_field_allfield.unstack(-1) # stack the field dimension to fascilitate comparison with threshold, which doesnt have a field dimension
    ind = mat.index
    compare = (mat < threshold.loc[ind].values[:,None]).stack().loc[ind_orig]
    compare[np.isnan(fr_field_allfield).values] = np.nan # set the original nan values back to nan
    field_not_present_per_trial = compare
    
    if sw=='on':
        sw_ma = field_not_present_per_trial.iloc[:,:trial_range].mean(axis=1) >=field_not_present_frac
    elif sw=='off':
        sw_ma = field_not_present_per_trial.iloc[:,-trial_range:].mean(axis=1) >=field_not_present_frac
    return sw_ma, field_not_present_per_trial

def get_sw_ma_per_session(fr_field_allfield,day,threshold,trial_range=5,field_not_present_frac=0.8,sw='off'):
    '''
    use groupby first to do get_sw_ma for each session
    necessary for sw off, since the last few columns are padded nan, which would mess up the detection
    '''
    gpb = fr_field_allfield.groupby(level=(0,1,2))
    sw_off_ma_all = []
    field_not_present_all = []
    for k,val in gpb:
        val=val.dropna(axis=1,how='all')
        sw_off_ma_one,field_not_present_one=get_sw_ma(val,day,threshold,trial_range=trial_range,field_not_present_frac=field_not_present_frac,sw=sw)
        sw_off_ma_all.append(sw_off_ma_one)
        field_not_present_all.append(field_not_present_one)
    sw_off_ma_all = pd.concat(sw_off_ma_all,axis=0)
    field_not_present_all = pd.concat(field_not_present_all,axis=0)
    return sw_off_ma_all,field_not_present_all




# get field on one day, get activation as well as mask for popup and popdown
def get_within_out_diff_one_cell_two_day_all_field(X_df,check_sw_off_day=0,check_sw_on_day=1,field_bounds_final_allday=None,threshold_allday=None,field_not_present_frac = 0.7,n_day_pre=1,**kwargs):
    '''
    n_day_pre: if 1, get activation till the day before the getting field day; if 0, get activation including the getting field day
    '''
    if (field_bounds_final_allday is None) or (threshold_allday is None):
        field_bounds_final_allday, in_field_mask_allday,out_field_mask_allday,threshold_allday = get_field_all_day_per_cell(X_df,**kwargs)
    # day_l = X_df.columns.get_level_values(0).unique()
    pre_day_activation_d = {}
    prev_day_silent_ma_d = {}
    check_sw_day = {'off':check_sw_off_day,'on':check_sw_on_day}
    day_l = check_sw_day.values()
    # for d in day_l[:2]: # only look at first 2 days
    for sw,d in check_sw_day.items():
        get_field_day = d
        # prev_day = d-1
        threshold = threshold_allday.loc[get_field_day] # either the threshold of the previous day, or some fraction of overall; anyway the threshold used for detecting field on the prev_day
        X_df_compare = X_df.loc[:,day_l].dropna(axis=1,how='all')
        if get_field_day in field_bounds_final_allday.index.get_level_values(0):
            field_bounds_final = field_bounds_final_allday.loc[get_field_day]
            fr_field_allfield = get_within_out_diff_one_cell_all_field_one_day(X_df_compare,field_bounds_final)
            # get switch ma
            prev_day_silent_ma = get_prev_day_silent_mask_one_cell_all_field(fr_field_allfield,prev_day,threshold,field_not_present_frac = field_not_present_frac)
            pre_day_activation_d[d] = fr_field_allfield
            prev_day_silent_ma_d[d] = prev_day_silent_ma
        else:
            pre_day_activation_d[d] = pd.DataFrame(dtype=float)
            prev_day_silent_ma_d[d] = pd.Series(dtype=bool)
    pre_day_activation_d = pd.concat(pre_day_activation_d,axis=0).dropna()
    pre_day_activation_d.index.rename('get_field_day',level=0,inplace=True)
    prev_day_silent_ma_d = pd.concat(prev_day_silent_ma_d,axis=0)
    prev_day_silent_ma_d.index.rename('get_field_day',level=0,inplace=True)
    return pre_day_activation_d, prev_day_silent_ma_d



# ALL cell ALL day#
def get_within_out_diff_all_cell_all_day(fr_map_trial_df_all_day_sub,in_field_mask_all=None,out_field_mask_all=None,**kwargs):
    if in_field_mask_all is None:
        field_bounds_all, in_field_mask_all, out_field_mask_all = get_field_all_day_all_cell(fr_map_trial_df_all_day_sub,**kwargs)
    gpb = fr_map_trial_df_all_day_sub.groupby(level=(0,1)) # isnovel, uid
    pre_day_activation_d_all = {}
    test_res_d_all = {}
    for (k,val) in tqdm.tqdm(gpb):
        val = val.loc[k]
        in_field_mask_allday = in_field_mask_all.loc[k]
        out_field_mask_allday = out_field_mask_all.loc[k]
        #******
        pre_day_activation_d,test_res_d = get_within_out_diff_one_cell_all_day(val,in_field_mask_allday=in_field_mask_allday,out_field_mask_allday=out_field_mask_allday,**kwargs)
        pre_day_activation_d_all[k]=pre_day_activation_d
        test_res_d_all[k]=test_res_d
    pre_day_activation_d_all=pd.concat(pre_day_activation_d_all,axis=0)
    test_res_d_all=pd.concat(test_res_d_all,axis=0)
    return pre_day_activation_d_all, test_res_d_all

# def get_within_out_diff_all_cell_all_day_all_field(fr_map_trial_df_all_day_sub,field_bounds_all=None,threshold_all=None,field_not_present_frac = 0.7,**kwargs):
#     if (field_bounds_all is None) or (threshold_all is None):
#         field_bounds_all, in_field_mask_all, out_field_mask_all, threshold_all = get_field_all_day_all_cell(fr_map_trial_df_all_day_sub,**kwargs)
#     gpb = field_bounds_all.groupby(level=(0,1)) # isnovel, uid
#     pre_day_activation_d_all = {}
#     prev_day_silent_ma_d_all = {}
#     for (k,val) in tqdm.tqdm(gpb):
#         # print(k)
#         field_bounds_final_allday = val.loc[k]
#         # field_bounds_final_allday = field_bounds_all.loc[k]
#         X_df = fr_map_trial_df_all_day_sub.loc[k]
#         threshold_allday = threshold_all.loc[k]
#         pre_day_activation_d, prev_day_silent_ma_d = get_within_out_diff_one_cell_all_day_all_field(X_df,field_bounds_final_allday=field_bounds_final_allday,threshold_allday=threshold_allday,field_not_present_frac = field_not_present_frac,**kwargs)
#         pre_day_activation_d_all[k]=pre_day_activation_d
#         prev_day_silent_ma_d_all[k]=prev_day_silent_ma_d
#     pre_day_activation_d_all=pd.concat(pre_day_activation_d_all,axis=0)
#     prev_day_silent_ma_d_all=pd.concat(prev_day_silent_ma_d_all,axis=0)
#     return pre_day_activation_d_all, prev_day_silent_ma_d_all,field_bounds_all,threshold_all

def get_within_out_diff_all_cell_all_day_all_field(fr_map_trial_df_all_day_sub,field_bounds_all=None,threshold_all=None,**kwargs):
    if (field_bounds_all is None) or (threshold_all is None):
        field_bounds_all, in_field_mask_all, out_field_mask_all, threshold_all = get_field_all_day_all_cell(fr_map_trial_df_all_day_sub,**kwargs)
    gpb = field_bounds_all.groupby(level=(0,1)) # isnovel, uid
    pre_day_activation_d_all = {}
    prev_day_silent_ma_d_all = {}
    for (k,val) in tqdm.tqdm(gpb):
        # print(k)
        field_bounds_final_allday = val.loc[k]
        # field_bounds_final_allday = field_bounds_all.loc[k]
        X_df = fr_map_trial_df_all_day_sub.loc[k]
        threshold_allday = threshold_all.loc[k]
        pre_day_activation_d = get_within_out_diff_one_cell_all_day_all_field(X_df,field_bounds_final_allday=field_bounds_final_allday,threshold_allday=threshold_allday,**kwargs)
        pre_day_activation_d_all[k] = pre_day_activation_d
    pre_day_activation_d_all=pd.concat(pre_day_activation_d_all,axis=0)
    
    return pre_day_activation_d_all,field_bounds_all,in_field_mask_all, out_field_mask_all,threshold_all
        

    
    

#=======not used========#
def shuffle_test_frmap_get_peaks(X_df_field,n_min_roll = 10,nrepeats = 1000,up_quantile=0.8,low_quantile=0.1):
    '''
    X_df_field: npos x (nday x ntrial)
    it's all median despite the word mean
    get peaks that are also above some shuffle null 
    '''
    X_df_field_avg= X_df_field.median(axis=1)

    npos = X_df_field.shape[0]
    X_df_field_v = X_df_field.values.reshape(1,-1,order='F')
    nbins_tot = X_df_field_v.shape[0] * X_df_field_v.shape[1]
    
    tt=np.random.randint(n_min_roll,nbins_tot-n_min_roll, size=(nrepeats))
    X_v_shuffle=np.stack([np.roll(X_df_field_v,t).reshape(npos,-1,order='F') for t in tt])
    X_v_avg_shuffle=np.median(X_v_shuffle,axis=-1) # shuffle avg across trials
    X_v_avg_shuffle_mean = np.median(X_v_avg_shuffle,axis=0)
    X_v_avg_shuffle_up = np.quantile(X_v_avg_shuffle,up_quantile,axis=0)
    X_v_avg_shuffle_low = np.quantile(X_v_avg_shuffle,low_quantile,axis=0)
    
    issig = X_df_field_avg > X_v_avg_shuffle_up
    pdf_peaks = xs[scipy.signal.find_peaks(vals)[0]]
    sig_peaks = pdf_peaks[issig[pdf_peaks]]
    return sig_peaks,X_df_field_avg,X_v_avg_shuffle_mean,X_v_avg_shuffle_up,X_v_avg_shuffle_low



#==============switch trial detection=========#
def get_switch_trial(
    all_day_activation_sub,
    threshold_sub,
    sw_mag_thresh=0.4,
    trial_range = 5,
    field_present_frac = 0.6,
    field_not_present_frac=0.8,
    sw='on',
    ):
    # get switch on trial; criterion from pfdt.get_sw_ma (first 4 (80% of window=5) trials below thresh), 
    # in addition, find the first trial post the 5th, such that 3 trials within 5 trials are above thresh
    # in addition, the switch trial has to be 0.4 above the previous trial in normalized(by max within field) dF/F
    
    # for off trial: 
    # criterion from pfdt.get_sw_ma (last 4 (80% of window=5) trials below thresh), 
    # in addition, find the last trial before the 5th to last, such that 3 trials within 5 trials are below thresh
    # in addition, the switch trial has to be 0.4 below the previous trial in normalized(by max within field) dF/F
    
    all_day_activation_sub = all_day_activation_sub.dropna(axis=1,how='all')
    
    left,right=threshold_sub.align(all_day_activation_sub,axis=0,level=0)
    field_present_per_trial_sub=right>=left.values[:,None] # (n_neuron, n_field) x n_trial

    
    all_day_activation_sub_norm = all_day_activation_sub / all_day_activation_sub.max(axis=1).values[:,None]
    all_day_activation_sub_norm = all_day_activation_sub_norm.dropna(axis=1,how='all')
    field_present_per_trial_sub=field_present_per_trial_sub[all_day_activation_sub_norm.columns] # sneaky! field_present_per_trial_sub already replaced things with 0
    
    # number of trials where the field is present in the next window
    n_present_in_next_win= scipy.signal.convolve(field_present_per_trial_sub,np.ones(trial_range,dtype=int)[None,:],mode='full')[:,trial_range-1:]
    n_present_in_next_win = pd.DataFrame(n_present_in_next_win,index=field_present_per_trial_sub.index,columns=field_present_per_trial_sub.columns)
    
    sw_on_trial_l = {}
    for (uid,field_id),row in field_present_per_trial_sub.iterrows():
        activation_row=all_day_activation_sub_norm.loc[(uid,field_id),:]
        if sw=='on':
            ma_not_already_on = np.logical_not(row).iloc[:trial_range].sum()>=int(field_not_present_frac*trial_range)
            first_crit = ma_not_already_on
        elif sw=='off':
            # ma for if the field is indeed off at the end
            ma_off_end = np.logical_not(row).iloc[-trial_range:].sum()>=int(field_not_present_frac*trial_range)
            first_crit = ma_off_end
#         if ma_not_already_on:
        if first_crit:
            if sw=='on':
                ma_window=n_present_in_next_win.loc[(uid,field_id)]>=int(trial_range*field_present_frac)
                ma_magnitude = activation_row.diff()>=sw_mag_thresh
            elif sw=='off': # next window of trials, the number of trials above threshold should be below a number
                ma_window=n_present_in_next_win.loc[(uid,field_id)]<=int(trial_range*(1-field_present_frac))
                ma_magnitude = activation_row.diff()<=-sw_mag_thresh
            ma = np.logical_and(row.values,ma_window)
            ma=np.logical_and(ma,ma_magnitude)
            ind_above_thresh=np.nonzero(ma.values)[0]
            if len(ind_above_thresh)>0:
                if sw=='on':
                    first_above_thresh=ind_above_thresh[0]
                    if first_above_thresh>=trial_range:
                        sw_on_trial_l[(uid,field_id)] = first_above_thresh
                elif sw=='off': 
                    last_below_thresh=ind_above_thresh[-1]

                    if last_below_thresh<=all_day_activation_sub_norm.shape[1]-trial_range: 
                        sw_on_trial_l[(uid,field_id)] = last_below_thresh

    return sw_on_trial_l # or sw_off_trial_l
