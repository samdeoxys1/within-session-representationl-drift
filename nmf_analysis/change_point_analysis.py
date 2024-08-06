import ruptures as rpt
import numpy as np
import pandas as pd
import scipy
import os,sys,copy,itertools,pdb,importlib
import change_point_plot as cpp
importlib.reload(cpp)
import matplotlib.pyplot as plt
import seaborn as sns

def get_change_points_one_field(signal,pen,trial_index=None,model_type='pelt',cost='l2',min_size=2):
    if model_type =='pelt':
        model = rpt.Pelt(model=cost,jump=1,min_size=min_size)
    elif model_type=='dyn': # in this case pen=the number of change points
        model = rpt.Dynp(model=cost,jump=1,min_size=min_size)
    if isinstance(signal,pd.Series):
        signal = signal.values
    c = model.fit(signal)
    change_pts=np.array(model.predict(pen))[:-1] # get rid of the last one, which is just the length
    if trial_index is not None:
        change_pts = trial_index[change_pts]
    
    return change_pts
    
def get_change_points_all_fields(X_restacked_df_d,peak_inds_df_d=None,pen=0.2,model_type='pelt',cost='l2'):
    change_pts_population_d={}
    for key in X_restacked_df_d.keys():
        if peak_inds_df_d is None:
            peak_inds = X_restacked_df_d[key].index
        else:
            peak_inds = peak_inds_df_d[key].index
        X_restacked_df_peak_only = X_restacked_df_d[key].loc[peak_inds]
        change_pts_population = X_restacked_df_peak_only.apply(get_change_points_one_field, axis=1, args=(pen, X_restacked_df_peak_only.columns,model_type,cost))
        change_pts_population_d[key] = change_pts_population
    return change_pts_population_d

def predict_from_cpts(signal,cpts,return_var=False):
    '''
    recover a piecewise linear signal given the change points
    signal: np array
    cpts: include 0 and last; index in np array not pd
    '''
    signal_pred = np.zeros_like(signal)
    for ii in range(len(cpts)-1):
        signal_pred[cpts[ii]:cpts[ii+1]] = signal[cpts[ii]:cpts[ii+1]].mean()
    if return_var:
        signal_var = np.zeros_like(signal)
        for ii in range(len(cpts)-1):
            signal_var[cpts[ii]:cpts[ii+1]] = signal[cpts[ii]:cpts[ii+1]].var()
        return signal_pred,signal_var
    else:
        return signal_pred

def predict_from_cpts_wrapper(signal,ncpts,cost='l2',min_size=2,model_type=rpt.Dynp,return_var=False):
    model = model_type(model=cost,jump=1,min_size=min_size)
    c = model.fit(signal)
    cpts=model.predict(ncpts)
    cpts = np.array([0,*cpts])
    signal_pred=predict_from_cpts(signal,cpts,return_var=return_var)
    return signal_pred, cpts[1:-1] # return no first and last
from matplotlib.ticker import MaxNLocator
def predict_from_cpts_wrapper_plot(signal,ncpts,cost='l2',min_size=2,model_type=rpt.Dynp,fig=None,ax=None,flipy=False,
                                dolabel=False,dolegend=False
                                ):
    signal_pred, cpt = predict_from_cpts_wrapper(signal,ncpts,cost=cost,min_size=min_size,model_type=model_type)
    trials = np.arange(len(signal))
    if ax is None:
        fig,ax=plt.subplots(figsize=(4,6))
    ax.plot(signal,trials,c='grey',marker='o',alpha=0.8,linewidth=3,label='raw')
    ax.plot(signal_pred,trials,c='k',linewidth=3,label='fitted')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    if dolabel:
        ax.set(xlabel='Peak firing rate (Hz)',ylabel='Trial')
    if dolegend:
        ax.legend()
    if not flipy: # here meaning whether flipy for the heatmap; if heatmap does not need to flip, then this plot need to flip
        ax.invert_yaxis()
    return fig,ax,signal_pred,cpt


def predict_from_cpts_wrapper_allrows(X,pen,cost='l2',min_size=2,model_type=rpt.Pelt):
    pandarallel.initialize()
    # X_norm = scipy.stats.zscore(X,axis=1)
    X_norm = X / X.max(axis=1).values[:,None]
    res=X_norm.parallel_apply(lambda x:predict_from_cpts_wrapper(x.dropna().values,pen,cost=cost,min_size=min_size,model_type=model_type),axis=1)
    X_norm_pwc = pd.DataFrame([r[0] for r in res],index=X.index)
    cpts=pd.DataFrame([pd.Series(r[1]) for r in res],index=X.index)
    return X_norm_pwc, cpts

# sweep version of above, also add in the raw
def turn_X_into_pwc_sweep(X_all_norm,pen_l=[0.1,0.3,0.5]):
    '''
    X_all: dict, {trial_type: df}, df: nfields x ntrials
    '''
    X_norm_pwc_d_allpen = {}
    min_size = 1
    for pen in pen_l:
        X_norm_pwc_d = {}
        for k,X in X_all_norm.items():
            X_norm_pwc,cpts = predict_from_cpts_wrapper_allrows(X,pen,min_size=min_size)
            X_norm_pwc_d[k] = X_norm_pwc
        X_norm_pwc_d_allpen[pen] = X_norm_pwc_d
    X_to_be_analyzed = {'raw':X_all_norm}
    X_to_be_analyzed.update(X_norm_pwc_d_allpen)
    return X_to_be_analyzed

def get_inds_switch_sametrial_sorted(changes_df):
    '''
    sort the fields by when they switch; there may be duplicates
    (onoff x ntrials x nfields_within_trial) x 1
    '''
    inds_d = {1:{},-1:{}}
    for onoff in [1,-1]:
        for c in changes_df.columns:
            inds_d[onoff][c] = pd.Series(changes_df.index[changes_df[c]==onoff].to_numpy())
        inds_d[onoff] = pd.concat(inds_d[onoff])
    inds_d = pd.concat(inds_d)
    return inds_d



# for cross validation!
# NOT USED NOW
def leave_one_out_err(signal,tomask,ncpts,cost='l2'):
    index = np.arange(len(signal))
    model = rpt.Dynp(model=cost,jump=1,min_size=2)
    mask = np.ones(len(index),dtype=bool)
    mask[tomask] = False
    model.fit(signal[mask])
    cpts = model.predict(ncpts)
    cpts = [index[mask][cp] for cp in cpts[:-1]] # go back to original index
    cpts = np.array([0,*cpts,len(signal)])
    signal_hat=predict_from_cpts(signal,cpts)
    err =(signal_hat[tomask] - signal[tomask])**2
    return err

def cv_one_signal(signal,ncpts_l = [0,1,2,3,4],cost='l2'):
    err_l_d = {}
    for ncpts in ncpts_l:
        err_l=[]
        for tomask in range(len(signal)):
            err = leave_one_out_err(signal,tomask,ncpts)
            err_l.append(err)
        err_l_d[ncpts] = np.array(err_l)
    err_l_d = pd.DataFrame(err_l_d)
    best_ncpts = err_l_d.mean().idxmin()
    return best_ncpts, err_l_d


from pandarallel import pandarallel

def cv_all_signals(X,ncpts_l = [0,1,2,3,4],cost='l2'):
    pandarallel.initialize(progress_bar=True)
    # X = X.reset_index(drop=True)
    # ddata = dd.from_pandas(X,npartitions=npartitions)
    row_func = lambda x:cv_one_signal(x.values,ncpts_l = ncpts_l,cost=cost)[0]
    # df_func = lambda df:df.apply(row_func,axis=1)
    # res = ddata.map_partitions(df_func)
    res = X.parallel_apply(row_func,axis=1)
    return res

# ========convert pwc into changes_df, test ========#
def detect_switch_pwc(X_norm_pwc,switch_magnitude=0.,low_thresh=1.,high_thresh=0.):
    '''
    based on switch_magnitude, if 0 then any switch will count
    X_norm_pwc: max normed first and then made piecewise constant; not sure if the order matters much
    low_thresh, high_thresh, further limit the switch, such that the pre has to be below the low and post above the high, for ON (OFF is the opposite) 
    '''
    ntrials = len(X_norm_pwc.columns)
    Xmax=np.tile(X_norm_pwc.max(axis=1).values[:,None],(1,ntrials))
    X_norm_pwc_norm = np.divide(X_norm_pwc, Xmax,out=np.zeros_like(X_norm_pwc),where=Xmax!=0) # weirdly this actually still can't deal with 0 in Xmax???
    X_diff = X_norm_pwc_norm.diff(axis=1)
    switch_on = (X_diff > switch_magnitude) & (X_norm_pwc_norm.shift(axis=1) <= low_thresh) & (X_norm_pwc_norm >= high_thresh)
    switch_off = (X_diff < -switch_magnitude) & (X_norm_pwc_norm.shift(axis=1) >= high_thresh) & (X_norm_pwc_norm <= low_thresh)
    changes_df = np.zeros(X_norm_pwc_norm.shape,dtype=int)
    changes_df[switch_on.values] = 1
    changes_df[switch_off.values] = -1
    changes_df = pd.DataFrame(changes_df,index=X_norm_pwc_norm.index,columns=X_norm_pwc_norm.columns)
    return switch_on, switch_off, changes_df


# ======using hard thresholds ========= #
# two parameter for defining switch: threshold: ratio of max; required duration
def detect_switch_simple(X,thresh_ratio,min_fr_thresh):
    '''
    simplest case, only consider threshold crossing, no other criterion about the minimal duration
    0 is prepended during np.diff, such that if the field is on already, it is 0 is an on pt
    the last trial is appended just for completeness
    X: nfields x ntrials, df
    
    changes_df: nfields x (ntrials+1); if last =-1, that means the field is active until the very end;
    cross_pts_df: on and off points for each field
    '''
    
    thresh = np.maximum(X.max(axis=1) * thresh_ratio,min_fr_thresh).values
    onoff=(X.values > thresh[:,None]).astype(int)
    changes=np.diff(onoff,prepend=0,append=0)
    
    
    cross_on_pts = np.nonzero(changes==1)
    cross_off_pts = np.nonzero(changes==-1)
    func = lambda cross_pts: pd.DataFrame(cross_pts.groupby(level=0).apply(lambda x:x.values),index=X.index)
    cross_on_series=pd.Series(cross_on_pts[1],index=cross_on_pts[0])
    cross_on_series=func(cross_on_series)
    cross_off_series=pd.Series(cross_off_pts[1],index=cross_off_pts[0])
    cross_off_series=func(cross_off_series)
    cross_pts_df = pd.concat([cross_on_series,cross_off_series],axis=1)
    cross_pts_df.columns = ['on','off']
    
    cross_pts_df.index = X.index
    
    onoff_df = pd.DataFrame(onoff,index=X.index,columns=X.columns.astype(int))
    changes_df = pd.DataFrame(changes,index=X.index,columns=np.append(X.columns,-1).astype(int))
    
    return onoff_df,changes_df, cross_pts_df

def detect_switch_by_change(X,thresh_ratio_low=0.1,thresh_ratio_high=0.4,sustain_on=1):
    '''
    detect trials where it is above the high and the previous trial is below the low threshold ratio of the max of that field
    same as switch simple (except for beginning and end), if low==high
    sustain_on: the number of trials including the onset trial that need to be above low
    '''
    below_low=X <= np.max(X.values,axis=1,keepdims=True) * thresh_ratio_low
    above_high=X > np.max(X.values,axis=1,keepdims=True) * thresh_ratio_high
    switch_on=below_low.shift(1,axis=1)&above_high
    for i in range(sustain_on):
        switch_on = switch_on & np.logical_not(below_low.shift(-i,axis=1))
    switch_off=above_high.shift(1,axis=1)&below_low
    changes_df = np.zeros(X.shape,dtype=int)
    changes_df[switch_on.values] = 1
    changes_df[switch_off.values] = -1
    changes_df = pd.DataFrame(changes_df,index=X.index,columns=X.columns)

    return switch_on, switch_off, changes_df

    
def get_switch_ratio_per_trial(changes_df,skip_last_col=True):
    if skip_last_col:
        changes_df = changes_df.loc[:,changes_df.columns!=-1] # get rid of the last column
    
    switch_ratio={'on': (changes_df==1).mean(axis=0),
    'off': (changes_df==-1).mean(axis=0)}
    # 'both': changes_df.mean(axis=0)} # no need for this
    return switch_ratio

def switch_on_off_ratio_correlation(changes_df,sh=None,do_mask=False):
    '''
    changes_df
    sh: how much and which one to shift; positive=>on precedes off=>do -sh on the off, shift off to the left; negative=>off precedes on=> do -sh on the on, shift on to the left
    do_mask: whether to mask the fields that switch off before sh, making sure the lagged correlation would not be due to the same neuron on and off
    '''
    switch_on = changes_df==1
    switch_off = changes_df==-1
    nfields,ntrials = changes_df.shape
    if do_mask:
        mask = ~(switch_on.shift(sh,axis=1) & switch_off).any(axis=1) 
    else:
        mask = np.ones((nfields),dtype=bool)
    
    onratio = switch_on.loc[mask].mean(axis=0)
    offratio = switch_off.loc[mask].mean(axis=0)

    if sh is None:
        r1 = onratio.iloc[1:] #excluding the 0th, since both will necessarily be 0
        r2 = offratio.iloc[1:]
    elif sh>0:
        r1 = onratio.iloc[:-sh]
        r2 = offratio.shift(-sh).dropna()
    elif sh<0:
        r1 = onratio.shift(-sh).dropna()
        r2 = offratio.iloc[:sh]
    else:
        print('sh cannot be 0')
        return

        
    corr=scipy.stats.pearsonr(r1,r2)
    return corr

def popup_trial_into_switch_df(popup_trial,ntrials):
    '''
    popup_trial: nfields x [trial, ispopup]
    ntrials: either a number or pd.Index
    turn into:
    switch_on: nfields x ntrials, binary
    '''
    if isinstance(ntrials,float) or isinstance(ntrials,int):
        trial_index=np.arange(int(ntrials))
    else:
        trial_index = ntrials
    switch_on = pd.DataFrame(False,index=popup_trial.index,columns=trial_index,dtype=bool)
    for key,val in popup_trial['trial'].items():
        switch_on.loc[key,val] = True
    switch_on = switch_on.dropna(axis=1)
    return switch_on

def detect_popup(X,window=5,thresh=0.6,rate_thresh=0.1):
    '''
    for the selected place fields, X is the average fr aross trials; 
    detect the first trial when the rate is above rate_thresh * max, AND the consecutive (window) # of trials
     contain at least (thresh*window) # of trials above the rate_thresh
    X: nfields x ntrials
    '''
    atleast=int(window * thresh)
    cross_thresh = (X > X.max(axis=1).values[:,None]*rate_thresh).astype(int)
    
    ma = cross_thresh.rolling(window,min_periods=1,axis=1).sum().shift(1-window,axis=1) >= atleast
    ma = ma & (cross_thresh.diff(axis=1)==1) # need to be a switch! otherwise pre switch can pass, by the tolerant window
    ispopup = (ma.sum(axis=1) > 0) & (~ma.iloc[:,0])
    popup_trial=pd.DataFrame(ma.to_numpy().nonzero()).T.groupby(0).min()
    popup_trial = pd.DataFrame(X.columns[popup_trial[1]],index=X.index[popup_trial.index],columns=['trial'])
    popup_trial = popup_trial.reindex(X.index)
    popup_trial['ispopup'] =ispopup
    switch_on = popup_trial_into_switch_df(popup_trial,X.columns)
    return popup_trial, switch_on 

def sweep_detect_popup(X,rate_thresh_l,window=5,thresh=0.8,doplot=False,min_size=2,alpha=0.05):
    '''
    min_size: in piecewise constant, min number of trials to be considered a segment
    '''
    count_l = {}
    sig_ct_l = {}
    for rate_thresh in rate_thresh_l:
        popup_trial,switch_on=detect_popup(X ,window=window,thresh=thresh,rate_thresh=rate_thresh)
        switch_on_popup_only=switch_on.loc[popup_trial['ispopup']]
        switch_on_popup_only=switch_on_popup_only.iloc[:,min_size:-(window-1)]
        p = switch_on_popup_only.mean().mean()
        npopups=switch_on_popup_only.sum()
        count_l[rate_thresh] = npopups
        q=1-(0.05 / npopups.shape[0])
        sig_ct = scipy.stats.binom.ppf(q,npopups.shape[0],p=p)
        sig_ct_l[rate_thresh] = sig_ct
    
    count_l = pd.concat(count_l)
    count_l = count_l.unstack() # nthresh x ntrials
    sig_ct_l = pd.Series(sig_ct_l)
    if doplot:
        fig,axs=cpp.plot_count_by_trial_with_thresh_sweep(count_l,sig_ct_l.values,rate_thresh_l)
        return count_l,sig_ct_l,fig,axs
    else:
        return count_l,sig_ct_l

# not good, don't use
def detect_disappear(X,window_on=5,thresh_window_on=0.6,rate_thresh=0.1,window_off=3,thresh_window_off=1.):
    popup_trial, switch_on = detect_popup(X,window=window_on,thresh=thresh_window_on,rate_thresh=rate_thresh)
    atleast = int(window_off * thresh_window_off)
    cross_thresh = (X <= X.max(axis=1).values[:,None]*rate_thresh).astype(int)
    ma = cross_thresh.rolling(window_off,min_periods=1,axis=1).sum().shift(1-window_off,axis=1) >= atleast
    notna_mask = popup_trial['trial'].notna() # popup, including first trial
    for key,val in popup_trial.loc[notna_mask]['trial'].items():
        ma.loc[key,:val] = False # before the pop up, nothing count as disappear
    isdisappear = (ma.sum(axis=1) > 0) & notna_mask 
    disappear_trial = pd.DataFrame(ma.to_numpy().nonzero()).T.groupby(0).min()
    disappear_trial = pd.DataFrame(X.columns[disappear_trial[1]],index=X.index[disappear_trial.index],columns=['trial'])
    disappear_trial = disappear_trial.reindex(X.index)
    disappear_trial['isdisappear'] = isdisappear
    switch_off = popup_trial_into_switch_df(disappear_trial,X.columns)
    return disappear_trial,switch_off

# def detect_first_chunk(fr_map_trial_one_neuron,field_boundary_one_neuron_one_field,window=5,thresh=0.6,rate_thresh=0.1):
#     '''
#     mean rate within field is obtained from:
#         fr_map_trial_one_neuron: nposbins x ntrials
#         field_boundary_one_neuron_one_field: series, 2 (start, end)

#     the first time when the mean rate within field is above a rate_thresh (as a ratio of max), for at least a proportion of window (given by thresh), 
#     within the window given by window
#     '''
#     atleast=int(window * thresh)
#     fr_series=fr_map_trial_one_neuron.loc[field_boundary_one_neuron_one_field['start']:field_boundary_one_neuron_one_field['end']].mean()
#     series = fr_series > fr_series.max() * rate_thresh    
#     ma=series.rolling(window,min_periods=1).sum().shift(1-window) > atleast
#     if ma.sum()>0:
#         popup_trial = series.loc[ma].index[0]
#         ispopup=True
#         if popup_trial==0:
#             ispopup=False
        
#     else:
#         ispopup=False
#         popup_trial=series.index[0]
#     return ispopup, popup_trial

#=====null model (switching on off) =======#

from collections import Counter
eps=1e-8
def get_transition_p_one_field(onoff):
    '''
    #-1,0,1 in change corresponds to index: 0,1,2 in transition_mat
    # 0,1 in onoff corresponds to 0 1 in transition_mat
    '''
    counts = Counter(zip(onoff[:-1],onoff[1:]))
    transition_mat = np.zeros((2,2))
    for ind,c in counts.items():
        transition_mat[ind[0],ind[1]] = c
#     transition_mat = pd.DataFrame(transition_mat,index=[-1,0,1],columns=[-1,0,1])
    transition_mat = pd.DataFrame(transition_mat)
    transition_p=transition_mat / np.sum(transition_mat.values,axis=1,keepdims=True)
    transition_p=transition_p.fillna(eps)
    return transition_p
    
def get_transition_p_population(onoff_df):
    transition_p_all=onoff_df.apply(get_transition_p_one_field,axis=1)
    transition_p_all = transition_p_all.apply(lambda x:x / np.sum(x.values,axis=1,keepdims=True))
    return transition_p_all

def switch_population(X_curr_np,transition_p_all_np):
    '''
    X_curr_np: nfields, 
    transition_p_all_np: nfields x 2 x 2 nd array
    '''
    transition_p_selected = transition_p_all_np[np.arange(X_curr_np.shape[0]),X_curr_np,:]
    x_next = np.random.rand(*X_curr_np.shape)
    x_next = (x_next < transition_p_selected[:,1]).astype(int)
    return x_next

def gen_switching_data(transition_p_all,x_init,ntrials=30):
    '''
    transition_p_all: series of df; nfields x (2 x 2)
    x_init: series: nfields
    '''
    transition_p_all_np = np.stack(transition_p_all.values)
    nneurons = transition_p_all.shape[0]
    x_sim = np.zeros((nneurons,ntrials),dtype=int)
    x_sim[:,0] = x_init.values
    for n in range(1,ntrials):
        x_sim[:,n] = switch_population(x_sim[:,n-1],transition_p_all_np)
    x_sim = pd.DataFrame(x_sim, index=x_init.index)
    return x_sim

# null model: circular shuffle
def gen_circular_shuffle(changes_df,nrepeats=200,min_cpd_win=2,new_start_inds=None):
    '''
    changes_df: anything nfields x ntrials, can be the original data or the one after the change point detection
    '''
    
    nfields,ntrials_orig=changes_df.shape
    X = changes_df.values[:,min_cpd_win:(ntrials_orig-(min_cpd_win-1))]
    ntrials = X.shape[1]
    X_all_roll = np.array([np.roll(X,i,axis=1) for i in range(X.shape[1])]) # generate all possible circular shifts for the population together
    # np.random.seed(10)
    if new_start_inds is None:
        new_start_inds=np.random.randint(ntrials-1,size=(nrepeats,nfields)) # sample independently for each field the starting point in the shift
    # sim_l=np.array([X_all_roll[new_start_inds[i],np.arange(nfields)] for i in range(new_start_inds.shape[0])])
    X_sim_l = []
    for i in range(new_start_inds.shape[0]):
        x_sim_one = X_all_roll[new_start_inds[i],np.arange(nfields)]
        x_sim_one_padded=np.concatenate([np.zeros((nfields,min_cpd_win)),x_sim_one,np.zeros((nfields,min_cpd_win-1))],axis=1)
        x_sim_one_padded_df = pd.DataFrame(x_sim_one_padded,index=changes_df.index,columns=changes_df.columns)
        X_sim_l.append(x_sim_one_padded_df)
    # X_sim_l = [pd.DataFrame(X_all_roll[new_start_inds[i],np.arange(nfields)],index=changes_df.index,columns=changes_df.columns) for i in range(new_start_inds.shape[0])] # using indexing to select the corresponding shift for each field in each shuffle
    return X_sim_l

def get_switch_ratio_in_shuffles(X_sim_l,is_changes_df=True):
    '''
    is_changes_df: if False, then need to run detect_switch_simple first
    '''
    sr_l =[]
    for X_sim in X_sim_l:
        if is_changes_df:
            changes_df_sim = X_sim
            sw = get_switch_ratio_per_trial(changes_df_sim, skip_last_col=False)    
            sr_l.append(sw)
    return sr_l

def test_sig_switch_ratio(sr_data,sr_l):
    cdf_d = {}
    for ii,k in enumerate(sr_data.keys()):
        sr_on=pd.DataFrame([sr[k] for sr in sr_l]).T
        cdf=(sr_data[k].values[:,None] > sr_on.values).mean(axis=1)
        # import pdb
        # pdb.set_trace()
        cdf[sr_on.values.sum(axis=1)<0.01] = 0.5 # if shuffle is 0, then set cdf to 0.5
        cdf_d[k] = cdf
    cdf_d= pd.DataFrame(cdf_d,index=sr_data['on'].index)
    return cdf_d

def test_switch_ratio_wrapper(X,detect_func=detect_switch_by_change,alpha=0.05,do_bonf=True,doplots=False,nrepeats=200,**kwargs):
    '''
    get switch ratio using detect_switch_by_change for X: nfields x ntrials
    do circular shuffle
    get the cdf at the data switch ratio for the null distribution
    cdf_d: ntrials x 3, cols=['on','off','both']

    detect_func: detect_switch_by_change or detect_switch_pwc    
    '''
    switch_on,switch_off,changes_df = detect_func(X,**kwargs)

    sr_data = get_switch_ratio_per_trial(changes_df,skip_last_col=False)

    X_sim_l = gen_circular_shuffle(changes_df,nrepeats=nrepeats)

    sr_l=get_switch_ratio_in_shuffles(X_sim_l,is_changes_df=True)
    
    cdf_d = test_sig_switch_ratio(sr_data,sr_l)
    if doplots:
        if do_bonf:
            alpha = alpha / X.shape[1]
        fig,axs=cpp.plot_switch_ratio_with_shuffle(sr_data,sr_l,alpha=alpha)
        return cdf_d,sr_data,changes_df,fig,axs
    else:
        return cdf_d,sr_data,changes_df

def sweep_test_switch_ratio(X,tosweep_key,tosweep_val,detect_func=detect_switch_by_change,alpha=0.05,do_bonf=True,doplots=False,nrepeats=200,**kwargs):
    cdf_d_d = {}
    sr_data_d = {}
    changes_df_d = {}
    for val in tosweep_val:
        kwargs[tosweep_key] = val
        cdf_d,sr_data,changes_df=test_switch_ratio_wrapper(X,detect_func=detect_func,alpha=alpha,do_bonf=do_bonf,doplots=False,nrepeats=nrepeats,**kwargs)
        cdf_d_d[val] = cdf_d
        sr_data_d[val] = pd.concat(sr_data,axis=1).T
        changes_df_d[val] = changes_df
    cdf_d_d = pd.concat(cdf_d_d).unstack()
    # import pdb
    # pdb.set_trace()
    changes_df_d = pd.concat(changes_df_d)
    sr_data_d = pd.concat(sr_data_d,axis=0)
    if do_bonf:
        alpha = alpha / X.shape[1] # in some cases, like when there's sustain, this might be incorrect! 
    sig_d_d = (cdf_d_d > (1-alpha/2)) | (cdf_d_d < alpha / 2)
    if doplots:
        # fig,axs = cpp.plot_sweep_test_switch_ratio(cdf_d_d,tosweep_key,alpha=alpha,do_bonf=do_bonf)
        fig,axs = cpp.plot_sweep_test_switch_ratio(cdf_d_d,sr_data_d,sig_d_d,tosweep_key)
        return cdf_d_d, sig_d_d,sr_data_d, changes_df_d ,fig,axs
    else:
        return cdf_d_d, sig_d_d ,sr_data_d, changes_df_d


from scipy.spatial.distance import pdist,squareform,dice
#======compare to null model, get significantly covarying pairs=======#
def get_sig_pairs(onoff_df,X_sim_l):
    '''
    use the change points or on-off states to define pairwise distance
    use the shuffle to get a null distribution of such distances, specific to each pair
    onoff_df: nfields x ntrials, binary df, 
    X_sim_l: nrepeats of similated onoff_df

    
    '''
    res = {}
    dist = pdist(onoff_df,metric=dice)
    dist_sq=squareform(dist)
    res['dist']=dist
    res['dist_sq']=distsq
    all_dists_sim,all_dists_sim_sq=batch_dice_distance(np.array(X_sim_l))
    # all_dists_sim= np.array([pdist(X_sim_l[rr],metric=dice) for rr in range(len(X_sim_l))])
    res['all_dists_sim']=all_dists_sim
    res['all_dists_sim_sq'] = all_dists_sim_sq#np.array([squareform(ad) for ad in all_dists_sim])


    res['sig_thresh'] = np.quantile(all_dists_sim,0.05,axis=0)
    res['sig'] = dist < res['sig_thresh']
    res['sig_sq'] = squareform(res['sig'])
    res['sig_ilocs'] = np.nonzero(res['sig_sq'])
    res['p_val']=(dist > all_dists_sim).mean(axis=0)
    res['p_val_sq'] = squareform(res['p_val'])

    return res

def batch_dice_distance(xb):
    '''
    xb: n_batch x nfields x nfields, np array

    all nan, i.e. no switch on neurons, have been coded = 10
    '''
    ctt = np.einsum('bft,bnt->bfn',xb,xb)

    ctf = np.einsum('bft,bnt->bfn',xb,np.logical_not(xb))

    cft = np.swapaxes(ctf,1,2)

    D = (ctf + cft) / (ctf + cft + 2*ctt)
#     D=np.nan_to_num(D,nan=0) # a bit sloppy, since nan means one field has no turning on;
    D_l = []
    D=np.nan_to_num(D,nan=10)
    for d in D:
        d[np.diag_indices(d.shape[0])]=0
        D_l.append(squareform(d))
    D_l = np.array(D_l)
#     D_l = np.array([squareform(d) for d in D])
    return D_l,D
    

def sweep_test_switch_ratio_multisweep_onetrialtype(X,min_size,tosweep_key_l,tosweep_val_l,kwargs_l,detect_func=detect_switch_pwc,alpha=0.05,do_bonf=True,doplots=True,nrepeats=400):
    cdf_alltosweep_key = {}
    sig_alltosweep_key = {}
    sr_alltosweep_key = {}
    changes_df_alltosweep_key = {}
    fig_alltosweep_key = {}

    for tosweep_key,tosweep_val,kwargs in zip(tosweep_key_l,tosweep_val_l,kwargs_l):
        # if min_size > 1:
        #     res =sweep_test_switch_ratio(X.iloc[:,min_size:(-(min_size-1))],tosweep_key,tosweep_val,detect_func=detect_func,alpha=alpha,do_bonf=do_bonf,doplots=doplots,nrepeats=nrepeats,**kwargs)
        # else:
        #     res = sweep_test_switch_ratio(X.iloc[:,min_size:],tosweep_key,tosweep_val,detect_func=detect_func,alpha=alpha,do_bonf=do_bonf,doplots=doplots,nrepeats=nrepeats,**kwargs)
        # get rid of min_size; previous would at least cut off the first trial, then on/off after one trial can't be detected
        res =sweep_test_switch_ratio(X,tosweep_key,tosweep_val,detect_func=detect_func,alpha=alpha,do_bonf=do_bonf,doplots=doplots,nrepeats=nrepeats,**kwargs)
        if doplots:
            cdf_d_d,sig_d_d,sr_data_d,changes_df_d,fig,axs = res
            fig_alltosweep_key[tosweep_key] = fig
        else:
            cdf_d_d,sig_d_d,sr_data_d,changes_df_d = res
        
        cdf_alltosweep_key[tosweep_key] = cdf_d_d
        sig_alltosweep_key[tosweep_key] = sig_d_d

        sr_alltosweep_key[tosweep_key] = sr_data_d
        changes_df_alltosweep_key[tosweep_key] = changes_df_d
    
    cdf_alltosweep_key = pd.concat(cdf_alltosweep_key)
    sig_alltosweep_key = pd.concat(sig_alltosweep_key)
    sr_alltosweep_key = pd.concat(sr_alltosweep_key)
    changes_df_alltosweep_key = pd.concat(changes_df_alltosweep_key)
    if doplots:
        return cdf_alltosweep_key,sig_alltosweep_key,sr_alltosweep_key,changes_df_alltosweep_key,fig_alltosweep_key    
    else:
        return cdf_alltosweep_key,sig_alltosweep_key,sr_alltosweep_key,changes_df_alltosweep_key

def sweep_test_switch_ratio_multisweep_onetrialtype_multipreprocess(X_allpen,key,min_size,tosweep_key_l,tosweep_val_l,kwargs_l,detect_func=detect_switch_pwc,alpha=0.05,do_bonf=True,doplots=True,nrepeats=400):  
    cdf_allpen = {}
    sig_allpen = {}
    sr_allpen = {}
    changes_df_allpen = {}
    fig_allpen = {}

    for pen, Xal in X_allpen.items():
        X = Xal[key]
        if pen=='raw':
            min_size=1 # when the X is not a pwc version, then change can happen from the trial 1 (0indexed)
        res = sweep_test_switch_ratio_multisweep_onetrialtype(X,min_size,tosweep_key_l,tosweep_val_l,kwargs_l,detect_func=detect_func,alpha=alpha,do_bonf=do_bonf,doplots=doplots,nrepeats=nrepeats)

        if doplots:
            cdf_d_d,sig_d_d,sr_data_d,changes_df_d,fig = res
            fig_allpen[pen] = fig
        else:
            cdf_d_d,sig_d_d,sr_data_d,changes_df_d = res
        
        cdf_allpen[pen] = cdf_d_d
        sig_allpen[pen] = sig_d_d

        sr_allpen[pen] = sr_data_d
        changes_df_allpen[pen] = changes_df_d
    
    cdf_allpen = pd.concat(cdf_allpen)
    sig_allpen = pd.concat(sig_allpen)
    sr_allpen = pd.concat(sr_allpen)
    changes_df_allpen = pd.concat(changes_df_allpen)
    if doplots:
        return cdf_allpen,sig_allpen,sr_allpen,changes_df_allpen,fig_allpen    
    else:
        return cdf_allpen,sig_allpen,sr_allpen,changes_df_allpen

def sweep_test_switch_ratio_multisweep_alltrialtype_multipreprocess(X_allpen,min_size,tosweep_key_l,tosweep_val_l,kwargs_l,detect_func=detect_switch_pwc,alpha=0.05,do_bonf=True,doplots=True,nrepeats=400):
    all_trialtypes =X_allpen[list(X_allpen.keys())[0]].keys()

    cdf_alltrialtype = {}
    sig_alltrialtype = {}
    sr_alltrialtype = {}
    changes_df_alltrialtype = {}
    fig_alltrialtype = {}

    for trialtype in all_trialtypes:
        res = sweep_test_switch_ratio_multisweep_onetrialtype_multipreprocess(X_allpen,trialtype,min_size,tosweep_key_l,tosweep_val_l,kwargs_l,detect_func=detect_switch_pwc,alpha=alpha,do_bonf=do_bonf,doplots=doplots,nrepeats=nrepeats)

        if doplots:
            cdf_d_d,sig_d_d,sr_data_d,changes_df_d,fig = res
            fig_alltrialtype[trialtype] = fig
        else:
            cdf_d_d,sig_d_d,sr_data_d,changes_df_d = res
        
        cdf_alltrialtype[trialtype] = cdf_d_d
        sig_alltrialtype[trialtype] = sig_d_d

        sr_alltrialtype[trialtype] = sr_data_d
        changes_df_alltrialtype[trialtype] = changes_df_d
    
    cdf_alltrialtype = pd.concat(cdf_alltrialtype)
    sig_alltrialtype = pd.concat(sig_alltrialtype)
    sr_alltrialtype = pd.concat(sr_alltrialtype)
    changes_df_alltrialtype = pd.concat(changes_df_alltrialtype)
    if doplots:
        return cdf_alltrialtype,sig_alltrialtype,sr_alltrialtype,changes_df_alltrialtype,fig_alltrialtype    
    else:
        return cdf_alltrialtype,sig_alltrialtype,sr_alltrialtype,changes_df_alltrialtype

# def sweep_test_switch_ratio_alltrialtype(Xal,min_size,tosweep_key,tosweep_val,detect_func=detect_switch_pwc,alpha=0.05,do_bonf=True,doplots=True,nrepeats=400,**kwargs):
#     '''
#     Xal: dicts of X, each nfields x ntrials df
#     '''
#     cdf_alltrialtype = {}
#     sig_alltrialtype = {}
#     sr_alltrialtype = {}
#     changes_df_alltrialtype = {}
#     fig_alltrialtype = {}
#     for trialtype,X_norm_pwc in Xal.items():
#         if min_size > 1:
#             res =sweep_test_switch_ratio(X_norm_pwc.iloc[:,min_size:(-(min_size-1))],tosweep_key,tosweep_val,detect_func=detect_func,alpha=alpha,do_bonf=do_bonf,doplots=doplots,nrepeats=nrepeats,**kwargs)
#         else:
#             res = sweep_test_switch_ratio(X_norm_pwc.iloc[:,min_size:],tosweep_key,tosweep_val,detect_func=detect_func,alpha=alpha,do_bonf=do_bonf,doplots=doplots,nrepeats=nrepeats,**kwargs)
        
#         if doplots:
#             cdf_d_d,sig_d_d,sr_data_d,changes_df_d,fig,axs = res
#             fig.suptitle(trialtype,fontsize=12)
#             plt.tight_layout()
#             fig_alltrialtype[trialtype] = fig
#         else:
#             cdf_d_d,sig_d_d,sr_data_d,changes_df_d = res

        
#         cdf_alltrialtype[trialtype] = cdf_d_d
#         sig_alltrialtype[trialtype] = sig_d_d
#         sr_alltrialtype[trialtype] = sr_data_d
#         changes_df_alltrialtype[trialtype] = changes_df_d
    
#     cdf_alltrialtype = pd.concat(cdf_alltrialtype)
#     sig_alltrialtype = pd.concat(sig_alltrialtype)
#     sr_alltrialtype = pd.concat(sr_alltrialtype)
#     changes_df_alltrialtype = pd.concat(changes_df_alltrialtype)

#     if doplots:

#         return cdf_alltrialtype,sig_alltrialtype,sr_alltrialtype,changes_df_alltrialtype,fig_alltrialtype    
#     else:
#         return cdf_alltrialtype,sig_alltrialtype,sr_alltrialtype,changes_df_alltrialtype


def get_switching_com_distribution_per_trial(all_fields,inds_d, field_loc_key='com',coarse_bins=np.arange(0,101,10)):
    '''
    all_fields: n_tot_fields x [start,end,com,peak,fr_peak,fr_mean]
    inds_d: series (([on/1, off/-1] x trial x nfields_switching_in_trial) 
    '''
    
    coswitching_coms= pd.Series(all_fields[field_loc_key].loc[inds_d.values].values,index=inds_d.index)
    field_loc_histogram = np.histogram(all_fields[field_loc_key],coarse_bins)[0]
    coswitching_coms_binned=coswitching_coms.groupby(level=[0,1]).apply(lambda x:np.histogram(x,coarse_bins)[0])
    coswitching_coms_binned_ = np.stack(coswitching_coms_binned.values,axis=0)
    coswitching_coms_binned = pd.DataFrame(coswitching_coms_binned_,index=coswitching_coms_binned.index)
    
    uid_l = np.array(inds_d.to_list())[:,0]
    coswitching_coms = coswitching_coms.to_frame().rename(columns={0:field_loc_key})
    coswitching_coms['uid'] = uid_l

    coswitching_coms_binned_ratio = coswitching_coms_binned / field_loc_histogram
    coswitching_per_trial_ratio = coswitching_coms_binned.sum(axis=1) / field_loc_histogram.sum()

    return coswitching_coms,coswitching_coms_binned,field_loc_histogram, coswitching_coms_binned_ratio, coswitching_per_trial_ratio


# PAIRWISE TEST
def get_shared_onoff(changes_one,return_outer=False):
    '''
    changes_one: df, nfields x ntrials
    '''
    on = (changes_one == 1).astype(int)
    off = (changes_one == -1).astype(int)

    on_outer=np.einsum('ft,dt->fd',on, on)
    share_on_count = len(np.nonzero(np.triu(on_outer,1))[0])
    off_outer=np.einsum('ft,dt->fd',off, off)
    share_off_count = len(np.nonzero(np.triu(off_outer,1))[0])
    onoff_outer_d = {1:on_outer>0,-1:off_outer>0} 
    np.fill_diagonal(onoff_outer_d[1],0) # get rid of diagonals
    np.fill_diagonal(onoff_outer_d[-1],0)

    onoff_outer_d = {k:pd.DataFrame(val,changes_one.index,changes_one.index) for k,val in onoff_outer_d.items()}
    
    share_onoff=(on_outer>0) & (off_outer > 0)
    np.fill_diagonal(share_onoff,0)
    share_onoff = np.triu(share_onoff)

    share_onoff_inds = np.nonzero(share_onoff)
    share_onoff_count = len(share_onoff_inds[0])
    
    share_onoff_ratio_on = share_onoff_count / share_on_count

    if return_outer:
        return share_onoff_inds, share_onoff_count, share_on_count, share_off_count, share_onoff_ratio_on, onoff_outer_d
    else:
        return share_onoff_inds, share_onoff_count, share_on_count, share_off_count, share_onoff_ratio_on

def shuffle_test_pair_share_onoff(changes_one,nrepeats=100,alpha=0.025):
    '''
    alpha: precomputed, no further transformation to one-sided or two-sided
    '''
    changes_shuffle_l = gen_circular_shuffle(changes_one,nrepeats)
    
    share_onoff_count_l=[]
    share_on_count_l = []
    share_off_count_l = []
    share_onoff_ratio_on_l = []
    
    
    for changes_shuffle in changes_shuffle_l:
        _, share_onoff_count, share_on_count, share_off_count,share_onoff_ratio_on = get_shared_onoff(changes_shuffle)
        share_onoff_count_l.append(share_onoff_count)
        share_on_count_l.append(share_on_count)
        share_off_count_l.append(share_off_count)
        share_onoff_ratio_on_l.append(share_onoff_ratio_on)
    
    share_shuffle = {'onoff':np.array(share_onoff_count_l),'on':np.array(share_on_count_l),
                     'off':np.array(share_off_count_l),'onoff_ratio_on':np.array(share_onoff_ratio_on_l)}
    
    share_data = {}
    share_onoff_inds, share_data['onoff'], share_data['on'], share_data['off'],share_data['onoff_ratio_on'] = get_shared_onoff(changes_one)
    
    cdf_d = {}
    issig_d = {}
    for k,shuffle in share_shuffle.items():
        data = share_data[k]
        cdf = np.mean(data >= shuffle)
        issig = cdf > (1-alpha)
        cdf_d[k] = cdf
        issig_d[k] = issig
    res = {'count':share_data,'cdf':cdf_d,'issig':issig_d}
    res = pd.DataFrame(res).unstack()
    return res,share_shuffle
        
        
    
