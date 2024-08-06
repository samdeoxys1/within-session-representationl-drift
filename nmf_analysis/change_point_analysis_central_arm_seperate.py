import ruptures as rpt
import numpy as np
import pandas as pd
import scipy
import os,sys,copy,itertools,pdb,importlib
import change_point_plot as cpp
importlib.reload(cpp)
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append('/mnt/home/szheng/projects/util_code/')
sys.path.append('/mnt/home/szheng/projects/nmf_analysis')
sys.path.append('/mnt/home/szheng/projects/nmf_analysis/scripts')
import process_central_arm as pca
import change_point_analysis as cpa
import data_prep_pyn as dpp
import place_cell_analysis as pa
import place_field_analysis as pf
import switch_analysis_one_session as saos
import misc


def switch_detection_combined(pf_res,all_fields,pf_fr,index_within_to_trial_index_df,spk_beh_df,changes_df_both_trialtype=None,task_ind=0,**kwargs):
    '''
    all_fields: from eg. pf_res['avg']['all_fields'].loc[task_ind], (trial_ind x neuron x field) x [start, end, com, peak, fr_peak, fr_mean]
    pf_fr: from eg. 
            fr_key = 'fr_peak'
            pf_fr = pd.concat(pf_res['avg']['params'],axis=0).loc[(slice(None),slice(None),fr_key),:]
            index=pf_fr.index.droplevel(2)
            pf_fr.index=index
            pf_fr = pf_fr.loc[task_ind]

    changes_df_both_trialtype: old changes_df computed seperately for both trialtypes for all neurons
        eg: switch_res['avg']['changes_df'].loc[task_ind,slice(None),0.3,'switch_magnitude',0.4]

    ====
    
    '''
    kwargs_=dict(switch_magnitude=0.4,loc_key='peak',pen=0.3,min_size=1)
    kwargs_.update(kwargs)

    splitter_fields,nonsplitter_fields, pf_fr_trialtype_combined_splitter, pf_fr_trialtype_combined_nonsplitter = pca.divide_central_fields_splitter_gather_params(all_fields,pf_fr,index_within_to_trial_index_df,task_ind=task_ind,**kwargs_)

    # max norm; consider making this an option if needed
    X_central = pf_fr_trialtype_combined_nonsplitter / pf_fr_trialtype_combined_nonsplitter.max(axis=1).values[:,None]
    X_central_pwc,cpts = cpa.predict_from_cpts_wrapper_allrows(X_central,pen=kwargs_['pen'],min_size=kwargs_['min_size'])
    switch_on_central, switch_off_central, changes_df_central = cpa.detect_switch_pwc(X_central_pwc, switch_magnitude=kwargs_['switch_magnitude'],low_thresh=1.,high_thresh=0.)

    changes_df_combined,changes_df_combined_d = pca.combine_changes_df_side_central(changes_df_central,changes_df_both_trialtype,nonsplitter_fields,index_within_to_trial_index_df.loc[task_ind]) # task_ind =None because index_within_to_trial_index_df has already been queried at the task_ind before being fed to the function

    trial_pos_info = pca.get_trial_pos_info(spk_beh_df,speed_thresh=0,task_ind=task_ind) #not sure if this is justified. want the time calculation to not have holes due to speed thresholds
    pos_to_time_func_per_trial = pca.get_pos_to_time_func_per_trial(trial_pos_info)

    pf_loc_combined,pf_all_field_combined = pca.combine_field_loc(pf_res,nonsplitter_fields,index_within_to_trial_index_df,loc_key=kwargs_['loc_key'],task_ind = task_ind)

    all_sw_times_v_combined = pca.get_all_switch_times_combined(pos_to_time_func_per_trial,pf_loc_combined,changes_df_combined)
    
    all_fields_times_v =pca.get_all_place_field_times_combined(pos_to_time_func_per_trial,pf_loc_combined)

    res = dict(all_sw_times_v_combined=all_sw_times_v_combined,\
                all_fields_times_v=all_fields_times_v,\
                changes_df_combined=changes_df_combined, \
                changes_df_combined_d=changes_df_combined_d,\
                pf_loc_combined=pf_loc_combined, \
                trial_pos_info=trial_pos_info, \
                pos_to_time_func_per_trial=pos_to_time_func_per_trial,\
                pf_all_field_combined=pf_all_field_combined,
                )
    return res

def get_diff_flatten(df,key='time',exclude_self=False):
    diff = np.abs(df[key].values[:,None] - df[key].values[None,:])
    diff_flat = diff[np.triu_indices(diff.shape[0],1)]
    if exclude_self:
        from_same_neuron_mask = np.equal.outer(df['uid'].values,df['uid'].values) # whether a pair of switches come from the same uid
        from_same_neuron_mask_flat = from_same_neuron_mask[np.triu_indices(from_same_neuron_mask.shape[0],1)]
        diff_flat = diff_flat[~from_same_neuron_mask_flat]
    return diff_flat



# count the number of co switches that happen within a window!
def count_diff_within(df,diff_flat=None,key='time',time_diff_thresh_up=30,time_diff_thresh_low=0,exclude_self=False,normalize_by_window=False):
    '''
    df: all_sw_times_v_combined_on: n_switch_on x [trialtype, uid, field_index, time, v, field_pos, trial_index, switch]
    time_diff_thresh_up, time_diff_thresh_low: range of the differences in switch timing. Differences within the range are considered co-switches
        unit in second
    normalize_by_window: if True, divide the count by the window size
    '''
    if diff_flat is None:
        diff_flat = get_diff_flatten(df,key=key,exclude_self=exclude_self)
    count = ((diff_flat < time_diff_thresh_up) & (diff_flat >= time_diff_thresh_low)).sum()
    if normalize_by_window:
        count = count / (time_diff_thresh_up - time_diff_thresh_low)
    return count,diff_flat

def count_diff_within_multi_windows(df,edges,key='time',exclude_self=False,normalize_by_window=False):
    '''
    more efficient especially when df has many rows
    '''
    diff_flat = None
    count_l = []
    # for i in range(len(edges)-1):
    #     low,up = edges[i],edges[i+1]
    #     count,diff_flat = count_diff_within(df,diff_flat=diff_flat,time_diff_thresh_up=up,time_diff_thresh_low=low,exclude_self=exclude_self,normalize_by_window=normalize_by_window)
    #     count_l.append(count)
    # count_l = np.array(count_l)
    diff_flat = get_diff_flatten(df,key=key,exclude_self=exclude_self)
    count_l,_=np.histogram(diff_flat,edges)
    
    return count_l,diff_flat

def count_diff_within_multi_windows_shuffle(all_sw_times_v_combined_shuffle_l,edges,key='time',onoff=1,exclude_self=False,normalize_by_window=False):
    '''
    all_sw_times_v_combined_shuffle_l:
    '''
    c_l = []
    for ast in all_sw_times_v_combined_shuffle_l:
        ast_on = ast.loc[ast['switch']==onoff]
        c,_ = count_diff_within_multi_windows(ast_on,edges,key=key,exclude_self=exclude_self,normalize_by_window=normalize_by_window)
        c_l.append(c)
    return np.array(c_l) # nshuffle x nbins

def count_diff_in_time(df,time_range,window_size=1.,key='time'):
    edges = np.arange(time_range[0],time_range[1]+window_size,window_size)
    c_l, _=np.histogram(df[key],edges)
    return c_l, edges

def count_switch(all_sw_times_v_combined,lims,all_fields_times_v=None,window_size=5,v_thresh=-100,key='field_pos',onoff=1):
    if all_fields_times_v is not None:
        df_l = [all_sw_times_v_combined.query('switch==@onoff'),all_fields_times_v]
    count_l = []
    for df in df_l:
        if 'v' in df.keys():
            df_speedfiltered=df.query('v>=@v_thresh')
        else:
            df_speedfiltered = df
        count_in_time,edges = count_diff_in_time(df_speedfiltered,lims,window_size=window_size,key=key)
        count_l.append(count_in_time)
    count_l = pd.DataFrame(count_l,index=[onoff,'baseline']).T
    count_l['ratio'] = count_l[onoff] / count_l['baseline']
    count_l['edges'] = edges[0:-1]
    return count_l, edges


def shuffle_changes_df_combined_get_sw_times_v_combined_shuffle(changes_df_combined_d,index_within_to_trial_index_df,pos_to_time_func_per_trial,pf_loc_combined,nrepeats=1000,task_ind=0):
    changes_df_combined_shuffle_l = pca.gen_circular_shuffle_changes_df_combined(changes_df_combined_d,index_within_to_trial_index_df.loc[task_ind],nrepeats=nrepeats)
    all_sw_times_v_combined_shuffle_l = []
    for cdcs in changes_df_combined_shuffle_l:
        all_sw_times_v_combined_shuffle = pca.get_all_switch_times_combined(pos_to_time_func_per_trial,pf_loc_combined,cdcs)
        all_sw_times_v_combined_shuffle_l.append(all_sw_times_v_combined_shuffle)
    return all_sw_times_v_combined_shuffle_l


def sweep_test_coswitch(all_sw_times_v_combined,all_sw_times_v_combined_shuffle_l,edges,all_fields_times_v=None,key='time',onoff=1,p_thresh=0.05,ci=0.95,do_ratio=False,exclude_self=False,normalize_by_window=False):
    '''

    all_fields_times_v: (nfields x ntrials) x [times, v, ...]; for computing a baseline count of field pairs within a window
        all_fields_times_v =pca.get_all_place_field_times_combined(pos_to_time_func_per_trial,pf_loc_combined)
    '''
    c_shuffle_d = {}
    c_data_d = {}
    rate_shuffle_d = {}
    rate_data_d = {}
    pval_d = {}
    sig_d = {}
    rate_sig_thresh_in_shuffle_d = {}
    rate_ci_low_in_shuffle_d = {}
    rate_ci_up_in_shuffle_d = {}

    c_ci_low_in_shuffle_d={}
    c_ci_up_in_shuffle_d={}
    
    # if exclude_self: # don't consider multiple fields of the same neuron as possible for co-switch
    #     n_pairs_tot = 0

    all_sw_times_v_combined_onoff = all_sw_times_v_combined.loc[all_sw_times_v_combined['switch']==onoff]
    ci_low,ci_up = (1 - ci)/2,(1 - ci)/2 + ci

    # baseline counts:
    if all_fields_times_v is not None:
        baseline_count_l,_ = count_diff_within_multi_windows(all_fields_times_v,edges,key=key,exclude_self=True) # should just be true to avoid double counting the same field within two trials?
        

    # count co switch:
    # nshuffle x nbins
    c_shuffle_l = count_diff_within_multi_windows_shuffle(all_sw_times_v_combined_shuffle_l,edges,key=key,onoff=onoff,exclude_self=exclude_self,normalize_by_window=normalize_by_window)
    # nbins
    c_data_l,_ = count_diff_within_multi_windows(all_sw_times_v_combined_onoff,edges,key=key,exclude_self=exclude_self,normalize_by_window=normalize_by_window)

    #pval: nbins
    pval_l = 1 - (c_data_l[None,:] > c_shuffle_l).sum(axis=0) / c_shuffle_l.shape[0]
    sig_l = pval_l <= p_thresh
    c_ci_low = np.quantile(c_shuffle_l, ci_low,axis=0)
    c_ci_up = np.quantile(c_shuffle_l, ci_up,axis=0)
    window_size = np.diff(edges)
    # pdb.set_trace()
    test_res_df = pd.DataFrame([edges[:-1],edges[1:],c_data_l,pval_l,sig_l,c_ci_low,c_ci_up,window_size],index=['window_low','window_high','count','pval','sig','count_ci_low','count_ci_up','window_size']).T
    test_res_df['window_median'] = test_res_df[['window_low','window_high']].median(axis=1)
    
    if all_fields_times_v is not None:
        test_res_df['baseline'] = baseline_count_l
    # test_res_df=test_res_df.set_index('window_median')

    # return test_res_df,c_shuffle_d#, rate_shuffle_d
    return test_res_df,c_shuffle_l#, rate_shuffle_d

def sweep_test_coswitch_wrapper(data_dir_full,
                                pf_res_save_fn='place_field_avg_and_trial_vthresh.p',
                                pf_shuffle_fn = 'fr_map_null_trialtype_vthresh.p',
                                speed_key='v',fr_key='fr_peak',
                                bin_size=2.2,
                                switch_res_query=(slice(None),0.3,'switch_magnitude',0.4),
                                nrepeats_sw = 100,
                                edges = None,
                                save_fn = 'switch_res_window.p',
                                load_only=False,
                                dosave=False,force_reload=False,
                                task_ind = 0,
                                prep_force_reload=False,
                                ):
    # create subdir
    save_dir = misc.get_or_create_subdir(data_dir_full,'py_data','switch_analysis')
    save_fn, test_res = misc.get_res(save_dir,save_fn,force_reload)
    if (test_res is not None) or load_only: # load only would skip the computation that follows
        return test_res
    

    # load stuff
    prep_res = dpp.load_spk_beh_df(data_dir_full,force_reload=prep_force_reload,extra_load={})
    spk_beh_df=prep_res['spk_beh_df']
    cell_cols_d = prep_res['cell_cols_d']
    _,spk_beh_df = dpp.group_into_trialtype(spk_beh_df)
    spk_beh_df,_=dpp.add_lin_binned(spk_beh_df,bin_size=bin_size,nbins=None) # when there's bin_size , nbins doesn't matter
    index_within_to_trial_index_df = dpp.index_within_to_trial_index(spk_beh_df)
    
    cell_cols = cell_cols_d['pyr']
    fr_map_dict=pa.get_fr_map_trial(spk_beh_df,cell_cols,trialtype_key='trial_type',speed_key='v',speed_thresh=1.,order=['smooth','divide'])
    fr_map_trial_d = {k:val[0] for k,val in fr_map_dict.items()}
    fr_map_trial_df_d=pd.concat({k:pf.fr_map_trial_to_df(fr_map_trial_d[k],cell_cols) for k in fr_map_dict.keys()},axis=0)

    pf_res =pf.field_detection_both_avg_trial_wrapper(data_dir_full, dosave=True,force_reload=False, bin_size=bin_size,
                                        save_fn = pf_res_save_fn, 
                                        shuffle_fn = pf_shuffle_fn,
                                        smth_in_bin=2.5, speed_thresh=1.,speed_key=speed_key,load_only=True,
                                        )
    all_fields_all_trialtype=pd.concat(pf_res['avg']['all_fields'],axis=0)

    switch_res = saos.switch_analysis_one_session(data_dir_full,place_field_res=None,force_reload=False,nrepeats=1000,
                                              save_fn=saos.SAVE_FN(saos.FR_KEY),load_only=True)

    plt.close('all')

    task_l = spk_beh_df['task_index'].unique()

    if edges is None:  
        edges = [0,0.1,1,10,30,60,120,200,300,400,500,600]
        

    for tsk_ind in task_l:
        if tsk_ind ==task_ind: # temporary fix!!!!! extend to multi task in the future
            all_fields = all_fields_all_trialtype.loc[task_ind]

            
            pf_fr = pd.concat(pf_res['avg']['params'],axis=0).loc[(slice(None),slice(None),fr_key),:]
            index=pf_fr.index.droplevel(2)
            pf_fr.index=index
            
            pf_fr = pf_fr.loc[task_ind]
            changes_df_both_trialtype = switch_res['avg']['changes_df'].loc[(task_ind,*switch_res_query)]
            # do switch detection with trial types combined
            res = switch_detection_combined(pf_res, all_fields, pf_fr, index_within_to_trial_index_df, spk_beh_df, changes_df_both_trialtype=changes_df_both_trialtype,task_ind=task_ind)
            
            pos_to_time_func_per_trial = res['pos_to_time_func_per_trial']
            pf_loc_combined= res['pf_loc_combined']
            changes_df_combined_d = res['changes_df_combined_d']
            all_sw_times_v_combined = res['all_sw_times_v_combined']
            all_fields_times_v = res['all_fields_times_v']
            
            # shuffle changes_df
            all_sw_times_v_combined_shuffle_l = shuffle_changes_df_combined_get_sw_times_v_combined_shuffle(changes_df_combined_d,index_within_to_trial_index_df,pos_to_time_func_per_trial,pf_loc_combined,nrepeats=nrepeats_sw,task_ind=task_ind)
            
            test_res_df_onoff_d = {}
            c_shuffle_d_onoff_d = {}
            for onoff in [1,-1]:
                test_res_df_onoff,c_shuffle_d_onoff = sweep_test_coswitch(all_sw_times_v_combined,all_sw_times_v_combined_shuffle_l,edges,all_fields_times_v=all_fields_times_v,onoff=onoff)
                test_res_df_onoff_d[onoff] = test_res_df_onoff
                c_shuffle_d_onoff_d[onoff] = c_shuffle_d_onoff
            test_res_df_onoff_d = pd.concat(test_res_df_onoff_d,axis=0)
    
    test_res = {'test_res_df':test_res_df_onoff_d,'all_sw_times_v_combined_shuffle_l':all_sw_times_v_combined_shuffle_l,'count_shuffle':c_shuffle_d_onoff_d,**res}

    # next: write save functions, then explore spatial windows
    misc.save_res(save_fn,test_res,dosave=dosave)

    return test_res

    

def switch_count_with_beh_var_over_time(all_sw_times_v_combined,spk_beh_df,task_index=0,beh_vars = ['v','lin_binned','theta_phase','theta_amp','trial']):
    '''

    sw_beh_df: ntime x [beh_vars, 1, -1], treating switch counts within a time window as spike
    '''
    spk_beh_df = spk_beh_df.query('task_index==@task_index')
    beh_vars = [xx for xx in beh_vars if xx in spk_beh_df.columns ]
    index=spk_beh_df.index
    dt = np.median(np.diff(index))
    index=np.append(index-dt,index[-1]+dt)
    sw_beh_df = spk_beh_df[beh_vars]
    for onoff in [1,-1]:
        df = all_sw_times_v_combined.query('switch==@onoff')
        count,_=np.histogram(df['time'],index)
        
        sw_beh_df[onoff] = count
    return sw_beh_df

def get_total_pf_per_pos_bin(pf_all_field_combined,spk_beh_df,bins=10,task_ind=0,pos_key='peak',doplot=False):
    '''
    get the number of place fields per position bin
    '''
    spk_beh_df_ta=spk_beh_df.query('task_index==@task_ind')
    gpb=spk_beh_df.groupby('trial_type')
    pf_xy_trialtype = {}
    for (tsk_ind,tt_ind), val in gpb:
        
        if task_ind==task_ind: # for now just do one task_ind
            val['lin_binned']
            lin_binned = val['lin_binned'].values
            xy = val[['x','y']].values

            common_ma = pd.notna(lin_binned)
            func = scipy.interpolate.interp1d(lin_binned[common_ma].astype(float),xy[common_ma].astype(float),axis=0)
            if tt_ind ==0:
                pf_all_field_combined_sub =pf_all_field_combined.loc[[tt_ind,'both']]
            else:
                pf_all_field_combined_sub =pf_all_field_combined.loc[tt_ind]
            pf_xy = func(pf_all_field_combined_sub[pos_key].values.astype(float))
            pf_xy_trialtype[(task_ind,tt_ind)]=pd.DataFrame(pf_xy)
        pdb.set_trace()
    pf_xy_trialtype = pd.concat(pf_xy_trialtype.values(),keys=pf_xy_trialtype.keys())
    # count,edges_x,edges_y = np.histogram2d(pf_xy[:,0],pf_xy[:,1],bins=bins)
    pf_xy_trialtype = pf_xy_trialtype.loc[task_ind].values
    count,edges_x,edges_y = np.histogram2d(pf_xy_trialtype[:,0],pf_xy_trialtype[:,1],bins=bins)

    if doplot:
        fig,ax=plt.subplots(1,1,figsize=(6,4))
        # h=ax.hist2d(pf_xy[:,0],pf_xy[:,1],bins=bins)
        h=ax.hist2d(pf_xy_trialtype[:,0],pf_xy_trialtype[:,1],bins=bins)
        fig.colorbar(h[3],ax=ax)
        ax.set_title('# of fields')
        
    
        return count,edges_x,edges_y, fig,ax
    else:
        return count,edges_x,edges_y







def switch_count_by_pos2d(sw_beh_df,bins=10,doplot=False):
    '''
    count the number of switches within each position bin 2d
    '''
    sw_triggered_pos2d_d = {}
    count_d = {}
    for onoff in [1,-1]:
        sw_triggered_pos2d = []
        df = sw_beh_df[['x','y',onoff]]
        df = df.loc[df[onoff]>0]
        df_more_than_one_sw = df.loc[df[onoff]>1]
        for ind,row in df_more_than_one_sw.iterrows():
            sw_triggered_pos2d.extend([row[['x','y']] for _ in range(int(row[onoff]))])
        sw_triggered_pos2d = pd.concat(sw_triggered_pos2d,axis=1).T
        sw_triggered_pos2d = pd.concat([df[['x','y']],sw_triggered_pos2d],axis=0)
        sw_triggered_pos2d_d[onoff] = sw_triggered_pos2d

        count,edges_x,edges_y=np.histogram2d(sw_triggered_pos2d['x'],sw_triggered_pos2d['y'],bins=bins)
        count_d[onoff] = count
    
    if doplot:
        fig,axs=plt.subplots(2,1,figsize=(6,8))
        for ii,onoff in enumerate([1,-1]):
            h=axs[ii].hist2d(sw_triggered_pos2d_d[onoff]['x'],sw_triggered_pos2d_d[onoff]['y'],bins=bins)
            fig.colorbar(h[3],ax=axs[ii])
            axs[ii].set_title(onoff)
        plt.tight_layout()
    
        return count_d,edges_x,edges_y,sw_triggered_pos2d_d, fig,axs
    else:
        return count_d,edges_x,edges_y,sw_triggered_pos2d_d

