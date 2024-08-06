'''
gathering switches with relevant metrics so that can visualize the distribution
and select examples
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
sys.path.append('/mnt/home/szheng/projects/util_code')
sys.path.append('/mnt/home/szheng/projects/nmf_analysis/scripts')
import data_prep_pyn as dpp
import place_cell_analysis as pa
import place_field_analysis as pf
import switch_analysis_one_session as saos
import change_point_analysis_central_arm_seperate as cpacas
import process_central_arm as pca 
import plot_helper as ph
reload(ph)

def prep_load_one_sess(data_dir_full):
    prep_res = dpp.load_spk_beh_df(data_dir_full,force_reload=False,extra_load=dict(filtered='*thetaFiltered.lfp.mat'))
    spk_beh_df=prep_res['spk_beh_df']
    _,spk_beh_df = dpp.group_into_trialtype(spk_beh_df)
    # spk_beh_df,_=dpp.add_lin_binned(spk_beh_df,nbins=100)
    spk_beh_df,_=dpp.add_lin_binned(spk_beh_df,bin_size=2.2,nbins=None)
    cell_cols_d = prep_res['cell_cols_d']

    cell_cols = cell_cols_d['pyr']
    fr_map_dict=pa.get_fr_map_trial(spk_beh_df,cell_cols,trialtype_key='trial_type',speed_key='v',speed_thresh=1.,order=['smooth','divide'])
    fr_map_trial_d = {k:val[0] for k,val in fr_map_dict.items()}
    fr_map_trial_df_d=pd.concat({k:pf.fr_map_trial_to_df(fr_map_trial_d[k],cell_cols) for k in fr_map_dict.keys()},axis=0)



    pf_res =pf.field_detection_both_avg_trial_wrapper(data_dir_full, dosave=True,force_reload=False,nbins = 100, 
                                            save_fn = 'place_field_avg_and_trial_vthresh.p', 
                                            shuffle_fn='fr_map_null_trialtype_vthresh.p',
                                            smth_in_bin=2.5, speed_thresh=1.,speed_key='v',load_only=True
                                            )

    switch_res = saos.switch_analysis_one_session(data_dir_full,place_field_res=None,force_reload=False,nrepeats=1000,
                                                  save_fn=saos.SAVE_FN(saos.FR_KEY),load_only=True)
    
    return spk_beh_df,cell_cols_d,fr_map_trial_df_d,pf_res,switch_res

def get_fr_info(key,fr_trial,switch_trial):
    '''
    switch related fr info
    mean fr pre/post n trial relative to switch, possibly noramlzied by max_fr 
    '''
    key_split = key.split('_')
    pp = key_split[0]

    n = int(key_split[3]) if key_split[3]!='all' else key_split[3]
    do_norm = 'norm' in key
    
    trial_index = fr_trial.index

    switch_trial_iloc = np.nonzero(trial_index == switch_trial)[0][0]
    ntrials = len(trial_index)
    if pp=='pre':
        ed = switch_trial_iloc
        if n=='all':
            st = 0
        else:
            if switch_trial_iloc - n < 0: # exceeding bound, no double counting
                return np.nan 
            else:
                st = switch_trial_iloc - n
    elif pp =='post':
        st = switch_trial_iloc
        if n=='all':
            ed = ntrials
        else:
            if switch_trial_iloc + n > ntrials:
                return np.nan
            else:
                ed = switch_trial_iloc + n
    
    mean_fr = fr_trial.iloc[st:ed].mean()
    if do_norm:
        fr_max = fr_trial.max()
        mean_fr = mean_fr / fr_max 
        
    return mean_fr


# def get_ntrial_info(key,fr_trial,switch_trial):
def get_ntrial_info(key,fr_trial,switch_trial,pre_ntrials_in_segment,post_ntrials_in_segment):
    '''
    get info about how many trials pre/post are greater/less than a threshold relative to the max
    possibly trials are fractions
    key: eg post_ntrial_ge_90_perc_frac_total
    '''
    key_split = key.split('_')
    pp = key_split[0]
    gl = key_split[2]
    thresh = int(key_split[3]) / 100
    do_frac_total = 'frac_total' in key
    do_frac_segment = 'frac_segment' in key
    ntrials = len(fr_trial)
    fr_max = fr_trial.max()
    fr_thresh = fr_max * thresh
    if pp=='pre':
        fr_segment = fr_trial[(switch_trial-pre_ntrials_in_segment):switch_trial]
    elif pp=='post':
        fr_segment = fr_trial[switch_trial:(switch_trial+post_ntrials_in_segment)]
    segment_len = fr_segment.shape[0]
    compare_func = {'ge':np.greater_equal,'le':np.less_equal}[gl]
    N = compare_func(fr_segment,fr_thresh).sum()
    if do_frac_total:
        N = N / ntrials
    elif do_frac_segment:
        N = N / segment_len
    
    return N
        

def add_switch_trial_by_trialtype(all_sw_times_v_combined,trial_index_to_index_within_df,task_ind=0):
    '''
    both: use trial index; 0/1 use index within; add fraction
    '''
    # assign index within
    
    inds = trial_index_to_index_within_df.loc[task_ind].index.get_level_values(1).get_indexer(all_sw_times_v_combined['trial_index'].values)
    all_sw_times_v_combined['index_within'] = trial_index_to_index_within_df.loc[task_ind].iloc[inds].values # .loc[task_ind] is important here to match the line above

    # assign switch trial: for "both" fields, use trial index; for 0/1 fields, use index within
    field_masks = {k:all_sw_times_v_combined['trialtype']==k for k in all_sw_times_v_combined['trialtype'].unique()}
    try:
        both_mask= field_masks['both']
        all_sw_times_v_combined.loc[both_mask,'switch_trial'] = all_sw_times_v_combined.loc[both_mask,'trial_index'].values
        all_sw_times_v_combined.loc[~both_mask,'switch_trial'] = all_sw_times_v_combined.loc[~both_mask,'index_within'].values
    except: # in the case of linearmaze
        all_sw_times_v_combined['switch_trial'] = all_sw_times_v_combined['index_within']

    # normalize trial
    tt_l = trial_index_to_index_within_df.loc[task_ind].index.get_level_values(0).unique()
    n_trial_tot = trial_index_to_index_within_df.loc[task_ind].index.get_level_values(-1).max() + 1
    trial_len_d = {'both': n_trial_tot,
                }
    normalize_len = np.zeros(all_sw_times_v_combined.shape[0])
    try:
        normalize_len[both_mask] = trial_len_d['both']
    except:
        pass

    for tt in tt_l:
        tt_trial_len = trial_index_to_index_within_df.loc[task_ind,tt].values.max()+1
        trial_len_d[tt] = tt_trial_len
        if tt in field_masks.keys():
            normalize_len[field_masks[tt]] = trial_len_d[tt]

    all_sw_times_v_combined['switch_trial_frac'] = all_sw_times_v_combined['switch_trial'] / normalize_len

    ntrials_d = trial_index_to_index_within_df.loc[task_ind].groupby(level=0).count()
    ntrials_d['both'] = ntrials_d.sum()
    gpb=all_sw_times_v_combined.groupby(['trialtype','uid','field_index'])
    pre_ntrials_in_segment = []
    post_ntrials_in_segment = []
    for k,val in gpb:
        # ntrials = ntrials_d[task_ind,k[0]]
        ntrials = ntrials_d[k[0]]
        seg_ntrials = np.diff(val['switch_trial'],prepend=0,append=ntrials).astype(int)
        pre_ntrials_in_segment_onefield = seg_ntrials[:-1]
        post_ntrials_in_segment_onefield = seg_ntrials[1:]
        pre_ntrials_in_segment_onefield = pd.Series(pre_ntrials_in_segment_onefield,val.index)
        pre_ntrials_in_segment.append(pre_ntrials_in_segment_onefield)
        post_ntrials_in_segment_onefield = pd.Series(post_ntrials_in_segment_onefield,val.index)
        post_ntrials_in_segment.append(post_ntrials_in_segment_onefield) 
    pre_ntrials_in_segment = pd.concat(pre_ntrials_in_segment)
    post_ntrials_in_segment = pd.concat(post_ntrials_in_segment)
    all_sw_times_v_combined['pre_ntrials_in_segment'] = pre_ntrials_in_segment
    all_sw_times_v_combined['post_ntrials_in_segment'] = post_ntrials_in_segment

    return all_sw_times_v_combined



import itertools
from itertools import product

def add_metrics(all_sw_times_v_combined,pf_fr_combined,trial_index_to_index_within_df_onetask):

    # get all keys for metrics to be added
    fr_info_keys = [f'{pp}_mean_fr_{n}' for (pp,n) in itertools.product(['pre','post'],[1,2,3,'all'])] # eg pre_mean_fr_1
    fr_info_keys =  fr_info_keys + [p+'_norm' for p in fr_info_keys]

    ntrial_info_keys = [f'{pp}_ntrial_{lg}_{n}_perc' for (pp,lg,n) in product(['pre','post'],['le','ge'],[10,30,50,70,90])]
    ntrial_info_keys = ntrial_info_keys +  [k+'_frac_segment' for k in ntrial_info_keys] + [k+'_frac_total' for k in ntrial_info_keys]

    extra_info_keys = fr_info_keys + ntrial_info_keys
    extra_info = {k:[] for k in extra_info_keys}

    for i,row in all_sw_times_v_combined.iterrows():
        trialtype = row['trialtype']
        uid = row['uid']
        field_index = row['field_index']
        # sw_trial = row['trial_index'] # different from the 'switch_trial' in the df; here want all fields to use trial_index, bc that's how pf_fr_combined is columned
        sw_trial = int(row['switch_trial'])
        fr_across_trials = pf_fr_combined.loc[(trialtype,uid,field_index)].dropna()
        if trialtype!='both': # then reindex into trial within
            trialindex = fr_across_trials.index
            fr_across_trials.index = trial_index_to_index_within_df_onetask.loc[slice(None),trialindex].values
        pre_ntrials_in_segment = int(row['pre_ntrials_in_segment'])
        post_ntrials_in_segment = int(row['post_ntrials_in_segment'])

        for key in fr_info_keys:
            extra_info[key].append(get_fr_info(key,fr_across_trials,sw_trial))
        for key in ntrial_info_keys:
            extra_info[key].append(get_ntrial_info(key,fr_across_trials,sw_trial,pre_ntrials_in_segment,post_ntrials_in_segment))


    extra_info = pd.DataFrame(extra_info)
    all_sw_info =pd.concat([all_sw_times_v_combined.reset_index(drop=True),extra_info],axis=1)
    return all_sw_info

def get_all_sw_add_metrics_all_tasks(sw_res,pf_params_recombined,spk_beh_df,is_changes_df=False,do_add_metrics=True,pf_loc_key='peak',pf_fr_key='fr_peak'):
    trial_index_to_index_within_df = dpp.trial_index_to_index_within_trialtype(spk_beh_df)
    task_ind_l = spk_beh_df['task_index'].unique()
    all_sw_with_metrics_d = {}
    all_sw_d = {}
    if is_changes_df:
        cdf = sw_res
    else:
        cdf = sw_res['changes_df']
    for task_ind in task_ind_l:
        trial_pos_info = pca.get_trial_pos_info(spk_beh_df,speed_key = 'directed_locomotion',speed_thresh = 0.5,n_lin_bins=100,task_ind=task_ind)
        pos_to_time_func_per_trial = pca.get_pos_to_time_func_per_trial(trial_pos_info)
        changes_df_combined = cdf.loc[task_ind]
        pf_loc_combined = pf_params_recombined.loc[pf_loc_key].loc[task_ind]
        all_sw_v_one_task = pca.get_all_switch_times_combined(pos_to_time_func_per_trial,pf_loc_combined,changes_df_combined)
        all_sw_v_one_task = all_sw_v_one_task.reset_index(drop=True)
        all_sw_v_one_task = add_switch_trial_by_trialtype(all_sw_v_one_task,trial_index_to_index_within_df,task_ind=task_ind)
        all_sw_d[task_ind] = all_sw_v_one_task
        if do_add_metrics:
            # pdb.set_trace()
            all_sw_v_one_task_with_metrics = add_metrics(all_sw_v_one_task,pf_params_recombined.loc[pf_fr_key].loc[task_ind],trial_index_to_index_within_df.loc[task_ind])
        else:
            all_sw_v_one_task_with_metrics = all_sw_v_one_task
        all_sw_with_metrics_d[task_ind] = all_sw_v_one_task_with_metrics#all_sw_v_one_task
        
        
    all_sw_d = pd.concat(all_sw_d,axis=0)
    all_sw_with_metrics_d = pd.concat(all_sw_with_metrics_d,axis=0)
    return all_sw_d, all_sw_with_metrics_d


####### for OLD code#####
def add_switch_metrics_wrapper(data_dir_full,
                                dosave=True, save_dir='switch_analysis',save_fn='all_switch_info.p',
                                force_reload=False,
                                load_only=True,
                                **kwargs
                                ):
    kwargs_ = {'fr_key':'fr_peak','task_ind':0}                                
    kwargs_.update(kwargs)
    fr_key = kwargs_['fr_key']
    task_ind = kwargs_['task_ind']
    
    # create subdir
    save_dir = misc.get_or_create_subdir(data_dir_full,'py_data',save_dir)
    save_fn, res = misc.get_res(save_dir,save_fn,force_reload)
    if (res is not None) or load_only: # load only would skip the computation that follows
        return res
    

    spk_beh_df,cell_cols_d,fr_map_trial_df_d,pf_res,switch_res = prep_load_one_sess(data_dir_full)
    test_res = cpacas.sweep_test_coswitch_wrapper(data_dir_full,
                                pf_res_save_fn='place_field_avg_and_trial_vthresh.p',
                                pf_shuffle_fn = 'fr_map_null_trialtype_vthresh.p',
                                speed_key='v',fr_key=fr_key,
                                bin_size=2.2,
                                switch_res_query=(slice(None),0.3,'switch_magnitude',0.4),
                                nrepeats_sw = 2,
                                edges = None,
                                save_fn = 'switch_res_window.p',
                                load_only=False,
                                dosave=False,force_reload=False,
                                task_ind = task_ind,
                                prep_force_reload=False,
                                ) # if exists, load; if not compute, but only care about all_sw_times_v_combined, so nrepeats_sw low, and not save
    all_sw_times_v_combined = test_res['all_sw_times_v_combined']
    changes_df_combined = test_res['changes_df_combined']


    # get pf_fr_combined
    trial_index_to_index_within_df = dpp.trial_index_to_index_within_trialtype(spk_beh_df)
    index_within_to_trial_index_df = dpp.index_within_to_trial_index(spk_beh_df)
    X_raw = pd.concat(switch_res['avg']['X']['raw'],axis=0)
    pf_fr = pd.concat(pf_res['avg']['params'],axis=0).loc[task_ind].loc[(slice(None),fr_key),:]
    pf_fr = pf_fr.droplevel(1)
    all_fields = pd.concat(pf_res['avg']['all_fields'],axis=0).loc[task_ind]
    # ugly to have this here, but needed; more principled way would be to combined all the pf_res, and not just changes_df, when combining two trialtypes
    kwargs = dict(switch_magnitude=0.4,loc_key='peak',pen=0.3,min_size=1)
    splitter_fields,nonsplitter_fields, pf_fr_trialtype_combined_splitter, pf_fr_trialtype_combined_nonsplitter = pca.divide_central_fields_splitter_gather_params(all_fields,pf_fr,index_within_to_trial_index_df,task_ind=task_ind,**kwargs)
    pf_fr_combined,pf_all_field_combined = pca.combine_field_loc(pf_res,nonsplitter_fields,index_within_to_trial_index_df,loc_key=fr_key,task_ind = task_ind)

    all_sw_times_v_combined = add_switch_trial_by_trialtype(all_sw_times_v_combined,trial_index_to_index_within_df,task_ind=0)

    all_sw_info = add_metrics(all_sw_times_v_combined,pf_fr_combined)

    res = {'all_sw_info':all_sw_info,
          'pf_fr_combined':pf_fr_combined,
          'pf_all_field_combined':pf_all_field_combined,
          'fr_map_trial_df_d':fr_map_trial_df_d,
          'changes_df_combined':changes_df_combined,
          'spk_beh_df':spk_beh_df,
          'cell_cols_d':cell_cols_d,
          'fr_map_trial_df_d':fr_map_trial_df_d
        }

    misc.save_res(save_fn,res,dosave=dosave)
    return res


#===================SPATIAL DISTRIBUTION=============#
reload(ph)
plt.rcParams.update({'font.size':20})

def get_switch_spatial_distribution_one(count_df_pos,all_fields_pos,edges):
    sw_count_per_pos,edges=np.histogram(count_df_pos,bins=edges)
    baseline_count,edges=np.histogram(all_fields_pos,bins = edges)
    frac_per_pos = sw_count_per_pos / baseline_count
    return sw_count_per_pos, baseline_count,edges, frac_per_pos

def bootstrap_switch_spatial_distribution_one(count_df_pos,all_fields_pos,edges,nrepeats=100,sample_baseline=True):
    frac_sample_all = []
    sw_count_sample_all = []
    baseline_count_sample_all = []
    for i in range(nrepeats):
        count_sample = count_df_pos.sample(frac=1,replace=True)
        if sample_baseline:
            all_fields_sample = all_fields_pos.sample(frac=1,replace=True)
        else:
            all_fields_sample = all_fields_pos
        sw_count_sample,baseline_count_sample,_,frac_sample=get_switch_spatial_distribution_one(count_sample,all_fields_sample,edges)
        frac_sample_all.append(frac_sample)
        sw_count_sample_all.append(sw_count_sample)
        baseline_count_sample_all.append(baseline_count_sample)
    frac_sample_all = np.array(frac_sample_all)
    sw_count_sample_all = np.array(sw_count_sample_all)
    baseline_count_sample_all = np.array(baseline_count_sample_all)
    return sw_count_sample_all,baseline_count_sample_all,edges,frac_sample_all

def plot_switch_field_ratio(all_sw,all_fields,edges=None,field_loc_key='peak',n_bootstrap_repeats=None,ci=0.95,fig=None,axs=None,doplot=False):
    '''
    get the spatial distribution of switching, place fields, and the ratio
    
    all_sw: df: n_switchs x [info]
    all_field: df: n_fields x [info]
    can have multiindex
    '''
    sw_count_per_pos_d = {}
    frac_per_pos_d = {}
    frac_sample_all_d = {}
    count_df_d = {}
    
    if edges is None:
        edges = np.array([0,6.8,20,33.6,42,50.5,67,84,100])
    edges_center = (edges[:-1]+edges[1:])/2
    for onoff in [1,-1]:
        count_df = all_sw.loc[all_sw['switch']==onoff]
        count_df_d[onoff] = count_df
        sw_count_per_pos, baseline_count,edges, frac_per_pos=get_switch_spatial_distribution_one(count_df['field_pos'],all_fields[field_loc_key],edges)
        sw_count_per_pos_d[onoff] = sw_count_per_pos
        frac_per_pos_d[onoff] = frac_per_pos
        if n_bootstrap_repeats is not None:
            sw_count_sample_all,baseline_count_sample_all,edges,frac_sample_all = bootstrap_switch_spatial_distribution_one(count_df['field_pos'],all_fields[field_loc_key],edges,nrepeats=n_bootstrap_repeats)
            frac_sample_all_d[onoff] = frac_sample_all
    if doplot:
        if axs is None:
            fig,axs = plt.subplots(1,2,figsize=(20,6))
            for ii,onoff in enumerate([1,-1]):
                if n_bootstrap_repeats is None:
                    axs[ii].plot(edges_center,frac_per_pos_d[onoff],color='k',marker='o')
                else:
                    
                    ph.mean_bootstraperror_lineplot(frac_sample_all_d[onoff],data=frac_per_pos_d[onoff],xs=edges_center,ci=ci,fig=fig,ax=axs[ii],color='k',marker='o')
                axs[ii].set_ylabel('num. switch / num. place fields')
                axs[ii].set_title(onoff)
                ax2 = axs[ii].twinx()
                ax2.hist(all_fields[field_loc_key],bins = edges,density=True,alpha=0.3,label='num. place fields')
                ax2.hist(count_df_d[onoff]['field_pos'],bins = edges,density=True,alpha=0.3,label='num. switch')
                ax2.legend(bbox_to_anchor=[1,1.3])
                ph.plot_section_markers(fig=fig,ax=axs[ii])

        plt.tight_layout()
                
        return sw_count_per_pos_d, baseline_count, frac_per_pos_d, edges, fig,axs

        
    return sw_count_per_pos_d, baseline_count, frac_per_pos_d, edges


#========Example selection=========#
def select_fields_using_info(all_sw_info,info_key_d,nsamples=5,sample_by_ani=True):
    ma = np.ones(all_sw_info.shape[0]).astype(bool)
    for k,val in info_key_d.items():
        ma_ = all_sw_info[k].between(val[0],val[1])
        ma = np.logical_and(ma,ma_)
    all_sw_info_sub = all_sw_info.loc[ma]
    if sample_by_ani:
        all_sw_info_sample = all_sw_info_sub.groupby(level=0).sample(nsamples,replace=True)
        all_sw_info_sample = all_sw_info_sample.drop_duplicates()
    else:
        all_sw_info_sample=all_sw_info_sub.sample(nsamples)

    c1=all_sw_info_sub.groupby(level=0).count().iloc[:,0] 
    c2=all_sw_info.groupby(level=0).count().iloc[:,0]
    common_ind = c1.index.intersection(c2.index)
    frac_by_ani = c1.loc[common_ind] / c2.loc[common_ind]

    return all_sw_info_sample, all_sw_info_sub, frac_by_ani

def plot_example_ratemaps_and_rates(all_sw_info_onoff_sample,
                                    pf_all_field_combined_all,
                                    fr_map_trial_df_d_all,
                                    trial_index_to_index_within_trialtype_all,
                                    X_raw_all,
                                    X_pwc_all,
                                    quality='',onoff_str='on',
                                    save_dir_root="/mnt/home/szheng/ceph/place_variability/fig/general",task_ind = 0):
    gpb = all_sw_info_onoff_sample.groupby(level=0)
    for ani,val in gpb:
        val = val.loc[ani]
        nplots = val.shape[0]
        save_dir_full_ani = misc.get_or_create_subdir(save_dir_root,ani)
        for ii,(sess,row) in enumerate(val.iterrows()):
            sess = sess[0]
            save_dir_full_sess = misc.get_or_create_subdir(save_dir_full_ani,sess,f'switch_{onoff_str}_{quality}')
            fig,axs = plt.subplots(1, 2,figsize=(6*2,1*6),sharey=True,squeeze=True)
            ax = axs[0]
            trialtype = row['trialtype']
            uid = row['uid']
            field_index = row['field_index']
            index_within = row['index_within']
            trial_index = row['trial_index']

            pf_info=pf_all_field_combined_all.loc[ani,sess,trialtype,uid,field_index]        
            field_bound = (pf_info['start'],pf_info['end'])
            if trialtype=='both':
                trialtype = trial_index_to_index_within_trialtype_all[ani,sess].loc[task_ind].loc[slice(None),trial_index].index[0]
            data = fr_map_trial_df_d_all[ani,sess].loc[task_ind,trialtype,uid].dropna(axis=1).T           
            if data.isna().any().any():
                break
            fig,ax=ph.ratemap_one_raw(data,trial=index_within,field_bound=field_bound,fig=fig,ax=ax,title=(uid,field_index))
            
            # plot line
            ax = axs[1]
            xx=X_raw_all[ani,sess][task_ind,trialtype].dropna(axis=1).loc[uid,field_index]
            ax.plot(xx.values,xx.index,marker='o')
            xx=X_pwc_all[ani,sess][task_ind,trialtype].dropna(axis=1).loc[uid,field_index]
            ax.plot(xx.values,xx.index)
            plt.tight_layout()
            fn_full = os.path.join(save_dir_full_sess,f'uid_{uid}_field_{field_index}_trialtype_{trialtype}_trialwithin_{int(index_within)}.pdf')
            fig.savefig(fn_full)
            plt.close(fig)