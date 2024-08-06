import ruptures as rpt
import numpy as np
import pandas as pd
import scipy
import os,sys,copy,itertools,pdb,importlib
sys.path.append('/mnt/home/szheng/projects/nmf_analysis/')
import change_point_plot as cpp
importlib.reload(cpp)
import matplotlib.pyplot as plt
import seaborn as sns
import process_central_arm as pca
import change_point_analysis as cpa
import data_prep_pyn as dpp
import place_cell_analysis as pa
import place_field_analysis as pf
import switch_analysis_one_session as saos
import misc
import plot_helper as ph

def select_co_switch_pairs(all_sw_times_v_combined,thresh=1.,reference_field=None,nsamples=1,time_diff=None,onoff=1,diff_key='time',same_trialtype_only=False):
    '''
    all_sw_times_v_combined: from pca.get_all_switch_times_combined
    select a reference_field, if not given, and sample some fields whose switching occur within thresh relative to its switching.

    '''
    all_sw_times_v_combined_onoff = all_sw_times_v_combined.query('switch==@onoff')
    identity_columns=['trialtype','uid','field_index','trial_index']
    if time_diff is None:
        times = all_sw_times_v_combined_onoff[diff_key].values
        time_diff = np.abs(times[:,None] - times[None,:])
    if reference_field is not None:
        if isinstance(reference_field, tuple):
            ma = (all_sw_times_v_combined_onoff['trialtype']==reference_field[0])&(all_sw_times_v_combined_onoff['uid']==reference_field[1])&(all_sw_times_v_combined_onoff['field_index']==reference_field[2])
            # num index of the reference field within the difference matrix      
            reference_ind = np.nonzero(ma.values)[0] # n_occurances, 
    else:
        # numerical indices of the potential reference fields
        potential_reference_field_numind_l = np.nonzero(time_diff <= thresh)[0]
        reference_ind = np.random.choice(potential_reference_field_numind_l)
        reference_field = tuple(all_sw_times_v_combined_onoff.iloc[reference_ind][identity_columns])
        
    
    # compare the row of the reference field with thresh
    selection_mask = time_diff[reference_ind] <= thresh  
    # OR on the rows, i.e. any other field that co switch with any of the switches of the reference field
    if len(selection_mask.shape)>1:
        selection_mask = np.any(selection_mask,axis=0)

    fields_available_for_selection = all_sw_times_v_combined_onoff[identity_columns].loc[selection_mask]
    not_self_mask = [x!=reference_field for x in fields_available_for_selection.itertuples(index=False,name=None)]
    fields_available_for_selection = fields_available_for_selection.loc[not_self_mask]
    if same_trialtype_only:
        reference_trialtype = reference_field[0]
        fields_available_for_selection = fields_available_for_selection.query('trialtype==@reference_trialtype')


    navailables=fields_available_for_selection.shape[0]
    if navailables>0:
        nsamples = np.minimum(nsamples,navailables)
        sampled_fields=fields_available_for_selection.sample(nsamples)
        
    else:
        sampled_fields = None
    reference_field = pd.DataFrame(list(reference_field),index=identity_columns).T
    return sampled_fields,reference_field,fields_available_for_selection
        
    
    
def get_fields_with_lasting_switching(changes_df,all_fields, lasting_trial_thresh=2,onoff=1):
    '''
    selecting fields whose switching lasts for at least lasting_trial_thresh trials; the last trial also counts as a "lagging" event, i.e. only switches x trials before the last trial will count
    onoff: if 1, get switch on that lasts; if -1, switch off that lasts
    '''
    field_l = []
    ntrials = changes_df.shape[1]
    for (field,ind),row in changes_df.iterrows():
        row = row.dropna() # this way can use changes_df_combined! still need to make consistent with all_fields though, because right now all_fields is trialtype specific, but changes_df_combined is based on trialtype=0
        lead = np.nonzero((row==onoff).values)[0]
        lag = np.nonzero((row==-onoff).values)[0]
        lag = np.append(lag,ntrials)
        if len(lead)>0:
            # if len(lag)==0:
            #     for o in lead:

            #         field_info=all_fields.loc[field,ind]
            #         field_info['trial']=o
            #         field_l.append(field_info)

            # else:
            for o in lead:
                trial_diff = lag - o
                if np.all( (trial_diff>=lasting_trial_thresh) | (trial_diff<0)): # all lagging events either occur before or some trials after the leading event
                    field_info=all_fields.loc[field,ind]
                    field_info['trial']=o
                    field_l.append(field_info)
    field_l=pd.DataFrame(field_l)
    return field_l

def get_fields_with_lasting_switching_wrapper(switch_res,pf_res, lasting_trial_thresh=3,task_ind=0):
    '''
    wrapper: aimed for quick and dirty local use
    '''
    field_l_d_bothtrialtypes = {}
    for tt_ind in [0,1]:
        # careful!! the indices might change for switch_res and pf_res!!!
        changes_df = switch_res['avg']['changes_df'].loc[task_ind,tt_ind,0.3,'switch_magnitude',0.4].dropna(axis=1)
        all_fields=pf_res['avg']['all_fields'][task_ind,tt_ind]
        field_l_d = {}
        for onoff in [1,-1]:
            field_l_d[onoff] = get_fields_with_lasting_switching(changes_df,all_fields, lasting_trial_thresh=3,onoff=onoff)
        field_l_d_bothtrialtypes[tt_ind] = field_l_d
    return field_l_d_bothtrialtypes

#=====plotting=====###

def plot_co_switch_ratemaps(selected_fields,fr_map_concat_task,switch_trial=None,reference_field_range=None,fig=None,ax=None):
    lw = 3
    ls=':'
    nplots = selected_fields.shape[0]
    fig,axs=ph.subplots_wrapper(nplots,squeeze=False)
    for ii,(_,field) in enumerate(selected_fields.iterrows()):
        ax = axs.ravel()[ii]
        uid= field['uid']
        ax.imshow(fr_map_concat_task.loc[uid].T,aspect='auto')
        if switch_trial is not None:
            ax.hlines(switch_trial,*ax.get_xlim(),linewidth=lw,linestyle=ls,color='C0')
        if reference_field_range is not None:
            ax.vlines(reference_field_range[0],*ax.get_ylim(),linewidth=lw,linestyle=ls,color='C1')
            ax.vlines(reference_field_range[1],*ax.get_ylim(),linewidth=lw,linestyle=ls,color='C1')
        ax.set_title(tuple(field[['trialtype','uid','field_index']].values))
    return fig,axs

def plot_co_switch_waveforms(cell_metrics,selected_fields,fig=None,axs=None,plot_all_channels=False):
    nplots = selected_fields.shape[0]
    if axs is None:
        fig,axs=ph.subplots_wrapper(nplots,squeeze=False)
    
    vmin = np.inf
    vmax =-np.inf
    if plot_all_channels:
        for uid in selected_fields['uid']:
            uid_ind = np.nonzero(cell_metrics['UID']==uid)[0][0]
            wf = cell_metrics['waveforms'][uid_ind]['raw_all']
            vmin = np.minimum(np.min(wf),vmin)
            vmax = np.maximum(np.max(wf),vmax)
        vmin = vmin * 0.9
        vmax = vmax * 0.9

    for ii,uid in enumerate(selected_fields['uid']):
        uid_ind = np.nonzero(cell_metrics['UID']==uid)[0][0]
        time= cell_metrics['waveforms'][uid_ind]['time']
        ax = axs.ravel()[ii]
        if plot_all_channels:
            wf = cell_metrics['waveforms'][uid_ind]['raw_all']
            sns.heatmap(wf,cmap='vlag',ax=ax,center=0,vmin=vmin,vmax=vmax)
        else:
            wf = cell_metrics['waveforms'][uid_ind]['raw']
            ax.plot(time,wf)            
        

    return fig,axs


    
def visualize_coswitch_wrapper(all_sw_times_v_combined,field_l_d_bothtrialtypes,cell_metrics,fr_map_trial_df_d,
                                all_fields_all_trialtype=None,
                                trial_index_to_index_within_df=None,
                                task_ind =0,
                                    tt_ind = 0,
                                    coswitch_window = 1.,
                                    onoff = 1,
                                    field_index=0,
                                    reference_field=None,
                                    same_trialtype_only = True,
                                    diff_key='time'
                                ):
    '''
    plot the ratemaps of co-switching fields, relative to a reference field, 
    given using the field_index within field_l, or given directly

    field_l_d_bothtrialtypes: from get_fields_with_lasting_switching_wrapper
        list of fields whose switching last for more than x number of trials
    
    important: same_trialtype_only: if true, only select coswitching fields from the same trial

    PROBLEM: # still some mismatch between field_l (trialtype seperated) and all_sw_times_v_combined (trialtype combined), so sometimes this would return one plot

    '''
    
    
    if reference_field is None:
        if field_index is None:
            reference_field=None
            field = None
        else:
            field_l = field_l_d_bothtrialtypes[tt_ind][onoff] 
            field = field_l.iloc[field_index]
            reference_field = (task_ind,*field.name[:2])
    sampled_fields,reference_field,fields_available_for_selection = select_co_switch_pairs(all_sw_times_v_combined,thresh=coswitch_window,nsamples=1,reference_field=reference_field,same_trialtype_only=same_trialtype_only,time_diff=None,onoff=onoff,diff_key=diff_key)
    
    if field is None:
        assert all_fields_all_trialtype is not None
        assert trial_index_to_index_within_df is not None
        all_fields = all_fields_all_trialtype.loc[task_ind]
        tt_from_combined = reference_field['trialtype'][0]
        if tt_from_combined !='both': # update trialtype index for non "both" fields
            tt_ind = tt_from_combined
        trial_index = reference_field['trial_index'][0]
        index_and_tt = trial_index_to_index_within_df.loc[task_ind,slice(None),trial_index] # series: trialtype: index_within_trialtype
        index_within = index_and_tt.iloc[0]
        tt_ind = index_and_tt.index[0] # update trialtype index for central arm nonsplitter
        field = all_fields.loc[tt_ind,reference_field['uid'][0],reference_field['field_index'][0]]
        field['trial'] = index_within

    # fr_map_concat_task = fr_map_concat.loc[task_ind]
    selected_fields = pd.concat([reference_field,fields_available_for_selection],axis=0)

    fmt=fr_map_trial_df_d.loc[task_ind,tt_ind].dropna(axis=1)
    fig,axs=plot_co_switch_ratemaps(selected_fields,fmt,switch_trial=field['trial'],reference_field_range=field['start':'end'].values)
    # axs.ravel()[0].set_title(f"trial={field['trial']},field={field.name}\ntrialtype={tt_ind}")
    fig,axs=plot_co_switch_waveforms(cell_metrics,selected_fields,fig=None,axs=None)

    fig,axs=plot_co_switch_waveforms(cell_metrics,selected_fields,fig=None,axs=None,plot_all_channels=True)