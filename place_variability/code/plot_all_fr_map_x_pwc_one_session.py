import os
import sys
import traceback
import numpy as np
import scipy.io as sio
# Import other required libraries
import pandas as pd
sys.path.append('/mnt/home/szheng/projects/util_code')
sys.path.append('/mnt/home/szheng/projects/nmf_analysis')
import copy,pdb
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'
import misc
import database 
import data_prep_pyn as dpp
import place_cell_analysis as pa
import place_field_analysis as pf
import plot_helper as ph
import seaborn as sns
# filter db
# subdb = database.db.query("owner=='roman'").sort_values('n_pyr_putative',ascending=False)
subdb = database.db.sort_values('n_pyr_putative',ascending=False)


SAVE_DIR='fr_map_x_pwc'
# SAVE_FN='fr_map.p'
save_fn_func = lambda task,tt,uid,field:f'{task}_{tt}_uid{uid}_{field}.svg'

import pf_recombine_central as pfrc
import fr_map_one_session as fmos
import switch_detection_one_session as sdos
import get_all_switch_add_metrics as gasam

from matplotlib.ticker import MaxNLocator



####========these are for all fields========########

def plot_ratemap_one_neuron_all_fields_using_changes_df(fr_map_trial_df,all_fields,changes_df,index_within_to_trial_index_df,
                                       task_ind,tt,uid,flipy=True,
                                       fig=None,ax=None,shuffle_ratemap=False,vmax_clip_quantile=0.99,cmap='viridis',figsize=(3,2)
                                      ):
    '''
    ratemap for one neuron in one task, trialtype, all fields and change points marked
    '''
    fr_map_trial_df_oneneuron = fr_map_trial_df.loc[(task_ind,tt,uid),:].dropna(axis=1).T
    if ax is None:
        fig,ax=plt.subplots(figsize=figsize)
    index_within_l = index_within_to_trial_index_df.loc[task_ind,tt].index
    fr_map_trial_df_oneneuron=fr_map_trial_df_oneneuron.loc[index_within_l]

    vmax=np.quantile(fr_map_trial_df_oneneuron,vmax_clip_quantile)

    if shuffle_ratemap:
        new_start = np.random.randint(1,fr_map_trial_df_oneneuron.shape[0]-1)
        fr_map_trial_df_oneneuron_roll=np.roll(fr_map_trial_df_oneneuron.values,new_start,axis=0)
        ax=sns.heatmap(fr_map_trial_df_oneneuron_roll,ax=ax,cmap=cmap,vmax=vmax)    
    else:
        ax=sns.heatmap(fr_map_trial_df_oneneuron,ax=ax,cmap=cmap,vmax=vmax)
    
    # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    if flipy:
        ax.invert_yaxis()

    # retrieve all the fields within a trialtype (including "both") for a neuron
    # all_fields_sub = all_fields.loc[(slice(None),slice(None),uid),:]
    all_fields_sub = all_fields.loc[(task_ind,slice(None),uid),:]
    ma=np.array([x in [tt,'both'] for x in all_fields_sub.index.get_level_values(1)])
    all_fields_sub=all_fields_sub.loc[ma]

    # all_sw_d_sub = all_sw_d.loc[task_ind].query('uid==@uid & trialtype in [@tt,"both"]')

    changes_df_sub = changes_df.loc[(task_ind,[tt,'both'],uid),:]
    changes_df_sub = changes_df_sub.dropna(axis=1)
    
    

    for ii,(key,row) in enumerate(all_fields_sub.iterrows()):
        st=row['start']
        ed=row['end']
        # c = f'C{ii}'
        c = 'red'
        c_sw='white'
        tt_field = key[1]
        fig,ax=ph.plot_field_bound(st,ed,ax=ax,fig=fig,c=c)

        field_index=row.name[-1]
        # changes_df_sub_one_field = changes_df_sub.loc[(task_ind,slice(None),uid,field_index),:]
        changes_df_sub_one_field = changes_df_sub.loc[(task_ind,tt_field,uid,field_index),:]
        # changes_df_sub_one_field=changes_df_sub_one_field.dropna(axis=1)
        changes_df_sub_one_field=changes_df_sub_one_field.dropna()
        # index_within_l = np.nonzero(changes_df_sub_one_field.values)[1]
        index_within_l = np.nonzero(changes_df_sub_one_field.values)[0]
        # pdb.set_trace()
        # pdb.set_trace()
        if len(index_within_l)>0:
            # trial_index_l = changes_df_sub_one_field.columns[index_within_l]
            trial_index_l = changes_df_sub_one_field.index[index_within_l]
        # all_sw_d_sub_one_field=all_sw_d_sub.query('field_index==@field_index')
        # if all_sw_d_sub_one_field.shape[0]>0:
            # index_within_l = all_sw_d_sub_one_field['index_within'].values
            # trial_index_l = all_sw_d_sub_one_field['trial_index'].values
            sw_trial_in_tt_ma=np.array([x in index_within_to_trial_index_df.loc[task_ind,tt].values for x in trial_index_l])
            index_within_l_in_tt = index_within_l[sw_trial_in_tt_ma] # need to filter some of the index within, because they don't occur in the trial type, for the "both" fields
            if len(index_within_l_in_tt)>0:
                xlim = (st-5,ed+5)
                fig,ax=ph.plot_switch_trial(index_within_l_in_tt,xlim=xlim,c=c_sw,fig=fig,ax=ax)
    ax.set(xlabel='Position',ylabel='Trial')
    ax.set_xticks([])

    return fig,ax

def plot_ratemap_one_neuron_all_fields(fr_map_trial_df,all_fields,all_sw_d,index_within_to_trial_index_df,
                                       task_ind,tt,uid,flipy=True,vmax_clip_quantile=0.99,
                                       fig=None,ax=None
                                      ):
    '''
    ratemap for one neuron in one task, trialtype, all fields and change points marked
    '''
    fr_map_trial_df_oneneuron = fr_map_trial_df.loc[(task_ind,tt,uid),:].dropna(axis=1).T
    if ax is None:
        fig,ax=plt.subplots()
    index_within_l = index_within_to_trial_index_df.loc[task_ind,tt].index
    fr_map_trial_df_oneneuron=fr_map_trial_df_oneneuron.loc[index_within_l]
    
    vmax=np.quantile(fr_map_trial_df_oneneuron,vmax_clip_quantile)

    ax=sns.heatmap(fr_map_trial_df_oneneuron,ax=ax,cmap='Greys',vmax=vmax)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    if flipy:
        ax.invert_yaxis()

    # retrieve all the fields within a trialtype (including "both") for a neuron
    all_fields_sub = all_fields.loc[(slice(None),slice(None),uid),:]
    ma=np.array([x in [tt,'both'] for x in all_fields_sub.index.get_level_values(1)])
    all_fields_sub=all_fields_sub.loc[ma]

    all_sw_d_sub = all_sw_d.loc[task_ind].query('uid==@uid & trialtype in [@tt,"both"]')

    for ii,(_,row) in enumerate(all_fields_sub.iterrows()):
        st=row['start']
        ed=row['end']
        c = f'C{ii}'
        fig,ax=ph.plot_field_bound(st,ed,ax=ax,fig=fig,c=c)

        field_index=row.name[-1]
        all_sw_d_sub_one_field=all_sw_d_sub.query('field_index==@field_index')
        if all_sw_d_sub_one_field.shape[0]>0:
            index_within_l = all_sw_d_sub_one_field['index_within'].values
            trial_index_l = all_sw_d_sub_one_field['trial_index'].values
            sw_trial_in_tt_ma=np.array([x in index_within_to_trial_index_df.loc[task_ind,tt].values for x in trial_index_l])
            index_within_l_in_tt = index_within_l[sw_trial_in_tt_ma] # need to filter some of the index within, because they don't occur in the trial type, for the "both" fields
            if len(index_within_l_in_tt)>0:
                xlim = (st-5,ed+5)
                fig,ax=ph.plot_switch_trial(index_within_l_in_tt,xlim=xlim,c=c,fig=fig,ax=ax)
    ax.set(xlabel='Pos. bin',ylabel='Trial (within trial type')

    return fig,ax

def plot_x_raw_pwc_one_neuron_all_fields(X_raw,X_pwc,all_fields,index_within_to_trial_index_df,
                                       task_ind,tt,uid,flipy=True,
                                       fig=None,ax=None):
    '''
    plot the within field firing rate and the piece-wise constant fit, for all fields of one neuron, 
    within the task, trialtype
    '''
    
    if ax is None:
        fig,ax=plt.subplots()
    sns.despine()
    # retrieve all the fields within a trialtype (including "both") for a neuron
    all_fields_sub = all_fields.loc[(slice(None),slice(None),uid),:]
    ma=np.array([x in [tt,'both'] for x in all_fields_sub.index.get_level_values(1)])
    all_fields_sub=all_fields_sub.loc[ma]

    index_within_l = index_within_to_trial_index_df.loc[task_ind,tt].index

    for ii,(_,row) in enumerate(all_fields_sub.iterrows()):
        task_ind,tt_including_both,uid,field_id=row.name
        val_pwc = X_pwc.loc[task_ind,tt_including_both,uid,field_id].dropna()
        val = X_raw.loc[task_ind,tt_including_both,uid,field_id].dropna()
        if tt_including_both =='both':
            inds=index_within_to_trial_index_df.loc[(task_ind,tt)]
            val_pwc = val_pwc.loc[inds]
            val = val.loc[inds]
        trials = index_within_l#np.arange(len(val))
        c = f'C{ii}'
        ax.plot(val,trials,marker='o',alpha=0.8,c=c)
        ax.plot(val_pwc,trials,c=c,linewidth=3)
    ax.set(ylabel='Trial (within trial type)')
    ax.set(xlabel='Firing Rate (Hz)')
    # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    if not flipy: # here meaning whether flipy for the heatmap; if heatmap does not need to flip, then this plot need to flip
        ax.invert_yaxis()

    
    return fig,ax
    
def plot_ratemap_fr_one_neuron_all_fields(fr_map_trial_df,all_fields,X_raw,X_pwc,all_sw_d,index_within_to_trial_index_df,
                                       task_ind,tt,uid,flipy=True,
                                       fig=None,axs=None):
    '''
    combine the heatmap of ratemap and lineplot of within field rate
    '''
    
    if axs is None:
        fig,axs=plt.subplots(1,2,figsize=(6*2,4),gridspec_kw={'width_ratios': [1,1.5]},sharey=True)
    ax=axs[0]
    fig,ax=plot_x_raw_pwc_one_neuron_all_fields(X_raw,X_pwc,all_fields,index_within_to_trial_index_df,
                                       task_ind,tt,uid,flipy=flipy,
                                       fig=fig,ax=ax)
    ax=axs[1]
    fig,ax=plot_ratemap_one_neuron_all_fields(fr_map_trial_df,all_fields,all_sw_d,index_within_to_trial_index_df,
                                       task_ind,tt,uid,flipy=flipy,
                                       fig=fig,ax=ax
                                      )
    axs[1].set(ylabel="")
    axs[0].yaxis.set_major_locator(MaxNLocator(10,integer=True))
    axs[0].set_yticklabels(np.array(axs[0].get_yticks().tolist(),dtype=int))
    return fig,axs


# helper function#
def get_sw_ind_within_and_tt_ratemap(all_sw_d,task,tt,uid,field_id,trial_index_to_index_within_df):
    '''
    for "both" fields that switch, find the exact trial type of the switch and the index within
    '''
    all_sw_d = all_sw_d.loc[task] # forgot this before
    ma = (all_sw_d['trialtype']==tt) & (all_sw_d['uid']==uid) & (all_sw_d['field_index']==field_id) 
    all_sw_d_sub = all_sw_d.loc[ma]
    if all_sw_d_sub.shape[0]==0:
        exist_switch = False
    else:
        exist_switch = True
    if exist_switch:
        if tt=='both':
            ma=[x in all_sw_d_sub['switch_trial'].values for x in trial_index_to_index_within_df.loc[task].index.get_level_values(1)] # add .values, otherwise always give False
            # tt_ratemap = trial_index_to_index_within_df.loc[task].loc[ma].index[0][0]
            
            tt_ratemap = trial_index_to_index_within_df.loc[task].loc[ma].index[0][0]
            # sw_index_within = trial_index_to_index_within_df.loc[task].loc[ma].values[0]
            # pdb.set_trace()
            sw_index_within = trial_index_to_index_within_df.loc[task].loc[ma].loc[tt_ratemap].values[0] # for both, only keep the sw indices within the selected trialtype
        else:
            tt_ratemap=tt
            # pdb.set_trace()
            sw_index_within = all_sw_d_sub['switch_trial']
    else:
        sw_index_within = None
        if tt=='both': # if no switch, default to trialtype 0
            tt_ratemap = 0
        else:
            tt_ratemap = tt
    return sw_index_within,tt_ratemap

def plot_x_raw_and_pwc(X_raw_one,X_pwc_one,fig=None,ax=None,do_invert_y=True,
                        pwc_c='b',do_legend=True,plot_kws=None,
                        ):
    if plot_kws is None:
        plot_kws = {}
    plot_kws_ = {'ms':5,'pwc_linewidth':3}
    plot_kws_.update(plot_kws)
    if ax is None:
        fig,ax=plt.subplots()
    if do_invert_y:
        ax.invert_yaxis()
    toplot=dict(raw=X_raw_one,
    fitted = X_pwc_one)
    for k,val in toplot.items():
        trials = np.arange(len(val))
        if k=='raw':
            ax.plot(val,trials,label=k,marker='o',color='grey',ms=plot_kws_['ms'])
        else:
            ax.plot(val,trials,label=k,color=pwc_c,linewidth=plot_kws_['pwc_linewidth'])
    ax.yaxis.set_major_locator(MaxNLocator(10,integer=True))
    if do_legend:
        ax.legend()
    ax.spines[['top','right']].set_visible(False)
    ax.set_ylabel('Trial')
    # ax.set_xlabel('Peak firing rate (Hz)')
    ax.set_xlabel('Peak FR (Hz)')
    return fig,ax

#=====only one field=====#
from matplotlib import gridspec
def plot_ratemap_fr_one_field_avgfm(all_fields_row_one,fr_map_trial_df,X_raw,X_pwc,all_sw_d,
                            trial_index_to_index_within_df,
                            save_fig_fn = None,
                            close_fig = False,
                            vmax_clip_quantile=0.99,
                            vmax_relative_to_field=False,
                            fig=None,axs=None,cbar_ax=None,cmap='viridis',
                            figsize=(6,4)
                            ):
    '''
    three panels; trial-ratemap, FR across trial, mean FR
    '''
    
    field_bound,sw_index_within,tt_ratemap,fr_map_trial_one,X_raw_one,X_pwc_one = prep_ratemap(all_fields_row_one,fr_map_trial_df,X_raw,X_pwc,all_sw_d,
                            trial_index_to_index_within_df)
    fm_avg = fr_map_trial_one.mean(axis=0)
    
    if axs is None:
        fig=plt.figure(figsize=figsize,constrained_layout=True)
        gs = gridspec.GridSpec(2, 3, width_ratios=[1,1,0.05],height_ratios=[0.5,1],figure=fig)
        ax = fig.add_subplot(gs[0,1])
        axfr = fig.add_subplot(gs[1,0])
        ax_heat = fig.add_subplot(gs[1,1],sharex=ax,sharey=axfr)
        axs = np.array([ax,axfr,ax_heat])
        axs_ = axs[1:]
        cbar_ax = fig.add_subplot(gs[1,2])
    ax.plot(fm_avg)
    ax.set_ylabel('Avg. FR (Hz)')
    ax.xaxis.set_visible(False)
    sns.despine(ax=ax)
    
    

    fig,axs_=plot_ratemap_fr_one_field(all_fields_row_one,fr_map_trial_df,X_raw,X_pwc,all_sw_d,
                            trial_index_to_index_within_df,
                            save_fig_fn = None,
                            close_fig = False,
                            vmax_clip_quantile=vmax_clip_quantile,
                            vmax_relative_to_field=vmax_relative_to_field,fig=fig,axs=axs_,cbar_ax=cbar_ax,cmap=cmap
                                )
    return fig,axs


    
def prep_ratemap(all_fields_row_one,fr_map_trial_df,X_raw,X_pwc,all_sw_d,
                            trial_index_to_index_within_df):
    '''
    parse the data to show one trialtype; 
    '''
    task,tt,uid,field_id=all_fields_row_one.name
    field_bound = all_fields_row_one[['start','end']]

    sw_index_within, tt_ratemap=get_sw_ind_within_and_tt_ratemap(all_sw_d,task,tt,uid,field_id,trial_index_to_index_within_df)

    fr_map_trial_one = fr_map_trial_df.loc[task,tt_ratemap,uid].dropna(axis=1,how='all').T
    
    index_within_left = trial_index_to_index_within_df.loc[task,tt_ratemap].values
    fr_map_trial_one = fr_map_trial_one.loc[index_within_left]
    
    columns_one_tt = X_pwc.loc[task,tt_ratemap].dropna(axis=1,how='all').columns
    X_raw_one = X_raw.loc[task,tt,uid,field_id][columns_one_tt]
    X_pwc_one = X_pwc.loc[task,tt,uid,field_id][columns_one_tt]

    return field_bound,sw_index_within,tt_ratemap,fr_map_trial_one,X_raw_one,X_pwc_one
    


def plot_ratemap_fr_one_field(all_fields_row_one,fr_map_trial_df,X_raw,X_pwc,all_sw_d,
                            trial_index_to_index_within_df,
                            save_fig_fn = None,
                            close_fig = False,
                            vmax_clip_quantile=0.99,
                            vmax_relative_to_field=False,fig=None,axs=None,cbar_ax=None,cmap='viridis',figsize=(6*2,4),
                            do_legend=True,
                            plot_kws=None
                                ):
    

    task,tt,uid,field_id=all_fields_row_one.name
    field_bound,sw_index_within,tt_ratemap,fr_map_trial_one,X_raw_one,X_pwc_one = prep_ratemap(all_fields_row_one,fr_map_trial_df,X_raw,X_pwc,all_sw_d,
                            trial_index_to_index_within_df)

    if vmax_relative_to_field:
        vmax=np.quantile(X_raw_one,vmax_clip_quantile)
    else:
        vmax = np.quantile(fr_map_trial_one.values,vmax_clip_quantile)
    if axs is None:
        fig,axs=plt.subplots(1,2,figsize=figsize,gridspec_kw={'width_ratios': [1,1.5]})
    ax = axs[0]
    
    fig,ax=plot_x_raw_and_pwc(X_raw_one,X_pwc_one,fig=fig,ax=ax,do_legend=do_legend,plot_kws=plot_kws)
    ax=axs[1]
    
    ph.ratemap_one_raw(fr_map_trial_one,trial=sw_index_within,field_bound=field_bound,fig=fig,ax=ax,line_kws={},title=None,heatmap_kws={'vmax':vmax,'cbar_ax':cbar_ax,'cmap':cmap})
    

    # ax.set_yticks([])
    ax.yaxis.set_visible(False)
    # ax.invert_yaxis()
    ax.set_xlabel('Position (bin)')

    if save_fig_fn is not None:
        fig.savefig(save_fig_fn,bbox_inches='tight')
        print(f'{save_fig_fn} saved!')
    if close_fig:
        plt.close(fig)
    else:
        return fig, axs

def plot_ratemap_fr_all_fields(fr_map_trial_df,all_fields_recombined,all_sw_d,trial_index_to_index_within_df,X_raw,X_pwc,
                                save_dir='',save_fn_func=save_fn_func):
    
    for i,row in all_fields_recombined.iterrows():
        task,tt,uid,field_id = i
        save_fn = save_fn_func(task,tt,uid,field_id)
        save_fn_full = os.path.join(save_dir,save_fn)

        plot_ratemap_fr_one_field(row,fr_map_trial_df,X_raw,X_pwc,all_sw_d,
                            trial_index_to_index_within_df,
                            save_fig_fn = save_fn_full,
                            close_fig = True,
                                )

        

    

def load_preprocess_data(data_dir_full):
    prep_res = dpp.load_spk_beh_df(data_dir_full,force_reload=False,extra_load=dict(sessionPulses='*SessionPulses.Events.mat',filtered='*thetaFiltered.lfp.mat'))
    spk_beh_df=prep_res['spk_beh_df']
    trial_index_to_index_within_df = dpp.trial_index_to_index_within_trialtype(spk_beh_df)
    
    fr_map_all = fmos.main(data_dir_full,force_reload=False,load_only=True)
    fr_map_trial_df = fr_map_all['fr_map_trial_df']

    pf_res_recombine = pfrc.main(data_dir_full,force_reload=False,load_only=True)
    all_fields_recombined=pf_res_recombine['all_fields_recombined']

    sw_res = sdos.main(data_dir_full,force_reload=False,load_only=True)
    X_pwc = sw_res['X_pwc']
    X_raw = sw_res['X_raw']

    sw_info_res=gasam.main(data_dir_full,force_reload=False,load_only=True)
    all_sw_with_metrics_d = sw_info_res['all_sw_with_metrics_d']
    
    data = {'trial_index_to_index_within_df':trial_index_to_index_within_df,'fr_map_trial_df':fr_map_trial_df,
            'all_fields_recombined':all_fields_recombined,'X_pwc':X_pwc, 'X_raw':X_raw, 'all_sw_with_metrics_d':all_sw_with_metrics_d
    }
    return data

def analyze_data(data,*args,**kwargs):

    trial_index_to_index_within_df = data['trial_index_to_index_within_df']
    fr_map_trial_df = data['fr_map_trial_df']
    all_fields_recombined =data['all_fields_recombined']
    X_pwc = data['X_pwc']
    X_raw = data['X_raw']
    all_sw_d = data['all_sw_with_metrics_d']

    save_dir = kwargs['save_dir']

    plot_ratemap_fr_all_fields(fr_map_trial_df,all_fields_recombined,all_sw_d,trial_index_to_index_within_df,X_raw,X_pwc,
                                save_dir=save_dir,save_fn_func=save_fn_func)

    ma_d = {'on':(all_sw_d['post_ntrial_ge_50_perc_frac_segment'] >= 0.5) & (all_sw_d['switch'] ==1) & (all_sw_d['pre_ntrial_le_30_perc_frac_segment'] >= 0.5) ,
                'off':(all_sw_d['pre_ntrial_ge_50_perc_frac_segment'] >= 0.5) & (all_sw_d['switch'] ==-1) & (all_sw_d['post_ntrial_le_30_perc_frac_segment'] >= 0.5),
                }
        
    for k,ma in ma_d.items():
        save_dir_good = misc.get_or_create_subdir(save_dir,f'good_{k}',doclear=True) # delete old ones first
        ma = ma & (all_sw_d['field_pos']>=10) & (all_sw_d['field_pos']<=92) # particular to roman's maze. 
        ma =ma & (all_sw_d['post_ntrials_in_segment']>=4)
        all_sw_d_sub = all_sw_d.loc[ma]
        
        if all_sw_d_sub.shape[0]>0:
            inds=all_sw_d_sub.reset_index(level=0)[['level_0','trialtype','uid','field_index']] # level_0 is task
            inds=pd.MultiIndex.from_frame(inds)
            all_fields_recombined_sub = all_fields_recombined.loc[inds]
            plot_ratemap_fr_all_fields(fr_map_trial_df,all_fields_recombined_sub,all_sw_d,trial_index_to_index_within_df,X_raw,X_pwc, # use all_sw_d instead of all_sw_d_sub here so that all switches will show up on fr map
                                save_dir=save_dir_good,save_fn_func=save_fn_func)
            

                
    
# def save_results(results, session_path, output_folder):
#     # Save your results to a file
#     pass

def main(session_path,test_mode=False,
        analysis_args=[],
        analysis_kwargs={},
        dosave=True, save_dir=SAVE_DIR,save_fn=save_fn_func, force_reload=False,load_only=False,
    ):

    try:
        # create subdir
        save_dir = misc.get_or_create_subdir(session_path,'py_figures',save_dir)
        # save_fn, res = misc.get_res(save_dir,save_fn,force_reload)
        # if (res is not None) or load_only: # load only would skip the computation that follows
            # return res
        data = load_preprocess_data(session_path)
        if test_mode:
            # UPDATE SOME PARAMS!!!
            pass
        
        res = analyze_data(data,*analysis_args,save_dir=save_dir,**analysis_kwargs)
        # misc.save_res(save_fn,res,dosave=dosave)
        return res
        
    except Exception as e:
        tb_str = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
        tb_str.insert(0, f"Error in session: {session_path}\n")
        sys.stderr.writelines(tb_str)

if __name__ == "__main__":
    sess_ind = int(sys.argv[1])
    test_mode = bool(sys.argv[2])
    session_path = subdb['data_dir_full'][sess_ind]
    
    main(session_path, test_mode=test_mode)
