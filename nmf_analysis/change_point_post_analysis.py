import ruptures as rpt
import numpy as np
import pandas as pd
import scipy
import os,sys,copy,itertools,pdb,importlib
import change_point_plot as cpp
importlib.reload(cpp)
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
sys.path.append('/mnt/home/szheng/projects/util_code')
import plot_helper as ph
import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'

def reshape_switch_detection_result_all_sess(res_to_save_data_detection_l,sess_selected=None):
    '''
    res_to_save_data_detection_l: {(animal, session):res_to_save_data_detection}
        res_to_save_data_detection: {detection: res_to_save_data}, see switch_analysis_one_session.switch_analysis_one_session 
            res_to_save_data = dict(
            X = X_to_be_analyzed,
            cdf=cdf_alltrialtype,
            sig=sig_alltrialtype,
            sr=sr_alltrialtype,
            changes_df=changes_df_alltrialtype,
            fig=fig_alltrialtype
        )
    ===
    switch_detection_res_allsess
    {detection: {result_type: {result_concat_across_sessions}}
        result_concat_across_sessions: df: (animal, sess, xxx) x (onoff x trial)
    '''
    switch_detection_res_allsess = {}
    for detection in ['avg','trial_filter']:
        switch_detection_res_allsess[detection]={}
        for key in ['sig','cdf','changes_df']:
            switch_detection_res_allsess[detection][key] = pd.concat({k:val[detection][key] if val is not None else None for k,val in res_to_save_data_detection_l.items()},axis=0)
            if sess_selected is None:
                switch_detection_res_allsess[detection][key] = switch_detection_res_allsess[detection][key]
            else:
                switch_detection_res_allsess[detection][key] = switch_detection_res_allsess[detection][key].loc[(slice(None),sess_selected.values),:]
        switch_detection_res_allsess[detection]['sig_pos'] = switch_detection_res_allsess[detection]['sig'] * (switch_detection_res_allsess[detection]['cdf'] > 0.95)
    return switch_detection_res_allsess

def get_coswitching_examples_nonexamples_inds(X_raw,X_pwc, all_fields, changes_df, trial_index,onoff=1,do_sort=False):
    field_inds = changes_df.index[(changes_df.loc[:,trial_index] == onoff)]
    field_coms = all_fields.loc[field_inds,'com']
    field_coms = field_coms.sort_values()
    

    non_field_inds = changes_df.index[(changes_df.loc[:,trial_index]!=onoff)&(changes_df.loc[:,trial_index].notna()) ]
    non_field_coms = all_fields.loc[non_field_inds,'com']

    nonfield_coms_closest_ind = np.argmin(np.abs(np.subtract.outer(field_coms.values,non_field_coms.values)),axis=1)
    nonfield_coms_closest_ind = pd.Series(nonfield_coms_closest_ind).unique()
    nonfield_coms_selected = non_field_coms.iloc[nonfield_coms_closest_ind]
    if do_sort:
        field_coms = sort_by_active_trials_after_switch(X_raw,field_coms,trial_index,onoff=onoff)
        nonfield_coms_selected = sort_by_active_trials_after_switch(X_raw,nonfield_coms_selected,trial_index,onoff=onoff)
    all_field_selected_coms = pd.concat([field_coms,nonfield_coms_selected])

    return all_field_selected_coms, field_coms, nonfield_coms_selected

def sort_by_active_trials_after_switch(X,field_coms,trial_index,onoff=1,onthresh=0.05,offthresh=0.2):
    Xsub = X.loc[field_coms.index] 
    if onoff==1:
        ntrials_onoff=(Xsub.loc[:,trial_index:] >= (Xsub.max(axis=1).values[:,None] * onthresh)).sum(axis=1)
    elif onoff==-1:
        ntrials_onoff=(Xsub.loc[:,trial_index:] <= (Xsub.max(axis=1).values[:,None] * offthresh)).sum(axis=1)
    field_coms_sorted=field_coms.loc[ntrials_onoff.sort_values(ascending=False).index]
    
    return field_coms_sorted


# no com:
# plot coswitching neurons in one trial, with some non examples 
from matplotlib.ticker import MaxNLocator,MultipleLocator
def show_coswitching_examples_nonexamples(X_raw,X_pwc, all_fields, changes_df, trial_index,onoff=1,fig=None,axs=None,all_field_selected_coms=None,field_coms=None,nonfield_coms_selected=None,do_sort=False,normalize=True,lw = 2,ticklabelfontsize = 10):
    
    

    if (all_field_selected_coms is None) or (field_coms is None) or (nonfield_coms_selected is None):
        all_field_selected_coms, field_coms, nonfield_coms_selected = get_coswitching_examples_nonexamples_inds(X_raw,X_pwc, all_fields, changes_df, trial_index,onoff=onoff,do_sort=do_sort)
    seperating_line = len(field_coms)-0.5
        

    
    data_pwc = X_pwc.loc[all_field_selected_coms.index].dropna(axis=1)
    data_raw = X_raw.loc[all_field_selected_coms.index].dropna(axis=1).T.reset_index(drop=True).T
    index_within = np.nonzero(data_pwc.columns==trial_index)[0][0]
    data_pwc = data_pwc.T.reset_index(drop=True).T

    if normalize:
        data_pwc = data_pwc / data_pwc.max(axis=1).values[:,None]
        data_raw = data_raw / data_raw.max(axis=1).values[:,None]


    # if axs is None:
    #     fig,axs=plt.subplots(1,2,sharey=False,figsize=(10,6))
    if axs is None:
        # fig,axs=plt.subplots(1,1,sharey=False,figsize=(5,6))
        fig,axs=plt.subplots(1,1,sharey=False,figsize=(1.5,2))

    # sns.heatmap(data_pwc,ax=axs[0],cmap='Greys')
    
    ## axs[0].imshow(data_pwc,cmap='Greys',aspect='auto')
    # axs[0].hlines(seperating_line,*axs[0].get_xlim(),linewidth=lw,color='C1')
    # axs[0].vlines(index_within-0.5,*axs[0].get_ylim(),linewidth=lw,linestyle='-',color='C2')
    # yticks = [int(len(field_coms)/2),len(field_coms)+int(len(nonfield_coms_selected)/2)]
    # ncoswitch = len(field_coms)
    # percentile =ncoswitch / X_raw[trial_index].dropna().shape[0]
    # axs[0].set(ylabel="Place Field",yticks=yticks,xlabel="",title=f"Fitted FR\n(N co-switch = {ncoswitch}; {percentile:.2%})")
    # axs[0].set_yticklabels(['co-switch in\none trial','nonexamples'],fontsize=ticklabelfontsize)
    axs.imshow(data_raw,cmap='Greys',aspect='auto',interpolation='none')
    axs.hlines(seperating_line,*axs.get_xlim(),linewidth=lw,color='C1')
    axs.vlines(index_within-0.5,*axs.get_ylim(),linewidth=lw,linestyle='-',color='C2')
    yticks = [int(len(field_coms)/2),len(field_coms)+int(len(nonfield_coms_selected)/2)]
    ncoswitch = len(field_coms)
    percentile =ncoswitch / X_raw[trial_index].dropna().shape[0]
    axs.set_ylabel("Place Field",fontsize=ticklabelfontsize)
    axs.set(yticks=yticks)
    axs.set_title(f"Peak FR\n(N co-switch = {ncoswitch}; {percentile:.2%})",fontsize=ticklabelfontsize)
    axs.set_yticklabels(['co-switch in\none trial','nonexamples'],fontsize=ticklabelfontsize)
    axs.set_xlabel('Trial',fontsize=ticklabelfontsize)
    axs.xaxis.set_major_locator(MaxNLocator(integer=True))
    axs.set(frame_on=False)
    # # sns.heatmap(data_raw,ax=axs[1],cmap='Greys')
    # axs[1].imshow(data_raw,cmap='Greys',aspect='auto')
    # axs[1].hlines(seperating_line,*axs[1].get_xlim(),linewidth=lw,color='C1')
    # axs[1].vlines(index_within-0.5,*axs[0].get_ylim(),linewidth=lw,linestyle='-',color='C2')
    # axs[1].set(ylabel="",yticks=[],title=f"Raw FR\n(N co-switch = {ncoswitch}; {percentile:.2%})",xlabel='')
    # # fig.supxlabel("Trial (within trial type)",fontsize=20)
    # fig.supxlabel("Trial",fontsize=20)
    # plt.tight_layout(w_pad=6)
    cbar=fig.colorbar(axs.images[0])
    cbar.set_ticks([0,1])
    # axs[0].set(frame_on=False)
    # axs[1].set(frame_on=False)
    axs.xaxis.set_major_locator(MaxNLocator(nbins=4,integer=True))
    
    return fig,axs


# plot coswitching neurons in one trial, with some non examples 
def show_coswitching_examples_nonexamples_withcom(X_raw,X_pwc, all_fields, changes_df, trial_index,onoff=1,fig=None,axs=None,all_field_selected_coms=None,field_coms=None,nonfield_coms_selected=None,do_sort=False):
    
    lw = 4

    if (all_field_selected_coms is None) or (field_coms is None) or (nonfield_coms_selected is None):
        all_field_selected_coms, field_coms, nonfield_coms_selected = get_coswitching_examples_nonexamples_inds(X_raw,X_pwc, all_fields, changes_df, trial_index,onoff=onoff,do_sort=do_sort)
    seperating_line = len(field_coms)
    

    
    data_pwc = X_pwc.loc[all_field_selected_coms.index]
    data_raw = X_raw.loc[all_field_selected_coms.index]
    if axs is None:
        fig,axs=plt.subplots(1,3,sharey=False)
    sns.heatmap(data_pwc,ax=axs[0],cmap='Greys')
    axs[0].hlines(seperating_line,*axs[0].get_xlim(),linewidth=lw,color='C1')
    axs[0].vlines(trial_index,*axs[0].get_ylim(),linewidth=lw,linestyle=':',color='C0')
    yticks = [int(len(field_coms)/2),len(field_coms)+int(len(nonfield_coms_selected)/2)]
    axs[0].set(ylabel="Place Field",yticks=yticks,yticklabels=['co-switch in\none trial','nonexamples'],xlabel="Trial",title=f"Fitted FR\n(N co-switch = {len(field_coms)})")
    sns.heatmap(data_raw,ax=axs[1],cmap='Greys')
    axs[1].hlines(seperating_line,*axs[1].get_xlim(),linewidth=lw,color='C1')
    axs[1].vlines(trial_index,*axs[0].get_ylim(),linewidth=lw,linestyle=':',color='C0')
    axs[1].set(ylabel="",yticks=[],title=f"Raw FR\n(N co-switch = {len(field_coms)})",xlabel="Trial")
    axs[2].invert_yaxis()
    axs[2].plot(all_field_selected_coms,np.arange(len(all_field_selected_coms)),marker='o',c='k')
    axs[2].set(ylabel="",yticks=[],xlabel="COM (bin)",title='Location',xticks=[0,25,50,75,100])
    axs[2].hlines(seperating_line,*axs[2].get_xlim(),linewidth=lw,color='C1')
    plt.tight_layout()
    
    return fig,axs
    
def get_n_sig_trials_and_meta_data(switch_detection_res_allsess,db_sorted,detection='avg',split_trialtype=True):
    '''
    return : df, (n_ani x n_sess) x (on x switch_detection_stuff; off x ,,, n_pyr; n_trial; etc.)

    
    '''
    # df=switch_detection_res_allsess[detection]['sig_pos'].groupby(level=(0,1,4,5,6)).any().groupby(axis=1,level=0).sum().astype(int) # any is counting once every change occuring on the same trial index across the trial types
    # 

    if split_trialtype:
        df=switch_detection_res_allsess[detection]['sig_pos'].fillna(0).groupby(axis=1,level=0).sum().astype(int) # seperate into trialtype
    else:
        df=switch_detection_res_allsess[detection]['sig_pos'].fillna(0).groupby(level=(0,1,2,4,5,6)).sum().groupby(axis=1,level=0).sum().astype(int) # sum: count in all significant trials across the trial types
    df=df.unstack(level=(-1,-2,-3))
    # db_sorted_reind_filtered=db_sorted.reset_index(drop=True).set_index(['animal_name.1','sess_name']).loc[df.index]
    db_sorted_reind=db_sorted.reset_index(drop=True).set_index(['animal_name.1','sess_name'])
    
    df['n_pyr'] = df.index.map(lambda x:db_sorted_reind.loc[x[:2],'n_pyr_putative'])
    if split_trialtype:    
        ntrials_by_trialtype = switch_detection_res_allsess[detection]['sig_pos']['on'].groupby(level=(0,1,2,3)).apply(lambda x:x.droplevel((-1,-2,-3)).dropna(axis=1).shape[1])
        df['n_trial'] = ntrials_by_trialtype
    else:
        ntrials_by_task = switch_detection_res_allsess[detection]['sig_pos']['on'].groupby(level=(0,1,2,3)).apply(lambda x:x.droplevel((-1,-2,-3)).dropna(axis=1).shape[1]).groupby(level=(0,1,2)).sum()
        df['n_trial'] = ntrials_by_task
    
    return df

def plot_n_sig_trials_and_meta_data(switch_detection_res_allsess,db_sorted,meta_key,onoff_str='on',detection='avg',n_sig_trial_meta_df=None,sw_ind=(0.4,'switch_magnitude',0.3),
                                fig=None,ax=None,split_trialtype=True,
                                **kwargs
):
    if n_sig_trial_meta_df is None:
        n_sig_trial_meta_df = get_n_sig_trials_and_meta_data(switch_detection_res_allsess,db_sorted,detection=detection,split_trialtype=split_trialtype)
    if ax is None:
        fig,ax=plt.subplots()

    ax.scatter(n_sig_trial_meta_df[meta_key],n_sig_trial_meta_df.loc[:,(onoff_str,*sw_ind)])
    ax.set(ylabel='N Significant Trials\n Per Session')
    return fig,ax,n_sig_trial_meta_df


def get_per_ani_sig_trial_hist(switch_detection_res_allsess,detection='avg',sw_ind=(0.3,'switch_magnitude',0.4),onoff_str='on',split_trialtype=True):
    if split_trialtype:
        ani_tt_slice = (slice(None),slice(None),slice(None),slice(None))
        df=switch_detection_res_allsess[detection]['sig_pos'].loc[(*ani_tt_slice,*sw_ind),onoff_str].droplevel((-1,-2,-3))
        per_ani_sig_trial_hist_splittrialtype=(df==True).sum(axis=1).groupby(level=(0,1,2)).sum().groupby(level=(0,2)).value_counts()
        return per_ani_sig_trial_hist_splittrialtype
    else:
        print('not implemented yet')
        return 

def plot_per_ani_sig_trial_hist_oneani(per_ani_sig_trial_hist_splittrialtype,ani,task_index=0,split_trialtype=True,ax=None):
    count_one=per_ani_sig_trial_hist_splittrialtype.loc[ani,task_index].sort_index()
    ax=count_one.plot.bar(ax=ax)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    nsess=per_ani_sig_trial_hist_splittrialtype.loc[ani,task_index].sum()
    ax.set(ylabel='Count', xlabel='Num Significant Trials',title=f'Significant Trials per Session\n(N session={nsess})')
    return ax


#=========== rate map by trial==========#

### barebone version of ratemap with field bounds
def ratemap_one_raw(data,trial=None,field_bound=None,fig=None,ax=None,line_kws={},title=None):
    line_kws_ = {'linewidth':3,'linestyle':':'}
    line_kws_.update(line_kws)
    if ax is None:
        fig,ax=plt.subplots()
    ax=sns.heatmap(data,ax=ax)
    if trial is not None:
        ax.hlines(trial,*ax.get_xlim(),color='C0',**line_kws_)
    if field_bound is not None:
        field_st,field_end = field_bound        
        ax.vlines(field_st,*ax.get_ylim(),color='C1',**line_kws_)
        ax.vlines(field_end,*ax.get_ylim(),color='C1',**line_kws_)
    if title is not None:
        ax.set_title(title)
    return fig,ax




def ratemap_one(trial,all_fields,fr_map_trial_df_d,task_ind,tt_ind,uid,field_id,fig=None,ax=None,do_title=True,line_kws={},do_shuffle=False):
    line_kws_ = {'linewidth':3,'linestyle':':'}
    line_kws_.update(line_kws)
    if ax is None:
        fig,ax=plt.subplots()
    field_st = all_fields.loc[uid,field_id]['start']
    field_end = all_fields.loc[uid,field_id]['end']
    frmap = fr_map_trial_df_d.loc[task_ind,tt_ind,uid].T.dropna(axis=0)
    if do_shuffle: # circ shuffle trial
        ntrials = frmap.shape[0]
        shift_int = np.random.randint(1,ntrials-1)
        new_ind = np.roll(frmap.index,shift_int)
        frmap = frmap.iloc[new_ind]
    sns.heatmap(frmap,cmap='Greys',ax=ax)

    # mark trial and field bounds
    ax.hlines(trial,*ax.get_xlim(),color='C0',**line_kws_)
    ax.vlines(field_st,*ax.get_ylim(),color='C1',**line_kws_)
    ax.vlines(field_end,*ax.get_ylim(),color='C1',**line_kws_)
    if do_title:
        ax.set_title(f'{uid},{field_id}')
    return fig,ax

def ratemap_multiple(trial,all_field_selected_coms,all_fields,fr_map_trial_df_d,task_ind,tt_ind,do_title=True,do_suptitle=True,line_kws={},fig=None,axs=None,do_shuffle=False):
    nplots = all_field_selected_coms.shape[0]
    if nplots > 0:
        if axs is None:
            fig,axs = ph.subplots_wrapper(nplots)
        for ii in range(nplots):
            ax = axs.ravel()[ii]
            uid,field_id = all_field_selected_coms.index[ii]
            com = all_field_selected_coms.iloc[ii]
            fig,ax=ratemap_one(trial,all_fields,fr_map_trial_df_d,task_ind,tt_ind,uid,field_id,fig=fig,ax=ax,do_title=do_title,line_kws=line_kws,do_shuffle=do_shuffle)
        if do_suptitle:
            fig.suptitle(f'task{task_ind}, trialtype{tt_ind}\n trial{trial}')
        plt.tight_layout()
    return fig,axs

def coswitch_ratemap_onetrial(fr_map_trial_df_d,X_pwc, all_fields, changes_df, trial=1,onoff=1,do_title=True,seperate_ex_nonex=False,
                              line_kws = {'linewidth':3,'linestyle':':'},do_suptitle=True
                             ):
    all_field_selected_coms, field_coms, nonfield_coms_selected = cppa.get_coswitching_examples_nonexamples_inds(None,X_pwc, all_fields, changes_df, trial,onoff=onoff,do_sort=False)
    if not seperate_ex_nonex:
        fig,axs=ratemap_multiple(trial,all_field_selected_coms,all_fields,fr_map_trial_df_d,task_ind,tt_ind,line_kws=line_kws,do_suptitle=do_suptitle)
        return fig,axs
    else:
        fig_sel,axs_sel=ratemap_multiple(trial,field_coms,all_fields,fr_map_trial_df_d,task_ind,tt_ind,line_kws=line_kws,do_suptitle=do_suptitle)
        fig_non,axs_non=ratemap_multiple(trial,nonfield_coms_selected,all_fields,fr_map_trial_df_d,task_ind,tt_ind,line_kws=line_kws,do_suptitle=do_suptitle)
        return fig_sel,axs_sel, fig_non,axs_non

