'''
analyzing stability after switching
'''
import numpy as np
import pandas as pd
import os,sys,pdb,copy
import scipy

import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
import seaborn as sns


def get_var_one_row_selected_cols(row,st,ed,var_func='var',return_na_if_len_mismatch=True):
    
    row_ = row.dropna().iloc[st:ed]

    if len(row_)<(ed-st):
        if return_na_if_len_mismatch:
            return np.nan

    if var_func=='var':
        return row_.var()
    elif var_func=='std':
        return row_.std()
    elif var_func=='fano':
        return row_.var() / row_.mean()
    elif var_func=='cv':
        return row_.std() / row_.mean()



def get_sw_no_sw_pf_par_var_control_ntrial(changes_df_all,pf_params_recombined_all,par_key='peak',var_func='var',
                                    random_start=None,nosw_start_frac=0.5,min_size=2,nosw_start_pre_min=False,nosw_start_pre_peak=False,ntrials_after_sw=5,ntrials_pre_sw=0):
    '''
    nosw_start_frac is prioritized; fraction of trial where noswitch fields are counted to start

    '''
    ma_d = {}
    for on_count in [0,1]:
        ma = ((changes_df_all==1).sum(axis=1)==on_count) 
        ma_d[on_count] = ma

    # for no sw fields, subselect the stable firing ones
    fr_thresh = 1
    frac_trial_thresh = 0.9
    firing_most_trials=pf_params_recombined_all.loc[(slice(None),slice(None),'fr_peak')].apply(lambda x:(x.dropna()>fr_thresh).mean()>frac_trial_thresh,axis=1) 
    ma_d[0] = np.logical_and(ma_d[0] ,firing_most_trials)

    masked=changes_df_all.loc[ma_d[1]].fillna(-0.1)==1
    row_ind,col_ind = np.nonzero(masked.values)

    pf_par_on =pf_params_recombined_all.loc[(slice(None),slice(None),par_key)].loc[ma_d[1]]
    pf_par_nosw =pf_params_recombined_all.loc[(slice(None),slice(None),par_key)].loc[ma_d[0]]
    fr_peak_nosw= pf_params_recombined_all.loc[(slice(None),slice(None),'fr_peak')].loc[ma_d[0]]

    par_var_nosw_d = {}
    # par_nosw_d = {}
    par_nosw_d = []
    # no_sw 
    for ind,row in pf_par_nosw.iterrows():
        row_dropna=row.dropna()
        fr_row_dropna=fr_peak_nosw.loc[ind].dropna()
        if nosw_start_frac is not None: # prioritize nosw_start_frac; at what fraction of trials to start looking at the no switch field parameters
            min_size = int(len(row_dropna) * nosw_start_frac)
        else:
            min_size = np.minimum(min_size,len(row_dropna)-1)

        # ind_min_size=row_dropna.index[min_size]
        # iloc_ind_min_size=np.nonzero(row.index==ind_min_size)[0][0]
        iloc_ind_min_size = min_size # for get_var_one_row_selected_cols, dropna and then use iloc; min_size already in terms of iloc
        if nosw_start_pre_peak:
            # ind_min_size = fr_row_dropna.idxmax()
            # iloc_ind_min_size=np.nonzero(row.index==ind_min_size)[0][0]
            iloc_ind_min_size = np.argmax(fr_row_dropna.values)
        elif nosw_start_pre_min:
            # ind_min_size = fr_row_dropna.idxmin()
            # iloc_ind_min_size=np.nonzero(row.index==ind_min_size)[0][0]
            iloc_ind_min_size = np.argmin(fr_row_dropna.values)
        elif random_start is not None:
            maxn = len(row_dropna)-(ntrials_after_sw+ntrials_pre_sw) # upper bound for randomly selecting index, some distance below the array end
            maxn_bound = random_start + (ntrials_after_sw+ntrials_pre_sw) # some distance above the array start
            maxn = np.maximum(maxn,maxn_bound) # pick the max among the two, because from the end can get negative
            maxn = np.minimum(maxn,len(row_dropna)) # pick the min because the distance above the start can be above the array length
            random_start = np.minimum(random_start,maxn-1) # make start smaller than maxn
            iloc_ind_min_size = np.random.randint(random_start,maxn)
            
        par_var_nosw_d[ind]=get_var_one_row_selected_cols(row,iloc_ind_min_size,iloc_ind_min_size+ntrials_after_sw+ntrials_pre_sw,var_func=var_func,return_na_if_len_mismatch=True)

        par_nosw_one = row_dropna.iloc[iloc_ind_min_size:iloc_ind_min_size+ntrials_after_sw+ntrials_pre_sw].reset_index(drop=True)
        # par_nosw_d[ind] = par_nosw_one
        par_nosw_d.append(par_nosw_one)

    par_var_nosw_d = pd.Series(par_var_nosw_d)
    par_nosw_d = pd.concat(par_nosw_d,axis=1).T

    # prep sw variables
    # par_after_on_d = {}
    par_after_on_d=[]
    par_var_after_on_d = {}
    
    for r,c in zip(row_ind,col_ind):
        # par_after_on = pf_par_on.iloc[r].dropna().iloc[c:c+ntrials_after_sw]
        
        ind_st = pf_par_on.iloc[r].index[c]
        try: # sometimes all nan, then pass
            
            iloc_ind_st=np.nonzero(pf_par_on.iloc[r].dropna().index==ind_st)[0][0]
            iloc_ind_st = np.maximum(iloc_ind_st-ntrials_pre_sw,2)
            ind=pf_par_on.index[r]
            # ntrials_after_sw+ntrials_pre_sw: new section length, since iloc_ind_st would be shifted backward by ntrials_pre_sw
            par_var_after_on_d[ind]=get_var_one_row_selected_cols(pf_par_on.iloc[r],iloc_ind_st,iloc_ind_st+ntrials_after_sw+ntrials_pre_sw,var_func=var_func,return_na_if_len_mismatch=True)

            par_after_on=pf_par_on.iloc[r].dropna().iloc[iloc_ind_st:iloc_ind_st+ntrials_after_sw+ntrials_pre_sw].reset_index(drop=True)
            # par_after_on_d[ind] = par_after_on
            par_after_on_d.append(par_after_on)
        except:
            pass
    
    par_var_after_on_d=pd.Series(par_var_after_on_d)
    par_after_on_d = pd.concat(par_after_on_d,axis=1).T

    par_var_nosw_sw_df = pd.concat({'nosw':par_var_nosw_d,'sw':par_var_after_on_d},axis=0)
    par_var_nosw_sw_df = par_var_nosw_sw_df.reset_index(level=0).rename({'level_0':'has_sw',0:f'{par_key}_{var_func}'},axis=1)
    
    par_var_nosw_sw_df['isnovel'] = par_var_nosw_sw_df.index.get_level_values(2)

    
    par_nosw_sw_df = pd.concat({'nosw':par_nosw_d,'sw':par_after_on_d},axis=0)
    # par_nosw_sw_df = par_nosw_sw_df.dropna(axis=0)

    return par_var_nosw_sw_df,par_nosw_sw_df


def get_sw_no_sw_pf_par_var(changes_df_all,pf_params_recombined_all,par_key='peak',min_size=2,half_nosw_trials=True ):
    # seperating pop up and non pop up
    # here the neurons are paired
    ma_d = {}
    for on_count in [0,1]:
    #     ma = ((changes_df_all==1).sum(axis=1)==on_count) & ((changes_df_all==-1).sum(axis=1)==0)
        ma = ((changes_df_all==1).sum(axis=1)==on_count) 
        ma_d[on_count] = ma
        

    # for no sw fields, subselect the stable firing ones
    fr_thresh = 1
    frac_trial_thresh = 0.9
    firing_most_trials=pf_params_recombined_all.loc[(slice(None),slice(None),'fr_peak')].apply(lambda x:(x.dropna()>fr_thresh).mean()>frac_trial_thresh,axis=1) 

    # further filter out non firing cells
    ma_d[0] = np.logical_and(ma_d[0] ,firing_most_trials)

    # # further filter: only fr_peak variance lower than in switched ones
    # fr_peak_var_sw_mean=pf_params_recombined_all.loc[(slice(None),slice(None),'fr_peak')].loc[ma_d[1]].var(axis=1).mean()
    # ma_sub=pf_params_recombined_all.loc[(slice(None),slice(None),'fr_peak')].loc[ma_d[0]].var(axis=1) <= fr_peak_var_sw_mean
    # ma_d[0] = np.logical_and(ma_d[0],ma_sub)




    masked=changes_df_all.loc[ma_d[1]].fillna(-0.1)==1
    row_ind,col_ind = np.nonzero(masked.values)

    pf_par_on =pf_params_recombined_all.loc[(slice(None),slice(None),par_key)].loc[ma_d[1]]
    pf_par_nosw =pf_params_recombined_all.loc[(slice(None),slice(None),par_key)].loc[ma_d[0]]
    

    
    if 'fr' in par_key:
        do_fano=True
    else:
        do_fano=False
    # prep no sw variables
    gpb=pf_par_nosw.groupby(level=(0,1,2,3))
    # exclude the first few trials
    var_combined={}
    
    for k,val in gpb:
        val = val.loc[k].dropna(axis=1,how='all')
        ntrials = val.shape[1]
        if half_nosw_trials:
            ntrials = ntrials//2
        if do_fano:
            var_combined[k]=val.iloc[:,min_size:ntrials].var(axis=1) / val.iloc[:,min_size:ntrials].mean(axis=1)
        else:
            var_combined[k]=val.iloc[:,min_size:ntrials].var(axis=1)
        
    var_combined = pd.concat(var_combined,axis=0)
    pf_par_nosw_var_per_neuron_unstack_halftrial = var_combined.groupby(level=(0,1,2,4)).mean().unstack(level=2).dropna(axis=0)
    
    # prep sw variables
    com_after_on_d = {}
    com_var_after_on_d = {}
    
    for r,c in zip(row_ind,col_ind):
        com_after_on = pf_par_on.iloc[r,c:].dropna().values
        ind=pf_par_on.index[r]
        com_after_on_d[ind]=pd.Series(com_after_on)
        if do_fano:
            com_var_after_on_d[ind] = com_after_on.var() / com_after_on.mean()
        else:
            com_var_after_on_d[ind] = com_after_on.var()

    # com_after_on_d_df = pd.concat(com_after_on_d,axis=0).unstack(level=-1)
    com_var_after_on_d=pd.Series(com_var_after_on_d)
    com_var_after_on_d_per_neuron_unstack= com_var_after_on_d.groupby(level=(0,1,2,4)).mean().unstack(level=2).dropna(axis=0)

#     pf_par_nosw_var = pf_par_nosw.var(axis=1)

#     pf_par_nosw_var_per_neuron_unstack=pf_par_nosw_var.groupby(level=(0,1,2,4)).mean().unstack(level=2).dropna(axis=0)

    # pf_par_both_sw_fam_nov=pd.concat({'nosw':pf_par_nosw_var_per_neuron_unstack,'sw':com_var_after_on_d_per_neuron_unstack},axis=1)

    # combine both
    pf_par_both_sw_fam_nov=pd.concat({'nosw':pf_par_nosw_var_per_neuron_unstack_halftrial,'sw':com_var_after_on_d_per_neuron_unstack},axis=1)


    return pf_par_both_sw_fam_nov

def filter_by_si_simple(par_var_nosw_sw_df,per_field_metrics_all,si_thresh=0.5):
    si_thresh=0.5
    si_ma = per_field_metrics_all['si'] > si_thresh
    ind_intersect = si_ma.index.intersection(par_var_nosw_sw_df.index)
    par_var_nosw_sw_df_highsi=par_var_nosw_sw_df.loc[si_ma.loc[ind_intersect]]
    data = par_var_nosw_sw_df_highsi
    return data

def plot_strip_box(par_var_nosw_sw_df,per_field_metrics_all,
            par_key = 'fr_peak',
            var_func='cv',
            random_start = 2,
            nosw_start_pre_peak = False,
            nosw_start_pre_min=False,
            nosw_start_frac=None,
            si_thresh=0.5,
            figsize = (3.9,2),
            savefig=False,
            figdir='',
            ntrials_after_sw=5,
                   
            ):
    x='isnovel'#'has_sw'
    y=f'{par_key}_{var_func}'
    hue='has_sw_int'#'has_sw'#'isnovel'
    fig,ax=plt.subplots(figsize=figsize)
    
    si_ma = per_field_metrics_all['si'] > si_thresh
    ind_intersect = si_ma.index.intersection(par_var_nosw_sw_df.index)
    par_var_nosw_sw_df_highsi=par_var_nosw_sw_df.loc[si_ma.loc[ind_intersect]]
    data = par_var_nosw_sw_df_highsi

    data['has_sw_int'] = data['has_sw'].apply(lambda x:{'nosw':0,'sw':1}[x])

    palette = {0:'blue',1:'red'}

    sns.stripplot(data=data,x=x,y=y,hue=hue,ax=ax,order=[0,1],hue_order=[0,1],dodge=True,alpha=1,s=2.,palette=palette)
    sns.boxplot(data=data,x=x,y=y,hue=hue,ax=ax,order=[0,1],hue_order=[0,1],palette=palette,medianprops=dict(color="white"),)

    sns.despine()
    ax.set(ylabel=f'{var_func} (field {par_key})',xlabel='')
    ax.set_xticklabels(['Familiar','Novel'])

    handles,labels=ax.get_legend_handles_labels()
    new_labels=['No Switch','Has switch-ON']
    ax.legend(handles[2:], new_labels,bbox_to_anchor=[0.9,1])

    ## add test
    from statannotations.Annotator import Annotator

    pairs = [
        ((0,0),(0,1)),
        ((1,0),(1,1)),
        ((0,0),(1,0)),
        ((0,1),(1,1)),
    ]


    annotator = Annotator(ax, pairs, data=data, x=x,hue=hue, y=y)
    annotator.configure(test='Mann-Whitney', text_format='star')#, loc='outside')
    annotator.apply_and_annotate()
    if savefig:
        for fmt in ['png','svg']:
            fn_full = f'{y}_vs_{x}_hue_{hue}_ntrials_after_sw_{ntrials_after_sw}_si_thresh_{si_thresh}_noswstartfrac_{nosw_start_frac:.0e}.{fmt}'
            if random_start is not None:
                fn_full = f'{y}_vs_{x}_hue_{hue}_ntrials_after_sw_{ntrials_after_sw}_si_thresh_{si_thresh}_randomstart_{random_start}.{fmt}'
            elif nosw_start_pre_peak:
                fn_full = f'{y}_vs_{x}_hue_{hue}_ntrials_after_sw_{ntrials_after_sw}_si_thresh_{si_thresh}_nosw_start_pre_peak.{fmt}'
            elif nosw_start_pre_min:
                fn_full = f'{y}_vs_{x}_hue_{hue}_ntrials_after_sw_{ntrials_after_sw}_si_thresh_{si_thresh}_nosw_start_pre_min.{fmt}'                

            fn_full = os.path.join(figdir,fn_full)
            print(fn_full,' saved!')
            fig.savefig(fn_full,bbox_inches='tight')


def get_relative_trace_filter_by_si(par_nosw_sw_df,per_field_metrics_all,trial_ind_relative=0,si_thresh=0.5,drift_thresh=100):
    par_nosw_sw_df_relative=par_nosw_sw_df - par_nosw_sw_df[trial_ind_relative].values[:,None]
    si_ma = per_field_metrics_all['si'] > si_thresh
    ind_intersect = si_ma.index.intersection(par_nosw_sw_df.droplevel(0).index)
    par_nosw_sw_df_relative = par_nosw_sw_df_relative.unstack(level=0).loc[si_ma.loc[ind_intersect]].stack(level=1).reorder_levels((6,0,1,2,3,4,5))
    ma=(par_nosw_sw_df_relative.abs()<=drift_thresh).all(axis=1)
    data=par_nosw_sw_df_relative.loc[ma]
    return data

def plot_drift_trace(par_nosw_sw_df,per_field_metrics_all,trial_ind_relative=0,
                     drift_thresh = 20,
                     si_thresh=0.5,
                     nosw_start_pre_peak=False,
                    nosw_start_pre_min=False,
                     random_start = 2,
                     ntrials_after_sw=5,
                     nosw_start_frac=None,
                     savefig=False,
                     figdir='',
                     figsize=(2,2),
                     par_key='peak'
                    ):
    par_nosw_sw_df_relative=par_nosw_sw_df - par_nosw_sw_df[trial_ind_relative].values[:,None]
    si_ma = per_field_metrics_all['si'] > si_thresh
    
    ind_intersect = si_ma.index.intersection(par_nosw_sw_df.droplevel(0).index)
    par_nosw_sw_df_relative = par_nosw_sw_df_relative.unstack(level=0).loc[si_ma.loc[ind_intersect]].stack(level=1).reorder_levels((6,0,1,2,3,4,5))
    ma=(par_nosw_sw_df_relative.abs()<=drift_thresh).all(axis=1)
    data=par_nosw_sw_df_relative.loc[ma]

    data_melt=data.groupby(level=(0,3)).apply(lambda x:x.melt()).reset_index(level=(0,1)).rename({'variable':'n_trial_after','value':'drift','level_0':'has_sw','level_1':'isnovel'},axis=1)
    data_melt = data_melt.reset_index(drop=True)


    palette={'nosw':'blue','sw':'red'}

    isnovel_str_d={0:'Familiar',1:'Novel'}
    fig,axs=plt.subplots(2,1,figsize=figsize,sharex=True)
    for isnovel in [0,1]:
        ax = axs[isnovel]
        ax.axhline(0,c='k',linestyle=':')
        sns.lineplot(data=data_melt.query('isnovel==@isnovel'),x='n_trial_after',y='drift',hue='has_sw',palette=palette,ax=ax,estimator='mean')
        sns.despine()
        # ax.legend(bbox_to_anchor=[0.9,1.])
        ax.get_legend().set_visible(False)
        # ax.set(ylabel='Pos. shift (bin)',xlabel='Trial (relative)')
        ax.set(xlabel='Trial (relative)',ylabel=None)
        ax.set(xticks=par_nosw_sw_df.columns)
        ax.set_title(isnovel_str_d[isnovel])

        handles,labels=ax.get_legend_handles_labels()
        new_labels=['No switch','Has switch ON']
        # ax.legend(handles, new_labels,bbox_to_anchor=[1.05,1])

        # if savefig:
        #     for fmt in ['png','svg']:
        #         fn_full = f'{par_key}_drift_vs_trial_ntrials_after_sw_{ntrials_after_sw}_isnovel_{isnovel}_si_thresh_{si_thresh:.0e}_noswstartfrac_{nosw_start_frac:.0e}.{fmt}'
        #         if random_start is not None:
        #             fn_full = f'{par_key}_drift_vs_trial_ntrials_after_sw_{ntrials_after_sw}_isnovel_{isnovel}_si_thresh_{si_thresh:.0e}_randomstart_{random_start}.{fmt}'
        #         elif nosw_start_pre_peak:
        #             fn_full = f'{par_key}_drift_vs_trial_ntrials_after_sw_{ntrials_after_sw}_isnovel_{isnovel}_si_thresh_{si_thresh:.0e}_nosw_start_pre_peak.{fmt}'
        #         elif nosw_start_pre_min:
        #             fn_full = f'{par_key}_drift_vs_trial_ntrials_after_sw_{ntrials_after_sw}_isnovel_{isnovel}_si_thresh_{si_thresh:.0e}_nosw_start_pre_min.{fmt}'                

        #         fn_full = os.path.join(figdir,fn_full)
        #         print(fn_full,' saved!')
        #         fig.savefig(fn_full,bbox_inches='tight')
    ax.legend(handles, new_labels,bbox_to_anchor=[1.05,1])
    sup=fig.supylabel('Pos. shift (bin)',fontsize=10)
    sup.set_position([-0.15,0.5])
    plt.subplots_adjust(hspace=0.5)

    if savefig:
        for fmt in ['png','svg']:
            fn_full = f'{par_key}_drift_vs_trial_ntrials_after_sw_{ntrials_after_sw}_si_thresh_{si_thresh:.0e}_noswstartfrac_{nosw_start_frac:.0e}.{fmt}'
            if random_start is not None:
                fn_full = f'{par_key}_drift_vs_trial_ntrials_after_sw_{ntrials_after_sw}_si_thresh_{si_thresh:.0e}_randomstart_{random_start}.{fmt}'
            elif nosw_start_pre_peak:
                fn_full = f'{par_key}_drift_vs_trial_ntrials_after_sw_{ntrials_after_sw}_si_thresh_{si_thresh:.0e}_nosw_start_pre_peak.{fmt}'
            elif nosw_start_pre_min:
                fn_full = f'{par_key}_drift_vs_trial_ntrials_after_sw_{ntrials_after_sw}_si_thresh_{si_thresh:.0e}_nosw_start_pre_min.{fmt}'                

            fn_full = os.path.join(figdir,fn_full)
            print(fn_full,' saved!')
            fig.savefig(fn_full,bbox_inches='tight')