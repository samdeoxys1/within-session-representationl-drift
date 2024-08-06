import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'
FS = 10

def sort_fr_map(fr_map):
    peaks_loc = fr_map.idxmax(axis=1)
    peaks_loc.loc[fr_map.sum(axis=1)==0] = -1 # make the not firing ones go to the top
    peaks_loc_sorted = peaks_loc.sort_values()
    order = peaks_loc_sorted.index
    return order

def plot_population_sequence(fr_map,order=None,fr_map_for_order=None,fig=None,ax=None,normalize=True,vmax=None,**kwargs):
    '''
    if fr_map_for_order and order: use fr_map_for_order to compute order
    '''
    cmap = kwargs.get('cmap','viridis')
    if ax is None:
        fig,ax=plt.subplots()
    
    if order is None:
        if fr_map_for_order is not None:
            order = sort_fr_map(fr_map_for_order)
        else:
            order = sort_fr_map(fr_map)        
    fr_map_sorted = fr_map.loc[order]
    if normalize:
        fr_map_sorted_normed = fr_map_sorted / fr_map_sorted.max(axis=1).values[:,None]
    else:
        fr_map_sorted_normed = fr_map_sorted
    if vmax is None:
        vmax = fr_map_sorted_normed.max().max() * 0.95
    ax.imshow(fr_map_sorted_normed,aspect='auto',vmax=vmax,interpolation='None',cmap=cmap)
    ax.set(frame_on=False)
    # ax=sns.heatmap(fr_map_sorted_normed,vmax=vmax,ax=ax,cmap=cmap)
    
    return fig, ax, order
    
def plot_multi_sequences_multi_orders(fr_map_l,order_l=None,fig=None,axs=None,normalize=True,reduce=False,vmax=None):
    '''
    if reduce: need to average before plotting
    '''
    nmap = len(fr_map_l)
    if order_l is None:
        order_l = [None] * nmap
    assert len(order_l) == nmap
    if axs is None:
        fig,axs = plt.subplots(nmap,nmap,figsize=(nmap * 4,nmap*4))
    
    for ii in range(nmap):
        order = order_l[ii]
        if reduce:
            ntrials_sub = fr_map_l[ii].shape[1]
            fr_map_for_plot = fr_map_l[ii].iloc[:,::2].mean(axis=1).unstack()
            fr_map_for_order = fr_map_l[ii].iloc[:,1::2].mean(axis=1).unstack()
        else:
            fr_map_for_plot = fr_map_l[ii]
            fr_map_for_order = None
            
        fig,ax,order=plot_population_sequence(fr_map_for_plot,order=order,fr_map_for_order=fr_map_for_order,fig=fig,ax=axs[ii,ii],normalize=normalize,vmax=vmax)
        order_l[ii] = order
    
    for ii in range(nmap):
        order = order_l[ii]
        for jj in range(nmap):
            if ii!=jj:
                if reduce:
                    fr_map_for_plot = fr_map_l[jj].mean(axis=1).unstack()
                else:
                    fr_map_for_plot = fr_map_l[jj]
                fig,ax,_=plot_population_sequence(fr_map_for_plot,order=order,fr_map_for_order=None,fig=fig,ax=axs[ii,jj],normalize=normalize,vmax=vmax)
    
    return fig,axs,order_l
    
def divide_fr_map_trial_into_blocks(fr_map_trial_df,nblocks=3,reduce=True): 
    '''
    fr_map_trial_df: (nneuron x nposbins) x ntrials
    reduce: average into one ratemap, or keep the trials
    '''
    fr_map_trial_df = fr_map_trial_df.dropna(axis=1)
    ntrials = fr_map_trial_df.shape[1]
    edges = np.linspace(0,ntrials,nblocks+1).astype(int)
    fr_map_l = []
    st_end_l = []
    for i in range(len(edges)-1):
        if reduce:
            fr_map_one = fr_map_trial_df.loc[:,edges[i]:edges[i+1]].mean(axis=1).unstack()
        else:
            fr_map_one = fr_map_trial_df.loc[:,edges[i]:edges[i+1]]
        fr_map_l.append(fr_map_one)
        st_end_l.append((edges[i],edges[i+1]))
    return fr_map_l,st_end_l
    
    
    

# subselect the switching neurons
def get_coswitching_field_one_trial(trial_index,changes_df,all_fields,trial_index_to_index_within_df,
                                    task_ind=0,onoff=1,
                                   ):
    '''
    get rid of duplicated and only keep the first
    '''
    tt_ind_within=trial_index_to_index_within_df.loc[task_ind,slice(None),trial_index]
    tt=tt_ind_within.index[0]
    index_within=tt_ind_within.values[0]
    all_index_within = trial_index_to_index_within_df.loc[task_ind,tt].values # sometimes it's not the same as arange, because some trials are dropped as bad trials
    index_within_within = np.nonzero(all_index_within==index_within)[0][0]
    cd_sub=changes_df.loc[task_ind,trial_index].dropna()==onoff
    uid_field_l=cd_sub.index[cd_sub]
    uid_l = uid_field_l.get_level_values(1)
    ma=np.logical_not(uid_l.to_series().duplicated().values)
    uid_l = uid_l[ma]
    uid_field_l = uid_field_l[ma]


    field_pos_l = all_fields.loc[task_ind].loc[uid_field_l]['peak']
    uid_field_l_sorted = field_pos_l.sort_values().index
    uid_l_sorted=uid_field_l_sorted.get_level_values(1)
    
    return int(tt),index_within,index_within_within,uid_l_sorted,uid_field_l_sorted
    
def plot_sequence_around_switch(trial_index,fr_map_trial_df,changes_df,all_fields,
                                trial_index_to_index_within_df,
                                task_ind=0,onoff=1,
                                norm_fr_map_per_neuron=True,n_tr_to_show=5,fig=None,axs=None):

    tt,index_within,index_within_within,uid_l_sorted,uid_field_l_sorted = get_coswitching_field_one_trial(trial_index,changes_df,all_fields,trial_index_to_index_within_df,
                                    task_ind=task_ind,onoff=onoff)

#     tt_ind_within=trial_index_to_index_within_df.loc[task_ind,slice(None),trial_index]
#     tt=tt_ind_within.index[0]
#     index_within=tt_ind_within.values[0]
    all_index_within = trial_index_to_index_within_df.loc[task_ind,tt].values # sometimes it's not the same as arange, because some trials are dropped as bad trials
#     index_within_within = np.nonzero(all_index_within==index_within)[0][0]
#     cd_sub=changes_df.loc[task_ind,trial_index].dropna()==onoff
#     uid_field_l=cd_sub.index[cd_sub]
#     uid_l = uid_field_l.get_level_values(1)
#     field_pos_l = all_fields.loc[task_ind].loc[uid_field_l]['peak']
#     uid_field_l_sorted = field_pos_l.sort_values().index
#     uid_l_sorted=uid_field_l_sorted.get_level_values(1)
    
    
    
    if norm_fr_map_per_neuron:
        fr_map_trial_df_norm = fr_map_trial_df.groupby(level=(0,1,2)).apply(lambda x:x/(x.max().max()))
    else:
        fr_map_trial_df_norm = fr_map_trial_df
    if axs is None:
        fig,axs=plt.subplots(1,n_tr_to_show,figsize=(1.6*n_tr_to_show,1.5),sharey=True)

    order = uid_l_sorted
    vmax = 1
    
    for ii in range(n_tr_to_show):
        try:
            tr = index_within_within - (n_tr_to_show//2)+ii # get the index within all_index_within
            tr = all_index_within[tr] # translate it into index_within
#             tr = index_within-(n_tr_to_show//2)+ii
            sub_pop_ratemap_one_trial=fr_map_trial_df_norm.loc[(task_ind,tt,uid_l_sorted),tr].unstack(level=-1).droplevel((0,1))
            vmax_=sub_pop_ratemap_one_trial.max().max() * 0.9
            # if vmax_ >vmax:
            if vmax_ < vmax:
                vmax=vmax_
        except:
            pass
    for ii in range(n_tr_to_show):
        try:
#             tr = index_within-(n_tr_to_show//2)+ii
            tr = index_within_within - (n_tr_to_show//2)+ii
            tr = all_index_within[tr]
            ax=axs[ii]
            sub_pop_ratemap_one_trial=fr_map_trial_df_norm.loc[(task_ind,tt,uid_l_sorted),tr].unstack(level=-1).droplevel((0,1))
            fig,ax,order=plot_population_sequence(sub_pop_ratemap_one_trial,order=order,normalize=False,fig=fig,ax=ax,vmax=vmax)
            title=tr-index_within
            if title==0:
                title='Switch trial'
            elif title>0:
                title=f'+{title}'
            ax.set_title(title)
            ax.set_xticks([])
        except Exception as e:
            print(e)
            pass
    
    
    
    axs[0].set_ylabel('Neuron')
    sxl=fig.supxlabel('Position',fontsize=FS)
    sxl.set_position([0.53,0.07])
    plt.tight_layout()
    
    cbar_ax = fig.add_axes([0.99, 0.35, 0.01, 0.4])  # Adjust [left, bottom, width, height] as needed
    fig.colorbar(ax.images[0], cax=cbar_ax)

    return fig,axs, uid_field_l_sorted,tt,index_within
    