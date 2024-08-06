# %%
from __future__ import unicode_literals
from asyncio import to_thread
import numpy as np
import scipy
import data_prep_new as dpn
import place_cell_analysis as pa
import plot_helper as ph
from importlib import reload
import itertools, sys, os, copy, pickle, pathlib
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import plot_raster as pr
import database

# %%
# fn="/mnt/home/szheng/ceph/ad/Chronic_H2/AZ10/AZ10_210315_sess1/py_data/shifted_fr_map.npz"
# res=np.load(fn)
# # %%
# res['1'].shape
# # %%
# DATABASE_LOC = '/mnt/home/szheng/ceph/ad/database.csv'
# db = pd.read_csv(DATABASE_LOC,index_col=[0,1])
# # db.sort_values('n_pyr_putative',ascending=False).groupby(level=0).head(1)
# azsess='AZ11_210504_sess10'
# nazsess='Naz2_210323_sess1'#'Naz1_210518_sess17'#'Naz2_210422_sess8'
# romsess='e15_13f1_220117'
# %%
db = database.db

# sess_name=nazsess
# data_dir_full = db.query(f'sess_name=="{sess_name}"').loc[:,'data_dir_full'].iloc[0]

# cell_metric,behavior,spike_times,uid,fr,cell_type,mergepoints,behav_timestamps,position,\
#                 rReward,lReward,endDelay,startPoint,visitedArm \
#     = dpn.load_sess(sess_name=sess_name, data_dir=None, data_dir_full=data_dir_full)

# df_dict, pos_bins_dict,cell_cols_dict = dpn.get_fr_beh_df(spike_times,uid,behav_timestamps,cell_type,position,visitedArm,startPoint,n_pos_bins=100)

# fr_map_all_trials_dict = pa.get_fr_map_trial(df_dict['pyr'],cell_cols_dict['pyr'],gauss_width=2.5,order=['smooth','divide'])

# fr_map_all_trials_dict = {0:fr_map_all_trials_dict[0][0],1:fr_map_all_trials_dict[1][0]}

# fr_map_dict = pa.get_fr_map_trial(df_dict['pyr'],cell_cols_dict['pyr'],gauss_width=2.5,order=['smooth','divide','average'],n_lin_bins=100)

# fr_map_dict = {0:fr_map_dict[0][0],1:fr_map_dict[1][0]}

# %%

def corrcoef3d(x,zscoredim,elwisedim,outerdim):
    '''
    get the elementwise corr matrix for some conditions
    ===
    x: n_neurons x n_places x n_trials
    zscoredim: the dim to do the correlation; leaving the rest untouched; eg for a neuron 
    elwisedim: the dim, the correlation is for each element within this dimension, eg neuron
    outerdim: the dim where outer product is done; eg trial by trial
    ==
    the idea of the einsum is to denote each dimension of the input by an index, then use the order and existence of the index in the output specification
    to denote what to do with each dimensions of the input; 
    thus to specify which dimension we want to apply an operation to, 
    we use fixed indices to specify the template of operation, ie output will always be 0,2,3, as 0->elwisedim, 2,3->outerdim for each
    then we just need to determine how to map each dimension of the input to these template indices 
    '''
    x_std = x.std(axis=zscoredim,keepdims=True)
    x_zscored = np.divide ( (x - np.mean(x,axis=zscoredim,keepdims=True)), x_std, out=np.nan*np.ones_like(x),where=x_std!=0)
    # corr = np.einsum('npt,npr->ntr',x_zscored,x_zscored)/x.shape[zscoredim]
    index_1 = np.zeros(3,dtype=int)
    index_1[elwisedim]=0
    index_1[zscoredim]=1
    index_1[outerdim]=2
    index_2 = np.zeros(3,dtype=int)
    index_2[elwisedim]=0
    index_2[zscoredim]=1
    index_2[outerdim]=3
    corr = np.einsum(x_zscored, index_1,x_zscored, index_2, [0,2,3]) /x.shape[zscoredim]
    corr_triu = np.triu(corr,k=1)
    triuinds=np.triu_indices(corr.shape[1],k=1)
    corr_triu=np.array([mat[triuinds] for mat in corr_triu])
    return corr,corr_triu

# %% 
# x=fr_map_all_trials_dict[0]
# zscoredim=1
# elwisedim=0
# outerdim=2
# corr,corr_triu = corrcoef3d(x,zscoredim,elwisedim,outerdim)

# %%
def get_variability_by_trial(fr_map_all_trials_dict,dosave=False,savedir=""):
    
    zscoredim=1
    elwisedim=0
    outerdim=2
    corr_dict={}
    corr_triu_dict={}
    corr_stats_dict_all_neurons = {0:{},1:{}}
    corr_stats_dict_agg = {0:{},1:{}}
    for k,fr_map_all_trials in fr_map_all_trials_dict.items():
        x = fr_map_all_trials
        corr_dict[k],corr_triu_dict[k] = corrcoef3d(x,zscoredim,elwisedim,outerdim)
        corr_stats_dict_all_neurons[k]['median'] = np.nanmedian(corr_triu_dict[k],axis=1)
        corr_stats_dict_all_neurons[k]['mean'] = np.nanmean(corr_triu_dict[k],axis=1)
        corr_stats_dict_all_neurons[k]['std'] = np.nanstd(corr_triu_dict[k],axis=1)
        corr_stats_dict_agg[k]['median_median'] = np.nanmedian(corr_stats_dict_all_neurons[k]['median'])
    res = {'corr':corr_dict,'corr_triu':corr_triu_dict,'corr_stats_allneuron':corr_stats_dict_all_neurons,'corr_stats_agg':corr_stats_dict_agg}
    if dosave:
        fn = os.path.join(savedir,'variability_by_trial.p')
        pickle.dump(res,open(fn,'wb'))
        print(f'{fn} saved')
    return res

# res = get_variability_by_trial(fr_map_all_trials_dict)
# %%
# fr_dict={}
# for k in [0,1]:
#     # fr_dict[k] = fr_map_dict[k].mean(axis=1)
#     fr_dict[k] = fr_map_dict[k].max(axis=1)
# k=0
# plt.scatter(fr_dict[k],res['corr_stats_allneuron'][k]['median'])
# plt.show()
# %%
from functools import reduce
def load_mask_placecells(sess_name,data_dir_full=None,peak_thresh=1):
    '''
    mask_dict:{0:,1:}, each mask is a pd.Series, index are the unit names, values are Bools for whether it is a place cell
    '''
    # sess_name='AZ10_210315_sess1'
    data_dir_full = db.query(f'sess_name=="{sess_name}"').loc[:,'data_dir_full'].iloc[0]
    pydata_folder = os.path.join(data_dir_full, 'py_data')

    shifted_fr_map = np.load(os.path.join(pydata_folder,'shifted_fr_map.npz'),allow_pickle=True)
    unit_names = shifted_fr_map['unit_names']
    shifted_fr_map = {0:shifted_fr_map['0'],1:shifted_fr_map['1']}

    fr_map_res = pickle.load(open(os.path.join(pydata_folder,'fr_map.p'),'rb'))
    fr_map = fr_map_res['fr_map']

    mask_dict={}
    for k in shifted_fr_map.keys():
        thresh = np.quantile(shifted_fr_map[k],0.99,axis=0)
        thresh_df=pd.DataFrame(thresh,index=unit_names)
        # intersection = list(set(thresh_df.index)&set(fr_map[k]))
        mask_shuffle = (fr_map[k] > thresh_df.loc[fr_map[k].index]).sum(axis=1)>0 # assume thresh_df / unit_names is a superset of fr_map[k].index, since the latter excludes interneurons
        mask_peak = (fr_map[k] > peak_thresh).sum(axis=1)>0

        mask = mask_shuffle & mask_peak
        mask_dict[k] = mask
    # final_mask = reduce(np.logical_or,mask_dict.values())
    return mask_dict

def mask_trial_by_occupancy(fr_map_trial_dict,occ_map_trial_dict,thresh_ratio=0.1):
    '''
    if more than a thresh of position bins have 0 occupancy, throw out that trial
    '''

    thresh = thresh_ratio * occ_map_trial_dict[0].shape[0]
    mask_dict = {k:(occ==0).sum(axis=0)<thresh for k,occ in occ_map_trial_dict.items()}
    fr_map_trial_dict_new = {k:fr_map_trial_dict[k][:,:,mask] for k,mask in mask_dict.items()}
    return fr_map_trial_dict_new

def load_mask_get_stability(sess_name,data_dir_full=None,peak_thresh=1,skip_mask=False):
    if not skip_mask:      
        mask_dict = load_mask_placecells(sess_name,data_dir_full=data_dir_full,peak_thresh=peak_thresh)
    # sess_name='AZ10_210315_sess1'
    data_dir_full = db.query(f'sess_name=="{sess_name}"').loc[:,'data_dir_full'].iloc[0]
    pydata_folder = os.path.join(data_dir_full, 'py_data')

    fr_map_res = pickle.load(open(os.path.join(pydata_folder,'fr_map.p'),'rb'))
    fr_map_trial_dict = fr_map_res['fr_map_trial']
    occ_map_trial_dict = fr_map_res['occupancy_map_trial']

    fr_map_trial_dict_new = mask_trial_by_occupancy(fr_map_trial_dict,occ_map_trial_dict,thresh_ratio=0.1)

    res = get_variability_by_trial(fr_map_trial_dict_new)

    corr_stats_allneuron=res['corr_stats_allneuron']
    agg_stats_dict={}
    per_neuron_dict={}
    for k,csa in corr_stats_allneuron.items():
        if not skip_mask:
            corr_median_per_neuron_masked=csa['median'][mask_dict[k].values]
        else:
            corr_median_per_neuron_masked=csa['median']
        per_neuron_dict[k] = corr_median_per_neuron_masked
        agg_stats_dict[k]=np.nanmedian(corr_median_per_neuron_masked)

    return agg_stats_dict, per_neuron_dict

# agg_stats_dict, per_neuron_dict = load_mask_get_stability(sess_name)

def get_stability_per_animal(animal_name,skip_mask=False,dosave=False,savedir='Analysis/py_data/place_cell_metrics',savefn='trial_by_trial_stability.p',force_reload=False):
    '''
    skip_mask: if False, exclude non-place cells
    '''
    data_dir_full = db.loc[db.index.get_level_values(0)==animal_name,'data_dir_full'].iloc[0]
    animal_dir = str(Path(data_dir_full).parent)
    if dosave:
        savedir_full = os.path.join(animal_dir,savedir)
        if not os.path.isdir(savedir_full):
            os.makedirs(savedir_full)
            print(f'{savedir_full} created!')
        savefn_full = os.path.join(savedir_full,savefn)
        if os.path.exists(savefn_full) and not force_reload:
            res=pickle.load(open(savefn_full,'rb'))
            return res['agg'],res['per_neuron'],res['sess']

    db_ani = db.loc[animal_name]
    failed_list = []
    agg_stats_dict_l = []
    sess_l = []
    per_neuron_dict_l = []

    for sess_name in db_ani['sess_name']:
        try:
            agg_stats_dict, per_neuron_dict = load_mask_get_stability(sess_name,skip_mask=skip_mask)
            agg_stats_dict_l.append(agg_stats_dict)
            per_neuron_dict_l.append(per_neuron_dict)
            sess_l.append(sess_name)
        except:
            failed_list.append(sess_name)

    agg_stats_dict_l=pd.DataFrame(agg_stats_dict_l)
    per_neuron_dict_l=np.array(per_neuron_dict_l,dtype=object)
    sess_l=np.array(sess_l,dtype=object)
    agg_stats_dict_l['sess'] = sess_l

    res={'agg':agg_stats_dict_l,'per_neuron':per_neuron_dict_l,'sess':sess_l,'failed':failed_list}
    if dosave:
        pickle.dump(res,open(savefn_full,'wb'))
        print(f"{savefn_full} saved!")

    return agg_stats_dict_l, per_neuron_dict_l, sess_l


# animal_name='AZ11'#'e16_2m1' #
# data_dir_full = db.loc[db.index.get_level_values(0)==animal_name,'data_dir_full'].iloc[0]
# animal_dir = str(Path(data_dir_full).parent)
# month_fn = os.path.join(animal_dir,'Months.mat')
# month = dpn.loadmat(month_fn)['months']
# agg_stats_dict_l, per_neuron_dict_l, sess_l = get_stability_per_animal(animal_name,skip_mask=False)
# sess_ind_l = np.array([int(sess.split('_')[2][4:]) for sess in sess_l]) # only works if sess_name is like az1_000000_sess1
# month_l = month[sess_ind_l-1]
# agg_stats_dict_l['month'] = month_l
# agg_stats_dict_l['sess'] = sess_ind_l
# # agg_stats_dict_l['month_int'] = np.round(month_l)
# agg_stats_dict_l.plot(x='sess',y=[0,1],marker='.');plt.title(animal_name);plt.show()
#%%
def agg_across_month(agg_stats_dict_l, per_neuron_dict_l, sess_l,month_l):
    '''
    pool neurons from the same month but different sessions together

    agg_stats_dict_l: df: columns are choice=0 and 1, with other info added later; each row is a session; containing the summary statistics of the stability across all neurons in one session and one choice type
    per_neuron_dict_l: [{0:[each neurons's median stability],1:[]}, ...(across sessions)... ]
    sess_l: [name of the corresponding sessions]
    month_l: the actual list of month of the relevant sessions
    
    ===
    agg_all_month_int:{0:{month_int: stability},1:{}}
    per_neuron_all_month_int:{0:{month_int: [stability per neuron]},1:{}}
    '''
    sess_ind_l = np.array([int(sess.split('_')[2][4:]) for sess in sess_l]) # only works if sess_name is like az1_000000_sess1
    # month_l = month[sess_ind_l]
    agg_stats_dict_l['month'] = month_l
    agg_stats_dict_l['month_int'] = np.round(month_l).astype(int)
    agg_stats_dict_l['sess'] = sess_ind_l
    gpb=agg_stats_dict_l.groupby('month_int')
    per_neuron_all_month_int = {0:{},1:{}}
    agg_all_month_int = {0:{},1:{}}
    for k,val in gpb:
        per_neuron_one_month = per_neuron_dict_l[val.index] # relying on the fact that index in agg_stats_dict_l should align with per_neuron_dict_l
        for i in per_neuron_all_month_int.keys():
            per_neuron_all_month_int[i][k] = np.concatenate([x[i] for x in per_neuron_one_month])
            agg_all_month_int[i][k] = np.nanmedian(per_neuron_all_month_int[i][k])
    return agg_all_month_int, per_neuron_all_month_int

# agg_all_month_int, per_neuron_all_month_int = agg_across_month(agg_stats_dict_l, per_neuron_dict_l, sess_l,month_l)

def mask_low_stability_sess(agg_stats_dict_l, *args, thresh = 0.1):
    '''
    mask some sessions that have stability below a threshold; stability below a threshold is deemed problematic
    masking based on agg_stats_dict_l; 
    args: per_neuron_dict_l, sess_l,month_l
    '''
    mask = np.logical_and(agg_stats_dict_l[0] >= thresh,agg_stats_dict_l[1] >= thresh)
    to_return = [agg_stats_dict_l.loc[mask].reset_index()] # remember to reset index such that the index between agg and per_neuron still match
    for arg in args:

        to_return.append(arg[mask])

    return to_return

# agg_stats_dict_l_ma, per_neuron_dict_l_ma, sess_l_ma,month_l_ma=mask_low_stability_sess(agg_stats_dict_l, per_neuron_dict_l, sess_l,month_l, thresh = 0.1)

# agg_all_month_int_ma, per_neuron_all_month_int_ma = agg_across_month(agg_stats_dict_l_ma, per_neuron_dict_l_ma, sess_l_ma,month_l_ma)

def plot_stability_across_sess(per_neuron_all_month,month_l=None,month_agg=False,fig=None,ax=None):
    '''
    depending on month_agg, per_neuron_all_month can have two structures:
    if month_agg:
        per_neuron_all_month: # dict {0:,1:} of dict {month:[]}; as a result of agg_across_month
    else:
        per_neuron_all_month: # list of dict of {0:,1:}
    '''

    if ax is None:
        fig,ax=plt.subplots()

    if month_agg:
        xlabel='month'
        for ch in per_neuron_all_month.keys():        
            xs = list(per_neuron_all_month[ch].keys())
            # plt.boxplot(list(per_neuron_all_month[ch].values()),positions=xs)
            ys = np.array([np.nanmean(ii) for ii in per_neuron_all_month[ch].values()]) # dict {0:,1:} of dict {month:[]}
            yerr = np.array([scipy.stats.sem(ii,nan_policy='omit') for ii in per_neuron_all_month[ch].values()])
            yerr = yerr * 1.96 # for 95% CI
            ax.errorbar(xs,ys,yerr,label=ch)
    else:
        for ch in per_neuron_all_month[0].keys():        
            if month_l is None:
                xlabel='session'
                xs = np.arange(len(per_neuron_all_month))
            else:
                xlabel='month'
                xs=month_l
            # plt.boxplot(list(per_neuron_all_month[ch].values()),positions=xs)
            ys = np.array([np.nanmean(ii[ch]) for ii in per_neuron_all_month]) # list of dict of {0:,1:}
            yerr = np.array([scipy.stats.sem(ii[ch],nan_policy='omit') for ii in per_neuron_all_month])
            yerr = yerr * 1.96 # for 95% CI
            ax.errorbar(xs,ys,yerr,label=ch)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('stability')
    plt.legend()
    # plt.show()
    return fig,ax

# plot_stability_across_sess(per_neuron_dict_l,month_l=month_l,month_agg=False,fig=None,ax=None)
# plot_stability_across_sess(per_neuron_all_month_int,month_l=None,month_agg=True,fig=None,ax=None)

#%%
#next: filter / not filter x agg by month / not agg
def plot_trial_by_trial_stability_one_animal(animal_name,filter_or_not_l = [True,False],agg_or_not_l = [True,False],skip_mask=False,thresh = 0.1,dosave=False,savedir='Analysis/py_figures/place_cell_metrics',savefn='trial_by_trial_stability_mean_neuron.png',force_reload=False):
    '''
    plot: filter out sessions whose stability is too low / not filter x agg by month / not agg
    filter or not: row
    agg or not: col
    '''
    data_dir_full = db.loc[db.index.get_level_values(0)==animal_name,'data_dir_full'].iloc[0]
    animal_dir = str(Path(data_dir_full).parent)
    
    # prepare for saving
    if dosave:
        savedir_full = os.path.join(animal_dir,savedir)
        if not os.path.isdir(savedir_full):
            os.makedirs(savedir_full)
            print(f'{savedir_full} created!')
        savefn_full = os.path.join(savedir_full,savefn)
        if os.path.exists(savefn_full) and not force_reload:
            return None,None

    agg_stats_dict_l, per_neuron_dict_l, sess_l = get_stability_per_animal(animal_name,skip_mask=skip_mask)
    sess_ind_l = np.array([int(sess.split('_')[2][4:]) for sess in sess_l]) # only works if sess_name is like az1_000000_sess1

    try:
        # agg across month
        month_fn = os.path.join(animal_dir,'Months.mat')
        month = dpn.loadmat(month_fn)['months']
        month_l = month[sess_ind_l-1] # sess_ind_l are 1-indices
        # month_int_l = np.round(month).astype(int)
        month_exist=True
    except:
        month_exist = False
        month_l = None
        month_l_ma = None
        print('no month available')
    
    if month_exist:
        # masked
        agg_stats_dict_l_ma, per_neuron_dict_l_ma, sess_l_ma,month_l_ma=mask_low_stability_sess(agg_stats_dict_l,per_neuron_dict_l, sess_l,month_l , thresh = thresh)
        # agg by month
        agg_all_month_int, per_neuron_all_month_int = agg_across_month(agg_stats_dict_l, per_neuron_dict_l, sess_l,month_l)
        agg_all_month_int_ma, per_neuron_all_month_int_ma = agg_across_month(agg_stats_dict_l_ma, per_neuron_dict_l_ma, sess_l_ma,month_l_ma)
    else:
        # masked
        agg_stats_dict_l_ma, per_neuron_dict_l_ma, sess_l_ma=mask_low_stability_sess(agg_stats_dict_l,per_neuron_dict_l, sess_l , thresh = thresh)

    fig,axes = plt.subplots(2,2,figsize=(12,12),sharex=True,sharey=True)
    # for ax in axes.ravel():
    #     ax.set_axis_off()
    
    data_dict = {True:{True:per_neuron_all_month_int_ma,False:per_neuron_dict_l_ma},False:{True:per_neuron_all_month_int,False:per_neuron_dict_l}}
    month_l_arg_dict = {True:{True:None,False:month_l_ma},False:{True:None,False:month_l}}

    for ii,dofilter in enumerate(filter_or_not_l):
        for jj, doagg in enumerate(agg_or_not_l):
            month_l_arg = month_l_arg_dict[dofilter][doagg]
            fig,axes[ii,jj]=plot_stability_across_sess(data_dict[dofilter][doagg],month_l=month_l_arg,month_agg=doagg,fig=fig,ax=axes[ii,jj])
            # axes[ii,jj].set_axis_on()
            axes[ii,jj].set_title(f'filter bad sess: {dofilter}\n agg within month: {doagg}')
            
    plt.tight_layout()
    if dosave:
        fig.savefig(savefn_full,bbox_inches='tight')
        print(f'{savefn_full} saved!')
    return fig,axes

def main():
    for animal_name in ['Naz1','Naz2','AZ10','AZ11']:
        fig,axes=plot_trial_by_trial_stability_one_animal(animal_name,filter_or_not_l = [True,False],agg_or_not_l = [True,False],skip_mask=False,dosave=True,savedir='Analysis/py_figures/place_cell_metrics',savefn='trial_by_trial_stability_mean_neuron.png')
        print(f'{animal_name} done!')

# main()    
# # %%
# plt.plot(fr_map_res['fr_map'][0].iloc[5])
# plt.show()
# # %%

# per_neuron_sess_concat_dict={}
# for ch in [0,1]:
#     per_neuron_sess_concat_dict[ch]=[per_neuron_dict_one_sess[ch] for per_neuron_dict_one_sess in per_neuron_dict_l]
# plt.boxplot(per_neuron_sess_concat_dict[0])

# plt.show()
# %%
# %%

