import numpy as np
import scipy
import matplotlib.pyplot as plt
import sys,os,copy,pdb,importlib
from importlib import reload
import pandas as pd
import seaborn as sns

sys.path.append('/mnt/home/szheng/projects/seq_detection2/code')
sys.path.append('/mnt/home/szheng/projects/util_code')
sys.path.append('/mnt/home/szheng/projects/nmf_analysis')
sys.path.append('/mnt/home/szheng/projects/nmf_analysis/scripts')
sys.path.append('/mnt/home/szheng/projects/cluster_spikes')
sys.path.append('/mnt/home/szheng/projects/place_variability/code')

import place_cell_analysis as pa
import misc
import plot_helper as ph

def get_sparsity(frmap,occu_map):

    gpb = frmap.groupby(level=(0,1,2,3),sort=False)
    spa_all = []
    for k,val in gpb:
        val = val.dropna(axis=1)
        spa = np.average(val**2 / (val.mean(axis=1)**2).values[:,None],weights=occu_map.loc[k].dropna(),axis=1)
        spa=pd.Series(spa,index=val.index)
        spa_all.append(spa)
    spa_all = pd.concat(spa_all,axis=0)
    return spa_all

def get_variability_metrics(frmap,occu_map,fr_map_trial_df_all):
    si_all = pa.get_si_from_frmap_and_occu(frmap,occu_map)
    fr_cv,mean_fr = pa.get_fr_cv(fr_map_trial_df_all,level_l=(0,1,2,3,4))
    spa_all = get_sparsity(frmap,occu_map)
    per_cell_metrics_d = {'si':si_all,'fr_cv':fr_cv,'mean_fr':mean_fr,'sparsity':spa_all}
    per_cell_metrics = pd.concat(per_cell_metrics_d,axis=1)

    return per_cell_metrics

def get_begin_end_corr(sim,n_tr=3,day_level=0):
    '''
    for any similarity matrix (df), organized by day-trial, get the mean first and last n_tr correlation across days
    '''
    day_l = sim.columns.get_level_values(day_level).unique()
    tr_l ={'beg':slice(0,n_tr),'end':slice(-n_tr,None)}
    beg_end_corr = {}
    for d1 in day_l:
        for d2 in day_l:
            for k1,tr1 in tr_l.items():
                for k2,tr2 in tr_l.items():        
    #                 corr=sim.loc[(d1,slice(None)),(d2,slice(None))].iloc[:n_tr,-n_tr:].mean().mean()
                    corr=sim.loc[(d1,slice(None)),(d2,slice(None))].iloc[tr1,tr2].mean().mean()
                    beg_end_corr[d1,d2,k1,k2]=corr

    beg_end_corr = pd.Series(beg_end_corr).unstack(level=1).unstack()
    return beg_end_corr

# def get_beg_end_diff_minus_end_begin_same_direct(sim,n_tr=3,day_level=0):
def get_end_beg_diff_minus_beg_end_same_direct(sim,n_tr=3,day_level=0,test='ranksums'):
    '''
    sim: correlation matrix
    get last n_tr trials and beg n_tr trials correlation across sessions - (corr_beg_end_n + corr_beg_end_n+1) within session
    '''
    day_l = sim.columns.get_level_values(day_level).unique()
    tr_l ={'beg':slice(0,n_tr),'end':slice(-n_tr,None)}
    diff_l = []
    end_beg_l = []
    beg_end_mean_l = []
    test_res_l = []
    for i in range(len(day_l)-1):
        end_beg_v = sim.loc[(i,slice(None)),(i+1,slice(None))].iloc[tr_l['end'],tr_l['beg']].values.flatten()
        end_beg = end_beg_v.mean()
        beg_end_i_v = sim.loc[(i,slice(None)),(i,slice(None))].iloc[tr_l['end'],tr_l['beg']].values.flatten()
        beg_end_i = beg_end_i_v.mean()
        beg_end_i_plus_1_v = sim.loc[(i+1,slice(None)),(i+1,slice(None))].iloc[tr_l['end'],tr_l['beg']].values.flatten()
        beg_end_i_plus_1 = beg_end_i_plus_1_v.mean()
        beg_end_mean = (beg_end_i + beg_end_i_plus_1) / 2
        beg_end_same_v=np.concatenate([beg_end_i_v,beg_end_i_plus_1_v])
        
        diff = end_beg - beg_end_mean
        end_beg_l.append(end_beg)
        beg_end_mean_l.append(beg_end_mean)
        diff_l.append(diff)
        if test=='ranksums':
            test_res = scipy.stats.ranksums(end_beg_v,beg_end_same_v)
            test_res_l.append(test_res)
    

    diff_l = np.array(diff_l)
    end_beg_l = np.array(end_beg_l)
    beg_end_mean_l = np.array(beg_end_mean_l)
    if test is not None:        
        test_res_l = pd.DataFrame(test_res_l)
        return diff_l, end_beg_l, beg_end_mean_l, test_res_l
    else:
        return diff_l, end_beg_l, beg_end_mean_l,None

def test_end_beg_diff_minus_beg_end_same_direct_noshuffle(sim,n_tr=3,day_level=0):
    day_l = sim.columns.get_level_values(day_level).unique()
    tr_l ={'beg':slice(0,n_tr),'end':slice(-n_tr,None)}
    diff_l = []
    end_beg_l = []
    beg_end_mean_l = []
    for i in range(len(day_l)-1):
        end_beg = sim.loc[(i,slice(None)),(i+1,slice(None))].iloc[tr_l['end'],tr_l['beg']].values.flatten()
        beg_end_i = sim.loc[(i,slice(None)),(i,slice(None))].iloc[tr_l['end'],tr_l['beg']].values.flatten()
        beg_end_i_plus_1 = sim.loc[(i+1,slice(None)),(i+1,slice(None))].iloc[tr_l['end'],tr_l['beg']].values.flatten()
        beg_end_same=np.concatenate([beg_end_i,beg_end_i_plus_1])
        diff=scipy.stats.ranksums(end_beg,beg_end_same)
        diff_l.append(diff)
    diff_l = pd.DataFrame(diff_l)
    return diff_l
    


import tqdm
def get_end_beg_diff_minus_beg_end_same_all(fr_map_trial_df_all_day_sub,cell_level=0,n_tr = 3,day_level=0,test='ranksums',alpha=0.05,do_bonf=True):
    '''
    loop get_end_beg_diff_minus_beg_end_same_direct across cells
    (uid x day) x [end_beg_across, beg_end_same, diff, across_bigger]
    '''
    gpb = fr_map_trial_df_all_day_sub.groupby(level=cell_level)
    beg_end_corr_all = {}
    diff_l_all = {}
    end_beg_l_all = {}
    beg_end_mean_l_all = {}
    test_res_l_all = {}
    for uid,val in tqdm.tqdm(gpb):

        X_df_ = val.loc[uid]
        sim = X_df_.corr()
        diff_l,end_beg_l,beg_end_mean_l,test_res_l = get_end_beg_diff_minus_beg_end_same_direct(sim,n_tr=n_tr,day_level=day_level,test=test)
        end_beg_l_all[uid] = end_beg_l
        beg_end_mean_l_all[uid] = beg_end_mean_l
        diff_l_all[uid] = diff_l
        test_res_l_all[uid] = test_res_l
    # beg_end_corr_all = pd.concat(beg_end_corr_all,axis=0)
    diff_l_all = pd.DataFrame(diff_l_all).T
    end_beg_l_all = pd.DataFrame(end_beg_l_all).T
    beg_end_mean_l_all = pd.DataFrame(beg_end_mean_l_all).T
    test_res_l_all = pd.concat(test_res_l_all,axis=0)

    beg_end_corr_diff_df=pd.concat({'end_beg_across':end_beg_l_all.stack(),
    'beg_end_same':beg_end_mean_l_all.stack(),'diff':diff_l_all.stack()},axis=1)
    beg_end_corr_diff_df['across_bigger']=beg_end_corr_diff_df['diff'] > 0 
    beg_end_corr_diff_df['statistic'] = test_res_l_all['statistic']
    beg_end_corr_diff_df['pvalue'] = test_res_l_all['pvalue']
    beg_end_corr_diff_df = beg_end_corr_diff_df.dropna(axis=0) # drop rows that contain nan
    if do_bonf:
        alpha = alpha / beg_end_corr_diff_df.index.get_level_values(1).nunique()
    beg_end_corr_diff_df['issig']=beg_end_corr_diff_df['pvalue'] < alpha
    return beg_end_corr_diff_df


def get_end_beg_diff_minus_beg_end_same_all_sessions(fr_map_trial_df_all_day,cell_level=0,n_tr = 3,day_level=0,test='ranksums',alpha=0.05,do_bonf=True):
    '''
    fr_map_trial_df_all_day: (region, exp, isnovel, uid, pos) x (day x trial)
    '''
    gpb = fr_map_trial_df_all_day.groupby(level=(0,1,2))
    beg_end_corr_diff_df_allsess=  {}
    for k,fr_map_trial_df_all_day_sub in gpb:
        fr_map_trial_df_all_day_sub = fr_map_trial_df_all_day_sub.loc[k].dropna(axis=1,how='all')
        beg_end_corr_diff_df = get_end_beg_diff_minus_beg_end_same_all(fr_map_trial_df_all_day_sub,cell_level=cell_level,n_tr = n_tr,do_bonf=do_bonf)
        beg_end_corr_diff_df_allsess[k]=beg_end_corr_diff_df
    beg_end_corr_diff_df_allsess = pd.concat(beg_end_corr_diff_df_allsess,axis=0)
    return beg_end_corr_diff_df_allsess




# def get_beg_end_diff_minus_end_begin_same(beg_end_corr,day_level=0):
# def get_end_beg_diff_minus_beg_end_same(beg_end_corr,day_level=0):
#     day_l = beg_end_corr.index.get_level_values(day_level).unique()
#     diff_l = []
#     end_beg_l = []
#     beg_end_mean_l = []
#     for i in range(len(day_l)-1):
#         end_beg = beg_end_corr.loc[(i,'end'),(i+1,'beg')]
#         beg_end_mean = ((beg_end_corr.loc[(i,'end'),(i,'beg')] + beg_end_corr.loc[(i+1,'end'),(i+1,'beg')])) / 2
#         diff = end_beg - beg_end_mean
#         end_beg_l.append(end_beg)
#         beg_end_mean_l.append(beg_end_mean)
#         diff_l.append(diff)
#     diff_l = np.array(diff_l)
#     end_beg_l = np.array(end_beg_l)
#     beg_end_mean_l = np.array(beg_end_mean_l)
#     return diff_l, end_beg_l, beg_end_mean_l


    

def shuffle_test_end_beg_diff_minus_beg_end_same(sim,n_roll_min=2,nrepeats = 200,n_tr=3):
    diff_l, end_beg_l, beg_end_mean_l=get_end_beg_diff_minus_beg_end_same_direct(sim,n_tr=3,day_level=0)
    # n_roll_min = 2
    diff_l_sh_l = []
    for i in range(nrepeats):
        trial_ind_roll = sim.index.to_frame().groupby(0).apply(lambda x:pd.DataFrame(np.roll(x,np.random.randint(n_roll_min,len(x)-n_roll_min),axis=0)))
        trial_ind_roll = pd.MultiIndex.from_frame(pd.DataFrame(trial_ind_roll))

        sim_reind = sim.loc[trial_ind_roll,trial_ind_roll]
    #     beg_end_corr_sh=gtcm.get_begin_end_corr(sim_reind,n_tr=3,day_level=0)
        diff_l_sh,end_beg_l_sh,beg_end_mean_l_sh = get_end_beg_diff_minus_beg_end_same_direct(sim_reind,n_tr=n_tr,day_level=0)
        diff_l_sh_l.append(diff_l_sh)
    diff_l_sh_l = np.array(diff_l_sh_l)
    pval = (diff_l <= diff_l_sh_l).mean(axis=0)
    return pval,diff_l_sh_l

def shuffle_test_end_beg_diff_minus_beg_end_same_all(fr_map_trial_df_all_day_sub,cell_level=0,n_roll_min=2,nrepeats=200,n_max=5,n_tr=3):
    '''
    fr_map_trial_df_all_day_sub: (uid x npos) x (day x trial)
    circularly shuffle the days of the correlation matrix within each day, to break the across day structure 
    shuffle_test_end_beg_diff_minus_beg_end_same applied to all neurons
    '''
    gpb = fr_map_trial_df_all_day_sub.groupby(level=cell_level)
    pval_d = {}
    diff_l_sh_l_all = {}
    ii=0
    
    for uid,val in tqdm.tqdm(gpb):    
        sim = val.corr()
        pval,diff_l_sh_l = shuffle_test_end_beg_diff_minus_beg_end_same(sim,n_roll_min=n_roll_min,nrepeats = nrepeats,n_tr=n_tr)
        pval_d[uid] = pval
        diff_l_sh_l_all[uid] = pd.DataFrame(diff_l_sh_l)
        ii=ii+1
        if ii>=n_max:
            break
    pval_d = pd.DataFrame(pval_d).T
    
    diff_l_sh_l_all = pd.concat(diff_l_sh_l_all,axis=0) # (cell x nshuffle) x nacrossday
    return pval_d, diff_l_sh_l_all

def end_beg_diff_minus_beg_end_same_show_one_cell(fr_map_trial_df_all_day,frmap_all_day,vmax_quantile=0.99,region='CA1',exp=0,isnovel=0,uid=0,
                                                dosave=False,savedir='',savefn_func = lambda x:f'across_within_diff_{x[0]}_exp{x[1]}_isnovel{x[2]}_cell{x[3]}',
                                                day_level = 0,n_tr=3
                                                ):
    X_df_ = fr_map_trial_df_all_day.loc[(region,exp,isnovel,uid),:].dropna(axis=1,how='all')
    fig,axs=plt.subplots(1,3,figsize=(12,4))
    ax=axs[0]
    fig,ax=ph.heatmap(X_df_.T,ax=ax,vmax_quantile=0.99,fig=fig)
    ph.plot_day_on_heatmap(X_df_.T,ax=ax)
    isnovel_d={0:'Familiar',1:'Novel'}
    title=f'{region} exp {exp} {isnovel_d[isnovel]} \ncell {uid}'
    ax.set(title=title,ylabel='Day-Trial')
    
    sim = X_df_.corr()
    beg_end_corr=get_begin_end_corr(sim,n_tr=n_tr,day_level=day_level)
    ax=axs[1]
    sns.heatmap(beg_end_corr,ax=ax,cmap='vlag')
    ph.plot_day_on_heatmap(beg_end_corr,vline=True,ax=ax)
    ax.set_ylabel('')
    ax.set_xlabel('')
    plt.tight_layout()

    ax=axs[2]
    x_avg=frmap_all_day.loc[(region,exp,isnovel,uid)].unstack(level=0).dropna(axis=1,how='all')
    avg_sim=x_avg.corr()
    sns.heatmap(avg_sim,ax=ax,cmap='vlag')
    # ph.plot_day_on_heatmap(beg_end_corr,vline=True,ax=ax)
    ax.set_ylabel('')
    ax.set_xlabel('Day')
    plt.tight_layout()

    if dosave:
        # savedir =misc.get_or_create_subdir(savedir)      
        savefn = savefn_func([region,exp,isnovel,uid])
        savefn_full = os.path.join(savedir,savefn)
        for fmt in ['svg','png']:
            savefn_full_fmt = savefn_full + f'.{fmt}'
            fig.savefig(savefn_full_fmt,bbox_inches='tight')


def end_beg_diff_minus_beg_end_same_shuffle_show_one_cell(fr_map_trial_df_all_day,
                                n_tr=3,day_level=0,region='CA1',exp=0,isnovel=0,uid=0,
                                dosave=False,savedir='',savefn_func = lambda x:f'across_within_diff_shuffle_{x[0]}_exp{x[1]}_isnovel{x[2]}_cell{x[3]}',
                                n_roll_min = 2,nrepeats = 200):
    X_df_ = fr_map_trial_df_all_day.loc[(region,exp,isnovel,uid),:].dropna(axis=1,how='all')
    sim = X_df_.corr()
    diff_l, end_beg_l, beg_end_mean_l=get_end_beg_diff_minus_beg_end_same_direct(sim,n_tr=n_tr,day_level=day_level)
    pval,diff_l_sh_l = shuffle_test_end_beg_diff_minus_beg_end_same(sim,n_roll_min=n_roll_min,nrepeats = nrepeats)
    nplots=diff_l_sh_l.shape[1]
    fig,axs=plt.subplots(1,nplots,figsize=(6*nplots,4))
    for k in range(nplots):
        ax=axs[k]
        ph.plot_shuffle_data_dist_with_thresh(diff_l_sh_l[:,k],diff_l[k],ax=ax,plot_ci_low=True)
        title=f'{region} exp {exp} isnovel {isnovel}\ncell {uid}\nday {k}'
        ax.set_title(title)
    plt.tight_layout()

    if dosave:
        # savedir =misc.get_or_create_subdir(savedir)      
        savefn = savefn_func([region,exp,isnovel,uid])
        savefn_full = os.path.join(savedir,savefn)
        for fmt in ['svg','png']:
            savefn_full_fmt = savefn_full + f'.{fmt}'
            fig.savefig(savefn_full_fmt,bbox_inches='tight')