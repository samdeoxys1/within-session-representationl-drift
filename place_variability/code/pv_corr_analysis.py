import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import sys,os,copy,pdb,importlib,pickle
from importlib import reload
import pandas as pd
sys.path.append('/mnt/home/szheng/projects/util_code')
import plot_helper as ph

def get_val_per_lag(mat):
    '''
    col: lag
    index: count
    '''
    mat = mat.dropna(axis=1,how='all').dropna(axis=0,how='all')
    mat = mat.values
    ntrials = mat.shape[0]
    vals_per_lag_l = {}
    for lag in range(1,ntrials-1):
        vals_per_lag = np.diag(mat,lag)
        vals_per_lag_l[lag] = pd.Series(vals_per_lag)
    return pd.DataFrame(vals_per_lag_l)

# per_neuron_metrics_all = per_field_metrics_all.groupby(level=(0,1,2,4)).max()
def divide_category_by_quantile(fr_map_trial_df_all,per_neuron_metrics_all,key,q=0.5,thresh=None,do_equal_sample=False,exclude_sess_ma=None):
    '''
    thresh: threshold in terms of value, priority
    q: quantile for defining the high category
    
    key: column for categorizing
    do_equal_sample: sample the same number of samples in the low category as in the high category
    exclude_sess_ma: eg: exclude_sess_ma = fr_map_trial_df_all.index.get_level_values(1)!='e13_26m1_210913'
    '''
    if thresh is None:
        thresh = per_neuron_metrics_all[key].quantile(q)
    high_ma = per_neuron_metrics_all[key] >= thresh
    non_ma = ~high_ma
    if do_equal_sample:
        low_ma = []
        for isnovel in [0,1]:
            n_sw = high_ma.loc[slice(None),slice(None),isnovel].sum()
#             pdb.set_trace()
            to_sample_from = per_neuron_metrics_all.loc[~high_ma].loc[(slice(None),slice(None),isnovel),:]
            if to_sample_from.shape[0] > n_sw:
                replace=False
            else:
                replace = True
            sample_non_sw = to_sample_from.sample(n_sw,replace=replace)
            low_ma_one = per_neuron_metrics_all.index.isin(sample_non_sw.index)
            low_ma.append(low_ma_one)
        low_ma = np.any(low_ma,axis=0)
    else:
        low_ma = non_ma
#     pdb.set_trace()
    if exclude_sess_ma is not None:
        fr_map_trial_df_all_ = fr_map_trial_df_all.loc[exclude_sess_ma].unstack(level=(-1,-3))
    else:
        fr_map_trial_df_all_ = fr_map_trial_df_all.unstack(level=(-1,-3))
    
    hi_ind = per_neuron_metrics_all.index[high_ma].intersection(fr_map_trial_df_all_.index)
    low_ind=per_neuron_metrics_all.index[low_ma].intersection(fr_map_trial_df_all_.index)

    fr_map_trial_df_all_high=fr_map_trial_df_all_.loc[hi_ind].stack(level=(-2,-1)).reorder_levels((0,1,2,5,3,4))
    fr_map_trial_df_all_low=fr_map_trial_df_all_.loc[low_ind].stack(level=(-2,-1)).reorder_levels((0,1,2,5,3,4))

    reorder = lambda df:df.groupby(level=(0,1,2)).apply(lambda x:x.sort_index(level=(3,4,5)).droplevel(level=(0,1,2)))
    fr_map_trial_df_all_high = reorder(fr_map_trial_df_all_high)
    fr_map_trial_df_all_low = reorder(fr_map_trial_df_all_low)
    
    df_d = {'high':fr_map_trial_df_all_high,'low':fr_map_trial_df_all_low}
    ind_d = {'high':hi_ind,'low':low_ind}
    
    return df_d,ind_d
    
    
def get_pv_corr_per_lag_all_category(df_d,pv_level=(1,2,3,4)):
    '''
    
    '''
    pv_corr_per_lag_d={}
    for ksi,df in df_d.items():
        gpb_sess=df.groupby(level=(0,1,2,3))
        pv_corr_alltrials = gpb_sess.corr()
        gpb_corr_sess=pv_corr_alltrials.groupby(level=(0,1,2,3))
        pv_corr_per_lag=gpb_corr_sess.apply(get_val_per_lag)
        pv_corr_per_lag_d[ksi]=pv_corr_per_lag
#     pv_corr_per_lag_d = pd.concat(pv_corr_per_lag_d,axis=0)
    return pv_corr_per_lag_d

def get_pv_corr_per_lag_all_category_multisample_wrapper(fr_map_trial_df_all,per_neuron_metrics_all,key,**kwargs):
    '''
    combine divide_category_by_quantile and get_pv_corr_per_lag_all_category, sample multiple times
    '''
    q=None
    thresh=kwargs.get('thresh',None)
    do_equal_sample=True
    exclude_sess_ma=kwargs.get('exclude_sess_ma',None)
    nrepeats = kwargs.get('nrepeats',5)
    pv_corr_per_lag_d_l = {'high':None,'low':{}}
    for n in range(nrepeats):
        df_d,ind_d = divide_category_by_quantile(fr_map_trial_df_all,per_neuron_metrics_all,key,q=q,thresh=thresh,do_equal_sample=do_equal_sample,exclude_sess_ma=exclude_sess_ma) 
        pv_corr_per_lag_d = get_pv_corr_per_lag_all_category(df_d,pv_level=(1,2,3,4)) # pv_corr_per_lag_d: {'high':pv_corr_per_lag, 'low':}
        pv_corr_per_lag_d_l['high'] = pv_corr_per_lag_d['high']
        # pv_corr_per_lag_d_l['low'].append(pv_corr_per_lag_d['low']) 
        pv_corr_per_lag_d_l['low'][n] = pv_corr_per_lag_d['low'] # 'high' are the same across reps, only low change
    
    pv_corr_per_lag_d_l['low'] = pd.concat(pv_corr_per_lag_d_l['low'],axis=1).groupby(level=1,axis=1).mean()

    return pv_corr_per_lag_d_l



    
def plot_pv_corr_vs_lag_line(pv_corr_per_lag_d,key='',dosave=False,figdir='./',figsize=(2,4),**kwargs):
    xticks = kwargs.get('xticks',[0,10,20,30])
    fig,ax=plt.subplots(figsize=figsize)
    color_l = ['C0','C1']
    linestyle_l={'high':'-','low':':'}
    for ksi,pv_corr_per_lag in pv_corr_per_lag_d.items():
        gpb_task = pv_corr_per_lag.groupby(level=(2))
    #     fig,ax=plt.subplots()
        isfam_d_str = {0:'Familiar',1:'Novel'}
        for k,val in gpb_task:
            c = color_l[k]
            linestyle = linestyle_l[ksi]

            fig,ax=ph.mean_error_plot(val,fig=fig,ax=ax,label=f'{isfam_d_str[k]}, {ksi}',linestyle=linestyle,c=c)
            ax.set(xlabel='Trial lag',ylabel='Pop. ratemap corr')
        ax.legend(bbox_to_anchor=[1.05,1])
    #     ax.set_title(ksi)
    ax.set_ylim([-0.1,1.])
    ax.set_xticks(xticks)
    ax.set_title(key)
    sns.despine()
    figfn = os.path.join(figdir,f'pv_corr_fam_nov_highlow_{key}_vs_triallag')
    if dosave:
        ph.save_given_name(fig,figfn,figdir=figdir)

#         for fmt in ['png','svg']:
#             figfn = os.path.join(figdir,f'pv_corr_fam_nov_highlow_{key}_vs_triallag.{fmt}')
#             fig.savefig(figfn,bbox_inches='tight')
            
    return fig,ax

from scipy.stats import linregress
def get_decay_slope_linregress(df):
    dfm=df.melt().dropna(axis=0)
    slope=linregress(dfm.iloc[:,0],dfm.iloc[:,1])[0]
    return slope

def get_decay_slope_per_sess(pv_corr_per_lag_d,kind='corr'):
    '''
    kind: corr: corr between trial and pvcorr
          slope: (last - first) / ntrials
    '''
    slope_per_sess_d = {}
    for k,val in pv_corr_per_lag_d.items():
        if kind=='corr':
            slope_per_sess=val.groupby(level=(0,1,2)).apply(lambda df:df.melt().corr().iloc[0,1])
        elif kind=='linregress':
            slope_per_sess=val.groupby(level=(0,1,2)).apply(get_decay_slope_linregress)
        slope_per_sess_d[k] = slope_per_sess
    slope_per_sess_d = pd.concat(slope_per_sess_d,axis=0)
#     slope_per_sess_d=slope_per_sess_d.reset_index(level=(0,3)).rename({'level_0':'category','level_3':'isnovel',0:'slope'},axis=1)
    slope_per_sess_d=slope_per_sess_d.reset_index(level=(0)).rename({'level_0':'category',0:'slope'},axis=1)
    return slope_per_sess_d

reload(ph)
def plot_slope_per_sess(slope_per_sess_d,fig=None,axs=None,dosave=False,key='',figdir='./',figsize=(4,3),xticklabel_map={'high':'Has switch','low':'No switch'}):
    '''
    slope_per_sess_d = get_decay_slope_per_sess(pv_corr_per_lag_d)
    '''
    if axs is None:
        fig,axs=plt.subplots(1,2,figsize=figsize,sharey=True)
    ylabel = 'Slope'
    for ii,isnovel in enumerate([0,1]):
        isnovel_str = ['Familiar','Novel'][isnovel]
        ax=axs[ii]
        df=slope_per_sess_d.loc[(slice(None),slice(None),isnovel),:]
        df=df.set_index('category',append=True).unstack('category')['slope']
        df.columns.name=''
        fig,ax=ph.paired_line_with_box(df,'high','low',fig=fig,ax=ax,dotest=True)
        ax.set_title(isnovel_str,pad=25.)
        ax.set_ylabel(ylabel)
        if xticklabel_map is not None:
            xticklabels=[xticklabel_map['high'],xticklabel_map['low']]
            ax.set_xticklabels(xticklabels,rotation=45)
    plt.tight_layout()
    if dosave:
        figfn =f'pvcorr_decay_slope_vs_{key}'
        ph.save_given_name(fig,figfn,figdir=figdir)
    return fig,axs

from matplotlib.ticker import MaxNLocator,LinearLocator
def visualize_example_contribution(contribution_one_cell,fr_map_trial,
                                   sess='',ti=0,tt=0,uid=None,dosave=False,
                                   fig=None,axs=None,
                                   figdir='./'):
    '''
    plot contribution to pvcorr vs trial_lag, ratemap, lap-lap corr vs trial_lag
    fr_map_trial: npos x ntrial
    '''
    if axs is None:
        fig,axs=plt.subplots(1,3,figsize=(9,3))
    mat = fr_map_trial.dropna(axis=1).T
    per_neuron_ratemap_corr=get_val_per_lag(mat.T.corr())
    triallag_xticks = [10,20]
    ax=axs[0]
    ph.mean_error_plot(contribution_one_cell,ax=ax,fig=fig)
    sns.despine(ax=ax)
    ax.set_xlabel('Trial lag')

    ax.set_ylabel('Contribution to\n PV correlation')
    ax.set_xticks(triallag_xticks)

    ax=axs[1]
    ph.heatmap(mat,ax=ax,fig=fig,ylabel='Trial')
    ax.set_title(f'{sess}\n {ti},{tt}, uid {uid}')
    ax.set_xticks([])
    # ax.set_yticks([])
    ax.yaxis.set_major_locator(LinearLocator(4))

    ax=axs[2]
    ph.mean_error_plot(per_neuron_ratemap_corr,ax=ax,fig=fig)
    ax.set_ylabel('Ratemap\ncorrelation')
    sns.despine(ax=ax)
    ax.set_xticks(triallag_xticks)
    plt.tight_layout()
    
    if dosave:
        figfn = f'contribution_to_corr_{sess}_{ti}_{tt}_{uid}'
        figdir = os.path.join(figdir,'ex')
        figdir = misc.get_or_create_subdir(figdir)
        ph.save_given_name(fig,figfn,figdir)
    
    return fig,axs

# get contribution per cell to the correlation

def get_contribution_per_cell_per_tr_pair_onesess(val):
    '''
    sess means ti,tt pair
    '''
    val = val.dropna(axis=1)
    val_z = scipy.stats.zscore(val,axis=0)
    contribution_per_cell_per_tr_pair = {}
    tr_lag_max = val_z.columns.max() - val_z.columns.min()
    for tr_lag in range(1,tr_lag_max+1):
        for tr_st in range(0,val_z.columns.max()-tr_lag+1): 
            if tr_st in val_z.columns and (tr_st + tr_lag) in val_z.columns:
                z_prod_tr_pair = val_z[tr_st] * val_z[tr_st + tr_lag]
            contribution_per_cell=z_prod_tr_pair.unstack().sum(axis=1) / len(z_prod_tr_pair)
            contribution_per_cell_per_tr_pair[tr_lag,tr_st] = contribution_per_cell
    contribution_per_cell_per_tr_pair = pd.concat(contribution_per_cell_per_tr_pair,axis=1)
    return contribution_per_cell_per_tr_pair

def get_slope_intercept_per_neuron(df):
    '''
    df: df, n neurons x (trial lag, start trial)
    '''
    sl_all = {}
    intercept_all = {}
    # for uid,row in contribution_per_cell_per_lag.iterrows():
    for uid,row in df.iterrows():
    #     reg_res = scipy.stats.linregress(row.index,row.values)
        row = row.dropna(axis=0)
        reg_res = scipy.stats.linregress(row.index.get_level_values(0),row.values)
        sl = reg_res[0]
        intercept = reg_res[1]
        sl_all[uid] = sl
        intercept_all[uid] = intercept
    sl_all = pd.Series(sl_all)
    intercept_all = pd.Series(intercept_all)
    sl_intercept_all = pd.concat({'slope':sl_all,'intercept':intercept_all},axis=1)
    
    return sl_intercept_all

from scipy.spatial.distance import pdist,squareform
from sklearn.metrics.pairwise import nan_euclidean_distances
def dist_one_df(x):
    x=x.dropna(axis=1,how='all')
#     mat=squareform(pdist(x.values.T,metric='sqeuclidean') / x.shape[0])
    mat = nan_euclidean_distances(x.values.T,x.values.T,squared=True) / x.shape[0]
    dist_df=pd.DataFrame(mat,index=x.columns,columns=x.columns)
    return dist_df

def get_plot_pf_param_population_corr_vs_lag(pf_params_all,ma_ind_new=None,pf_key='fr_peak',fig=None,ax=None,center_cell=None,zscore_cell=False,dist='corr'):
    
    df=pf_params_all.loc[(slice(None),slice(None),slice(None),slice(None),pf_key),:].droplevel(4)
    if center_cell is None:
        center_cell = False
        if pf_key=='peak' or pf_key=='com': # by default, no center; if location, then center
            center_cell=True
    if ma_ind_new is not None:
        pf_key_sub=df.reset_index(level=-1).loc[ma_ind_new].set_index('level_5',append='True')
    else:
        pf_key_sub = df
    if center_cell:
        pf_key_sub = pf_key_sub - pf_key_sub.mean(axis=1).values[:,None]
    if zscore_cell:
        pf_key_sub = scipy.stats.zscore(pf_key_sub,axis=1,nan_policy='omit')
#     pdb.set_trace()
    if dist =='corr':
        pf_key_corr_allsess = pf_key_sub.groupby(level=(0,1,2)).corr()
    elif dist=='mse':
        
        pf_key_corr_allsess = pf_key_sub.groupby(level=(0,1,2)).apply(dist_one_df)
#     pdb.set_trace()
    gpb_corr_sess=pf_key_corr_allsess.groupby(level=(0,1,2))
    pv_corr_per_lag=gpb_corr_sess.apply(get_val_per_lag)
    gpb_task=pv_corr_per_lag.groupby(level=(2))
    if ax is None:
        fig,ax=plt.subplots()
    isfam_d_str = {0:'Familiar',1:'Novel'}
    for k,val in gpb_task:
        fig,ax=ph.mean_error_plot(val,fig=fig,ax=ax,label=isfam_d_str[k],c=f'C{k}')
        ax.set(xlabel='Trial lag',ylabel=f'Pop. within-field {pf_key} {dist}')
        
    ax.legend()
    return fig,ax, pf_key_corr_allsess


# schematics
def gen_sim_plot(loc1 = 20,
    loc2 = 40,
    width=5,
    jitter = 0.,
    modulation = 1.,fig=None,ax=None,fs=15):
    xs=np.arange(100)
    r1=scipy.stats.norm(loc=loc1,scale=width).pdf(xs)
    r2=scipy.stats.norm(loc=loc2,scale=width).pdf(xs)
    r = np.concatenate([r1,r2])
    r3 = scipy.stats.norm(loc=loc1+jitter,scale=width).pdf(xs) * modulation
    r4 = scipy.stats.norm(loc=loc2+jitter,scale=width).pdf(xs) 
    if modulation!=1:
        r4 = scipy.stats.norm(loc=loc2+jitter,scale=width).pdf(xs) * 1.1
    r_next = np.concatenate([r3,r4])
    corr=np.corrcoef(r,r_next)
    
    if ax is None:
        fig,ax=plt.subplots()
    ax.plot(r,label='current trial',c='k')
    ax.plot(r_next,label='next trial',c='r')
    ax.set_title(f'r={corr[0,1]:.2f}')
    ax.set_xticks([len(xs)])
    ax.set_xticklabels([])
    ax.set_xlabel('Position (concatenated)')
#     ax.set_xticks([loc1,loc2 + len(xs)])
#     ax.set_xticklabels(['Neuron 1','Neuron 2'])
    height = np.maximum(r.max(),r_next.max()) * 1.05
    ax.text(loc1,height,'Neuron 1',horizontalalignment='center',fontsize=fs)
    ax.text(loc2+len(xs),height,'Neuron 2',horizontalalignment='center',fontsize=fs)
    ax.set_ylim([0,0.1])
    ax.legend(bbox_to_anchor=[1.25,0.9],fontsize=fs)
    ax.set_yticks([])
    
    sns.despine(left=True)
    
    return fig,ax
    
    