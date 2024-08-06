import ruptures as rpt
import numpy as np
import pandas as pd
import scipy
import os,sys,copy,itertools,pdb,importlib

import statsmodels
import statsmodels.api as sm
sys.path.append('/mnt/home/szheng/projects/nmf_analysis')
sys.path.append('/mnt/home/szheng/projects/place_variability/code')
sys.path.append('/mnt/home/szheng/projects/util_code')
import change_point_analysis as cpa
import plot_all_fr_map_x_pwc_one_session as plotfm
importlib.reload(plotfm)
import plot_helper as ph
import matplotlib.pyplot as plt

def fit_cpd_get_r2(xx,ncpts,cost='l2',min_size=2):
    
    xx_pwc,cpt=cpa.predict_from_cpts_wrapper(xx,ncpts,cost=cost,min_size=min_size)
    r2 = 1-np.var(xx - xx_pwc) / np.var(xx)
    return r2,xx_pwc

def fit_poly_regress_get_r2(xx,order=1,cost='l2',verbose=True):
    if cost=='l2':
        xs_l = []
        for o in range(1,order+1):
            xs = np.arange(len(xx)) ** o 
            xs_l.append(xs)
        xs_l = np.array(xs_l).T
#         pdb.set_trace()
        xs_l = sm.add_constant(xs_l)
        model = sm.OLS(xx,xs_l)
        results = model.fit()
        if verbose:
            print(results.summary())
        xx_pred=results.predict()
        r2 = results.rsquared
        return r2,xx_pred
            
    else:
        print('not implemented')
        pass
    
def fit_poly_regress_cpd_get_r2_all(X_raw, order=2,cost='l2',min_size=2):
    r2_reg_l = {}
    r2_cpd_l = {}
    for i,row in X_raw.iterrows():
        xx = row.dropna().values
        try:
            r2_reg,xx_pred=fit_poly_regress_get_r2(xx,order=order,verbose=False)
            r2_cpd,xx_pwc = fit_cpd_get_r2(xx,order,cost=cost,min_size=min_size)
            r2_reg_l[i] = r2_reg
            r2_cpd_l[i] = r2_cpd
        except:
            pass
    r2_reg_l = pd.Series(r2_reg_l)
    r2_cpd_l = pd.Series(r2_cpd_l)
    try:
        r2_df=pd.concat({'reg':r2_reg_l,'step':r2_cpd_l},axis=1)
        r2_df['step_minus_reg'] = r2_df['step'] - r2_df['reg']
    except:
        return None
    return r2_df
    
def fit_poly_regress_cpd_get_r2_all_multi_order(X_raw_one,cost='l2',min_size=2,ncpt_max=5):
    ntrials = X_raw_one.dropna(axis=1,how='all').shape[1]
    n_change_pts_max = np.minimum(int(ntrials // 4),ncpt_max) # 4 here is kinda arbitrary
    r2_df_d = {}
    for ncpt in range(1,n_change_pts_max+1):
        r2_df = fit_poly_regress_cpd_get_r2_all(X_raw_one, order=ncpt,cost='l2',min_size=2)
        r2_df_d[ncpt] = r2_df
    if r2_df_d !={}:
        r2_df_d = pd.concat(r2_df_d,axis=1)
    else:
        r2_df_d = None
    return r2_df_d
        
    



from matplotlib.ticker import MaxNLocator
def fit_plot_step_vs_continuous_example(fr_map_trial_df,
                                        X_raw,X_pwc,all_sw_d,trial_index_to_index_within_df,
                                        all_fields_recombined,
                                        best_n_one,
                                        ncpts=1,ii=0,
                                        ta=0,tt=0,min_size=2,
                                        uid=None,
                                        field_id=None,
                                        sess='',
                                        dosave=False,figdir='./',figfn = None,
                                        figsize=(6,2),do_legend=False,
                                        fig=None,
                                        axs=None

                                       ):
    # select
    if (uid is None) or (field_id is None):
        ma=best_n_one==ncpts
        uid,field_id = best_n_one.index[ma][ii]
#         print(uid,field_id)
    all_fields_row_one=all_fields_recombined.loc[ta,tt,uid,field_id]
    
    # fit
    xx = X_raw.loc[(ta,tt,uid,field_id),:].dropna().values
    r2_reg,xx_pred_reg=fit_poly_regress_get_r2(xx,order=ncpts,cost='l2',verbose=False)
    r2_step,xx_pred_step=fit_cpd_get_r2(xx,ncpts,cost='l2',min_size=2)
    
    
    
    # plot
    fig,axs=plotfm.plot_ratemap_fr_one_field(all_fields_row_one,fr_map_trial_df,X_raw,X_pwc,all_sw_d,
                            trial_index_to_index_within_df,
                            save_fig_fn = None,
                            close_fig = False,
                            vmax_clip_quantile=0.99,
                            vmax_relative_to_field=False,
                            fig=fig,axs=axs,figsize=figsize,do_legend=do_legend,
                            )

    ax=axs[0]
    ax.clear()
    fig,ax=plotfm.plot_x_raw_and_pwc(xx,xx_pred_step,fig=fig,ax=ax,do_legend=do_legend)
    ax.plot(xx_pred_reg,np.arange(len(xx_pred_reg)),linewidth = 3,linestyle='--',c='orange')
    if do_legend:
        ax.legend(['raw','fitted step','fitted cont.'],frameon=False,bbox_to_anchor=[-0.2,0.6])
    # title=f'R2 continuous: {r2_reg:.2f}\nR2 step: {r2_step:.2f}'
    title=f'R2 regression: {r2_reg:.2f}\nR2 CPM: {r2_step:.2f}'
    ax.set_title(title,pad=0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=2,integer=True))

    axs[1].set_xticks([])
    plt.tight_layout()

    if dosave:
        if figfn is None:
            figfn = f'cpd_vs_polyregress_ex_{sess}_{ta}_{tt}_{uid}_{field_id}'
            ph.save_given_name(fig,figfn,figdir)

    return fig,axs

# analysis of how many trials it take from valley to peak across a switch 

def find_peaks(signal):
    peak_l = []
    for i,s in enumerate(signal):
        if i==0 and signal[i+1] >s:
            peak_l.append(i)
        elif i>0 and i<(len(signal)-1):
            next_minus_curr = signal[i+1]-signal[i]
            curr_minus_pre = signal[i]-signal[i-1]
            if (next_minus_curr <= 0) and (curr_minus_pre >= 0) and (np.abs(next_minus_curr) +np.abs(curr_minus_pre))!=0: # the differences on two sides can't be both zero
                peak_l.append(i)
            
        elif i==(len(signal)-1) and signal[i-1]<s:
            peak_l.append(i)
    peak_l = np.array(peak_l)
    
    return peak_l
    
    

def get_peri_switch_x_raw(changes_df_row,X_raw_row,win=5):
    row = changes_df_row.dropna().values
    xx=X_raw_row.dropna()
    sw_ind_within_l=np.nonzero(row!=0)[0]
    peri_sw_x={}
    for i,swiw in enumerate(sw_ind_within_l):
        st = swiw - win
        ed = swiw + win
        if i > 0:
            prev_sw = sw_ind_within_l[i-1]
        else:
            prev_sw = 0

        if i < (len(sw_ind_within_l)-1):
            next_sw = sw_ind_within_l[i+1]
        else:
            next_sw = len(xx)
        # pick the lastest starting point among 0, start_window based on current change point, previous change point 
        st_final = np.max([0,st,prev_sw])
        # similarly for final, but reverse
        ed_final = np.min([ed,next_sw,len(xx)])

        peri_sw_x[i]=(xx.iloc[st_final:ed_final])
    peri_sw_x = pd.concat(peri_sw_x,axis=0).unstack()

    return peri_sw_x


def get_peri_switch_peak_valley_span(changes_df_row,X_raw_row, X_pwc_row):
    '''
    for each switch, look at the surrounding trials, for ON, at which trial is the post above the post mean FR for the first time, and which trial is the pre below the pre mean FR for the first time
    and the distance between the two
    vice versa for OFF
    '''
    # peaks=find_peaks(X_raw_row.values)
    # valleys=find_peaks(-X_raw_row.values)
    row = changes_df_row.dropna().values
    
    swiw_l=np.nonzero(row!=0)[0]
    span_d={}
    for i,swiw in enumerate(swiw_l): # for ON, choose previous valley and next peak
        if row[swiw]==1:
            
            ma_pre = (X_raw_row.values <= X_pwc_row.values)
            
            ma_post = (X_raw_row.values >= X_pwc_row.values)
            

        elif row[swiw]==-1:
            ma_pre = (X_raw_row.values >= X_pwc_row.values)
            ma_post = (X_raw_row.values <= X_pwc_row.values)

        # pdb.set_trace()
        ma_pre_nonzero=np.nonzero(ma_pre)[0]
        n_trial_between_sw_pre = swiw-1 - ma_pre_nonzero 
        ma_post_nonzero=np.nonzero(ma_post)[0]
        n_trial_between_sw_post =  ma_post_nonzero - swiw 
        if (n_trial_between_sw_pre>=0).sum()>0 and (n_trial_between_sw_post>=0).sum()>0: # drop weird cases

            pre_n_trial = np.min(n_trial_between_sw_pre[n_trial_between_sw_pre>=0]) # only looking at trial before the switch, pre_n should >=0; since swiw-1  >= pre indices 
            # how many trials before one threshold crossing; if right away then 0
            # swiw: the index when switch happen; max of nonzero ma should at most be swiw-1; when that happen pre_n_trial=0
            
            post_n_trial = np.min(n_trial_between_sw_post[n_trial_between_sw_post>=0]) # only looking at trial after the switch, post_n should >=0, 
   
            span = pre_n_trial + post_n_trial 
            onoff=row[swiw]
            
            span_d[i]={'trial span':span,'onoff':onoff,'pre_n_trial':pre_n_trial, 'post_n_trial':post_n_trial}
    span_d = pd.DataFrame(span_d)
    return span_d

# [OLD] local extrema
# def get_peri_switch_peak_valley_span(changes_df_row,X_raw_row):
#     '''
#     for each switch, look at the surrounding trials, the distance between the previous valley to the next peak for ON
#     vice versa for OFF
#     '''
#     peaks=find_peaks(X_raw_row.values)
#     valleys=find_peaks(-X_raw_row.values)
#     row = changes_df_row.dropna().values
    
#     swiw_l=np.nonzero(row!=0)[0]
#     span_d={}
#     for i,swiw in enumerate(swiw_l): # for ON, choose previous valley and next peak
#         if row[swiw]==1:
#             pre_to_select=valleys[valleys < swiw]
#             post_to_select=peaks[peaks >= swiw] # sw on can only be a peak? sometimes a valley 
#                                                 # (i.e. when the jump is in last trial but the min_win requirement makes it two trials
#                                                 # not good but for now ok
#         elif row[swiw]==-1:
#             pre_to_select=peaks[peaks < swiw] # similar to above concern
#             post_to_select=valleys[valleys >= swiw] # sw off can only be a valley?
        
#         if len(pre_to_select) > 0:
#             pre = np.max(pre_to_select)
#         else: # sth is wrong, mostly situation like mismatch between the detected change and the raw change, just drop it
#             return pd.DataFrame([])
        
#         if len(post_to_select) > 0:
#             post = np.min(post_to_select)
#         else:
# #             post = len(row)-1
#             return pd.DataFrame([])

#         dist = post - pre
#         onoff=row[swiw]
        
#         pre_fr=X_raw_row.iloc[pre]
#         post_fr=X_raw_row.iloc[post]
        
#         span_d[i]={'trial span':dist,'onoff':onoff,'pre_fr':pre_fr,'post_fr':post_fr,'post_minus_pre':post_fr-pre_fr}
#     span_d = pd.DataFrame(span_d)
#     return span_d

# apply above func to all rows
def get_peri_switch_peak_valley_span_all(X_raw_one,changes_df_one,X_pwc_one=None):
    span_d_one = {}
    for kk,X_raw_row in X_raw_one.iterrows():
        changes_df_row = changes_df_one.loc[kk].dropna()
        X_raw_row = X_raw_row.dropna()
        if X_pwc_one is not None:
            X_pwc_row = X_pwc_one.loc[kk].dropna()
            span_d = get_peri_switch_peak_valley_span(changes_df_row,X_raw_row,X_pwc_row)
        else:
            span_d = get_peri_switch_peak_valley_span(changes_df_row,X_raw_row)
        span_d_one[kk] = span_d
#     span_d_one = pd.concat(span_d_one).stack().unstack(level=2) # uid,field_id, sw_id
    span_d_one = pd.concat(span_d_one).stack()
    span_d_one = span_d_one.unstack(level=-2) # turn onoff and trialspan into columns
    nlevels=span_d_one.index.nlevels
    span_d_one = span_d_one.set_index('onoff',append=True).reorder_levels((-1,*np.arange(nlevels))) # put the onoff level to first    

    return span_d_one
    
