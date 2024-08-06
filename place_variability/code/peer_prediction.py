import numpy as np
import scipy
import matplotlib.pyplot as plt
import sys,os,copy,pdb,importlib
from importlib import reload
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import plot_helper as ph
import seaborn as sns
from sklearn.linear_model import LassoCV
import data_prep_pyn as dpp

def fit_glm(spk_beh_df_onetask,uid_endog,cell_cols_exog,trials_to_exclude=None,
                    verbose=False,model_type="linear",alpha=0.,
                        do_inh_only = True,
                        do_exc_only = False,
                        do_weighted_pred=False,
                            ):
    if trials_to_exclude is not None:
        ma=np.logical_not(spk_beh_df_onetask['trial'].isin(trials_to_exclude))
        spk_beh_df_onetask_trialexcluded=spk_beh_df_onetask.loc[ma]
    else:
        spk_beh_df_onetask_trialexcluded = spk_beh_df_onetask

    uid = uid_endog
    y=spk_beh_df_onetask_trialexcluded[uid].values
    X = spk_beh_df_onetask_trialexcluded[cell_cols_exog]
    x_cols = X.columns
    X = X.values
    X = sm.add_constant(X)

    if model_type =="poisson":
        glm_res=sm.genmod.GLM(y,X,family=sm.families.Poisson()).fit()   
    elif model_type == "linear":
        glm_res = sm.OLS(y,X).fit()
    elif model_type=="lasso":
        if isinstance(alpha,list): # doing cv
            reg=LassoCV(cv=5,alphas=alpha,random_state=0).fit(X,y)
            alpha_ = reg.alpha_
            print('lasso: ',alpha_)
        else:
            alpha_ = alpha
        glm_res = sm.OLS(y,X).fit_regularized(alpha=alpha_)
    if verbose:
        print(glm_res.summary())

    # glm_res_df=pd.DataFrame([glm_res.params,glm_res.pvalues],index=['coef','p'])
    glm_res_df=pd.DataFrame([glm_res.params],index=['coef'])
    glm_res_df = glm_res_df.iloc[:,1:]
    glm_res_df.columns=x_cols
    glm_res_df = glm_res_df.T
    glm_res_df['t'] = glm_res.tvalues[1:]
    glm_res_df['ci_low'] = glm_res.conf_int()[1:,0]
    glm_res_df['ci_high'] = glm_res.conf_int()[1:,1]
    try:
        glm_res_df['p'] = glm_res.pvalues[1:]
    except:
        pass
    
    return glm_res,glm_res_df

def fit_glm_predict_rate_change(spk_beh_df_onetask,uid_endog,cell_cols_exog,trials_to_exclude,
                        fr_map_trial_df_exog,all_fields_one,
                        verbose=False,model_type="poisson",alpha=0.,
                        p_thresh=None,
                        do_inh_only = False,
                        do_exc_only = False,
                        do_weighted_pred=True,
                        ):
    '''
    spk_beh_df_onetask: spk_beh_df after selecting the task_index and trial_type
    uid_endog: column to predict
    cell_cols_exog: the predictor columns
    trials_to_exclude: which (switch) trials to exclude in fitting 
    p_thresh: if a value not None, use it to filter significant coefficients for prediction
    do_inh_only: if True, only look at the predictor neurons with negative weights for prediction
    do_weighted_pred: if True, use the fitted regression weights for prediction, as opposed to averaging everything
    '''
    
    glm_res,glm_res_df = fit_glm(spk_beh_df_onetask,uid_endog,cell_cols_exog,trials_to_exclude=trials_to_exclude,
                    verbose=verbose,model_type=model_type,alpha=alpha,
                        do_inh_only = do_inh_only,
                        do_exc_only = do_exc_only,
                        do_weighted_pred=do_weighted_pred,
                            )

    # predict ratemap from model
    fr_map_one_tt_predictor = fr_map_trial_df_exog
    fr_map_one_tt_predictor_tr_pos_stacked=fr_map_one_tt_predictor.unstack(level=1)

    if do_weighted_pred:
        weights = glm_res_df['coef'].values
    else:
        weights = np.ones(glm_res_df.shape[0]) / glm_res_df.shape[0]
    
    if p_thresh is not None:
        if 'p' in glm_res_df.columns:
            weights[glm_res_df['p'] > p_thresh] = 0.
    if do_inh_only:
        weights[glm_res_df['coef'] >0] = 0.
    if do_exc_only:
        weights[glm_res_df['coef'] <0] = 0.
    if not do_weighted_pred: # renormalize the averaging
        eps = 1e-10
        weights = weights / np.sum(weights + eps)

    pred_val = weights.dot(fr_map_one_tt_predictor_tr_pos_stacked)
    pred_val = pd.Series(pred_val,index=fr_map_one_tt_predictor_tr_pos_stacked.columns)
    pred_val=pred_val.unstack()
    
    st,ed=all_fields_one[['start','end']].values
    mean_within_field_pred = pred_val.loc[:,st:ed].mean(axis=1).dropna()

    return glm_res, glm_res_df, pred_val, mean_within_field_pred

from matplotlib.ticker import MaxNLocator
# def post_fit_plot(mean_within_field_pred,
#             all_fields,fr_map_trial_df_all,
#             fr_map_one_tt_predictor,X_pwc,X_raw,
#             ti,tt,uid,field_id,do_weighted_pred=True,
#             inh_coef_ma = None,
#             ):
def post_fit_plot(mean_within_field_pred,
            all_fields,fr_map_trial_df_pyr_combined,
            fr_map_one_tt_predictor,X_pwc,X_raw,
            ti,tt,uid,field_id,do_weighted_pred=True,
            inh_coef_ma = None,
            ):
    # st,ed=field_bounds
    st,ed=all_fields.loc[ti,tt,uid,field_id][['start','end']].values
    # mean_within_field_pred = pred_val.loc[:,st:ed].mean(axis=1)
    fig,axs=plt.subplots(3,1,figsize=(8,12))
    ax=axs[0]
    # if tt!='both':
    #     fr_map_trial_one = fr_map_trial_df_all.loc[ti,tt,uid].T
    # else:
    #     fr_map_trial_one = fr_map_trial_df_all.loc[ti,uid].T
    fr_map_trial_one = fr_map_trial_df_pyr_combined.loc[(ti,tt,uid),:].T
    # field_bound=all_fields_recombined_all.loc[(ani,sess,ti,tt,uid,field_id),('start','end')]
    field_bound=(st,ed)
    fr_map_trial_one = fr_map_trial_one.dropna(axis=0,how='all')
    ph.ratemap_one_raw(fr_map_trial_one,trial=None,field_bound=field_bound,fig=fig,ax=ax,line_kws={},title=None,heatmap_kws={})

    ax=axs[1]
    if do_weighted_pred:
        label = 'predicted fr change'
        ylabel='Predicted FR Change\nusing Int (a.u.)'
    else:
        label = 'interneuron'
        ylabel='Mean selected Int. FR'
    if mean_within_field_pred is not None:
        mean_within_field_pred.plot(ax=ax,label=label,linewidth=3,marker='o')
    ax2=ax.twinx()
    ax2.plot(X_pwc.loc[ti,tt,uid,field_id].dropna().values,c='grey',label='raw',linewidth=3)
    ax2.plot(X_raw.loc[ti,tt,uid,field_id].dropna().values,c='black',label='fit',marker='o',linewidth=3)
    sns.despine(top=True,right=False,ax=ax)
    sns.despine(top=True,right=False,ax=ax2)
    ax.legend()
    ax2.legend(bbox_to_anchor=[1.3,1])
    ax.set(ylabel=ylabel,xlabel='Trial')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set(ylabel='Pyr FR (Hz)')
    ax.set_title(f'cell {uid} field {field_id}')

    ax=axs[2]
    sns.despine(ax=ax)
    if inh_coef_ma is None:
        fr_map_one_tt_predictor.groupby(level=1).mean().loc[st:ed].mean(axis=0).plot(ax=ax,marker='o')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set(xlabel='Trial',ylabel='Mean Int. FR')
    else:
        fr_map_one_tt_predictor.loc[inh_coef_ma.loc[inh_coef_ma].index].groupby(level=1).mean().loc[st:ed].mean(axis=0).plot(ax=ax,label='neg coef',marker='o')
        fr_map_one_tt_predictor.loc[inh_coef_ma.loc[~inh_coef_ma].index].groupby(level=1).mean().loc[st:ed].mean(axis=0).plot(ax=ax,label='the rest',marker='o')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set(xlabel='Trial',ylabel='Mean FR')
        ax.legend()


    plt.tight_layout()
    return fig,axs


# def fit_glm_predict_rate_change_wrapper(spk_beh_df,ti,tt,uid,field_id,cell_cols_d,
#                                     fr_map_trial_df_bothtt_int,
#                                     fr_map_trial_df_all_int,
#                                     all_sw_with_metrics,all_fields_one_sess,
#                                     **kwargs
#                                     ):

def fit_glm_predict_rate_change_wrapper(spk_beh_df,ti,tt,uid,field_id,cell_cols_d,
                                    fr_map_trial_df_int_combined,
                                    all_sw_with_metrics,all_fields_one_sess,
                                    **kwargs
                                    ):

    if tt=='both':
        spk_beh_df_onetask = spk_beh_df.loc[spk_beh_df['task_index']==ti]
    else:
        spk_beh_df_onetask = spk_beh_df.loc[spk_beh_df['trial_type']==(ti,tt)]
    uid_endog  = uid
    cell_cols_exog = cell_cols_d['int']

    # if tt=='both':
    #     fr_map_trial_df_exog = fr_map_trial_df_bothtt_int.loc[ti]
    # else:
    #     fr_map_trial_df_exog = fr_map_trial_df_all_int.loc[(ti,tt),:]
    
    fr_map_trial_df_exog = fr_map_trial_df_int_combined.loc[(ti,tt),:]

    if all_sw_with_metrics is not None:
        all_sw_one_field = all_sw_with_metrics.loc[ti].query('trialtype==@tt&uid==@uid')
        trials_to_exclude = all_sw_one_field['trial_index']
    else:
        trials_to_exclude = []


    all_fields_one = all_fields_one_sess.loc[(ti,tt,uid,field_id),:]
    # all_fields_one = all_fields_recombined_all.loc[(ani,sess,ti,tt,uid,field_id),:]
    # glm_res, glm_res_df, pred_val, mean_within_field_pred = pp.fit_glm_predict_rate_change(spk_beh_df_onetask,uid_endog,cell_cols_exog,trials_to_exclude,
    #                         fr_map_trial_df_exog,all_fields_one,model_type="linear",
    #                         verbose=False)
    alpha_l = [0.01,0.001,0.0001,0.00001,0.000001]
    p_thresh = kwargs.get('p_thresh',0.05)
    model_type = kwargs.get('model_type',"linear")
    do_weighted_pred = kwargs.get('do_weighted_pred',False)
    do_inh_only = kwargs.get('do_inh_only',True)#True
    do_exc_only = kwargs.get('do_exc_only',False)
    # pdb.set_trace()
    glm_res, glm_res_df, pred_val, mean_within_field_pred = fit_glm_predict_rate_change(spk_beh_df_onetask,uid_endog,cell_cols_exog,trials_to_exclude,
                            fr_map_trial_df_exog,all_fields_one,model_type=model_type,alpha=alpha_l,
                            p_thresh=p_thresh,
                            do_inh_only = do_inh_only, do_exc_only=do_exc_only,                                                                                       
                            do_weighted_pred=do_weighted_pred,
                            verbose=False)
    
    return glm_res, glm_res_df, pred_val, mean_within_field_pred

def get_min_max_tick(X_raw_one_row,trial_index_to_index_within_df,ti,tt):
    X_diff = X_raw_one_row.dropna().diff()
    max_tick_tr=X_diff.idxmax()
    min_tick_tr = X_diff.idxmin()
    if tt=='both':
        trials_to_exclude = [max_tick_tr,min_tick_tr]
    else:
        trials_to_exclude = [trial_index_to_index_within_df.loc[ti,tt,max_tick_tr],trial_index_to_index_within_df.loc[ti,tt,min_tick_tr]]
    return np.array(trials_to_exclude)

def get_trials_to_exclude_onefield(all_sw_with_metrics,X_raw_one,uid,field_id,ti,tt,trial_index_to_index_within_df
                        ):
    all_sw_one_field = all_sw_with_metrics.loc[ti].query('trialtype==@tt&uid==@uid')
    if all_sw_one_field.shape[0] > 0:
        trials_to_exclude = all_sw_one_field['trial_index']
    else:
        X_raw_one_row=X_raw_one.loc[(ti,tt,uid,field_id),:].dropna()
        trials_to_exclude = get_min_max_tick(X_raw_one_row, trial_index_to_index_within_df,ti,tt)
        
    return trials_to_exclude



def fit_glm_predict_rate_change_wrapper_exclude(spk_beh_df,ti,tt,uid,field_id,cell_cols_d,
                                    fr_map_trial_df_int_combined,
                                    all_sw_with_metrics,all_fields_one_sess,X_raw_one_sess,trial_index_to_index_within_df,
                                    **kwargs
                                    ):
    '''
    exclude either the switch trials or the largest tick trials
    '''

    if tt=='both':
        spk_beh_df_onetask = spk_beh_df.loc[spk_beh_df['task_index']==ti]
    else:
        spk_beh_df_onetask = spk_beh_df.loc[spk_beh_df['trial_type']==(ti,tt)]
    uid_endog  = uid
    cell_cols_exog = cell_cols_d['int']

    # if tt=='both':
    #     fr_map_trial_df_exog = fr_map_trial_df_bothtt_int.loc[ti]
    # else:
    #     fr_map_trial_df_exog = fr_map_trial_df_all_int.loc[(ti,tt),:]
    
    fr_map_trial_df_exog = fr_map_trial_df_int_combined.loc[(ti,tt),:]

    trials_to_exclude = get_trials_to_exclude_onefield(all_sw_with_metrics,X_raw_one_sess,uid,field_id,ti,tt,trial_index_to_index_within_df
                        )
    # if all_sw_with_metrics is not None:
    # all_sw_one_field = all_sw_with_metrics.loc[ti].query('trialtype==@tt&uid==@uid')
    # if all_sw_one_field.shape[0] > 0:
    #     trials_to_exclude = all_sw_one_field['trial_index']
    # else:
    #     X_raw_one_row=X_raw_one_sess.loc[(ti,tt,uid,field_id),:].dropna()
    #     X_diff = X_raw_one_row.diff()
    #     max_tick_tr=X_diff.idxmax()
    #     min_tick_tr = X_diff.idxmin()
    #     if tt=='both':
    #         trials_to_exclude = [max_tick_tr,min_tick_tr]
    #     else:
    #         trials_to_exclude = [index_within_to_trial_index_df.loc[ti,tt,max_tick_tr],index_within_to_trial_index_df.loc[ti,tt,min_tick_tr]]



    all_fields_one = all_fields_one_sess.loc[(ti,tt,uid,field_id),:]
    # all_fields_one = all_fields_recombined_all.loc[(ani,sess,ti,tt,uid,field_id),:]
    # glm_res, glm_res_df, pred_val, mean_within_field_pred = pp.fit_glm_predict_rate_change(spk_beh_df_onetask,uid_endog,cell_cols_exog,trials_to_exclude,
    #                         fr_map_trial_df_exog,all_fields_one,model_type="linear",
    #                         verbose=False)
    alpha_l = [0.01,0.001,0.0001,0.00001,0.000001]
    p_thresh = kwargs.get('p_thresh',0.05)
    model_type = kwargs.get('model_type',"linear")
    do_weighted_pred = kwargs.get('do_weighted_pred',False)
    do_inh_only = kwargs.get('do_inh_only',True)#True
    do_exc_only = kwargs.get('do_exc_only',False)
    # pdb.set_trace()
    glm_res, glm_res_df, pred_val, mean_within_field_pred = fit_glm_predict_rate_change(spk_beh_df_onetask,uid_endog,cell_cols_exog,trials_to_exclude,
                            fr_map_trial_df_exog,all_fields_one,model_type=model_type,alpha=alpha_l,
                            p_thresh=p_thresh,
                            do_inh_only = do_inh_only, do_exc_only=do_exc_only,                                                                                       
                            do_weighted_pred=do_weighted_pred,
                            verbose=False)
    
    return glm_res, glm_res_df, pred_val, mean_within_field_pred


def sweep_fit_glm_predict_rate_change(spk_beh_df,all_sw_with_metrics,all_fields_one_sess,
                                    index_within_to_trial_index_df,
                                    fr_map_trial_df_int_combined,
                                    # fr_map_trial_df_bothtt_int,
                                    # fr_map_trial_df_all_int,
                                    cell_cols_d,
                                    do_inh_only=True,
                                    do_weighted_pred=False,
                                    pval_thresh=None,
                                    ti=0):
    if all_sw_with_metrics is not None:
        gpb = all_sw_with_metrics.loc[ti].groupby(['trialtype','uid','field_index'])
    else:
        gpb = all_fields_one_sess.loc[ti].groupby(level=[0,1,2])
    mean_within_field_pred_all = {}
    r2_all = {}
    glm_res_df_all={}
    for k,val in gpb:
        tt,uid,field_id = k
        glm_res, glm_res_df, pred_val, mean_within_field_pred = fit_glm_predict_rate_change_wrapper(spk_beh_df,ti,tt,uid,field_id,cell_cols_d,
                                        # fr_map_trial_df_bothtt_int,
                                        # fr_map_trial_df_all_int,
                                        fr_map_trial_df_int_combined,
                                        all_sw_with_metrics,all_fields_one_sess,
                                        do_inh_only=do_inh_only,
                                        do_weighted_pred = do_weighted_pred, 
                                        pval_thresh = pval_thresh,                                                                                                                             
                                        )
        if tt=='both':
            
            mean_within_field_pred.index=index_within_to_trial_index_df.loc[ti].sort_values().values
        else:
            mean_within_field_pred.index=index_within_to_trial_index_df.loc[ti,tt].index
            
        mean_within_field_pred_all[k] = mean_within_field_pred
        r2_all[k] = glm_res.rsquared
        glm_res_df_all[k] = glm_res_df.T
    mean_within_field_pred_all = pd.concat(mean_within_field_pred_all,axis=0).unstack()
    r2_all = pd.Series(r2_all)
    glm_res_df_all = pd.concat(glm_res_df_all,axis=0)
    glm_res_df_all=glm_res_df_all.unstack().swaplevel(0,1,axis=1)

    return mean_within_field_pred_all,glm_res_df_all,r2_all

def sweep_fit_glm_predict_rate_change_exclude(spk_beh_df,all_sw_with_metrics,all_fields_one_sess,X_raw_one_sess,
                                    index_within_to_trial_index_df,
                                    fr_map_trial_df_int_combined,
                                    # fr_map_trial_df_bothtt_int,
                                    # fr_map_trial_df_all_int,
                                    cell_cols_d,
                                    do_inh_only=True,
                                    do_weighted_pred=False,
                                    pval_thresh=None,
                                    ti=0):
    '''
    fit for all neurons; for switch neurons exclude the switch trials; for non switch neurons exclude the largest uptick and downtick trials
    '''                                    
    # if all_sw_with_metrics is not None:
        # gpb = all_sw_with_metrics.loc[ti].groupby(['trialtype','uid','field_index'])
    trial_index_to_index_within_df = dpp.trial_index_to_index_within_trialtype(spk_beh_df)
    all_sw_with_metrics_reindex=all_sw_with_metrics.loc[ti].set_index(['trialtype','uid','field_index'],append=True)
    gpb = all_fields_one_sess.loc[ti].groupby(level=[0,1,2])
    mean_within_field_pred_all = {}
    r2_all = {}
    glm_res_df_all={}
    for k,val in gpb:
        tt,uid,field_id = k
        glm_res, glm_res_df, pred_val, mean_within_field_pred = fit_glm_predict_rate_change_wrapper_exclude(spk_beh_df,ti,tt,uid,field_id,cell_cols_d,
                                        # fr_map_trial_df_bothtt_int,
                                        # fr_map_trial_df_all_int,
                                        fr_map_trial_df_int_combined,
                                        all_sw_with_metrics,all_fields_one_sess,X_raw_one_sess,trial_index_to_index_within_df,
                                        do_inh_only=do_inh_only,
                                        do_weighted_pred = do_weighted_pred, 
                                        pval_thresh = pval_thresh,                                                                                                                             
                                        )
        if tt=='both':
            
            mean_within_field_pred.index=index_within_to_trial_index_df.loc[ti].sort_values().values
        else:
            mean_within_field_pred.index=index_within_to_trial_index_df.loc[ti,tt].index
            
        mean_within_field_pred_all[k] = mean_within_field_pred
        r2_all[k] = glm_res.rsquared
        glm_res_df_all[k] = glm_res_df.T
    mean_within_field_pred_all = pd.concat(mean_within_field_pred_all,axis=0).unstack()
    r2_all = pd.Series(r2_all)
    glm_res_df_all = pd.concat(glm_res_df_all,axis=0)
    glm_res_df_all=glm_res_df_all.unstack().swaplevel(0,1,axis=1)

    return mean_within_field_pred_all,glm_res_df_all,r2_all

def add_inh_fr_change_to_all_sw(all_sw_with_metrics,mean_within_field_pred_all,per_field_metrics_one,selected_inh_fr_trial_within_field=None,ti=0):
    change_in_inh_fr_allsw={}
    per_field_metrics_allsw={}
    if selected_inh_fr_trial_within_field is not None:
        change_in_selected_inh_fr_allsw = {}
    for i, row in all_sw_with_metrics.loc[ti].iterrows():
        tt,uid,field_id=row['trialtype'],row['uid'],row['field_index']
        sw_tr = row['switch_trial']
        sw = row['switch']
        change_in_inh_fr=mean_within_field_pred_all.loc[tt,uid,field_id].loc[sw_tr] - mean_within_field_pred_all.loc[tt,uid,field_id].loc[sw_tr-1]
        change_in_inh_fr_allsw[i] = change_in_inh_fr

        if selected_inh_fr_trial_within_field is not None:
            change_in_selected_inh_fr=selected_inh_fr_trial_within_field.loc[tt,uid,field_id].loc[sw_tr] - selected_inh_fr_trial_within_field.loc[tt,uid,field_id].loc[sw_tr-1]
            change_in_selected_inh_fr_allsw[i] = change_in_selected_inh_fr

        per_field_metrics_one_sw = per_field_metrics_one.loc[(ti,tt,uid,field_id),['fit_var_ratio','sparsity','mean']]
        per_field_metrics_allsw[i] = per_field_metrics_one_sw
    change_in_inh_fr_allsw = pd.Series(change_in_inh_fr_allsw)
    
    per_field_metrics_allsw = pd.concat(per_field_metrics_allsw,axis=0).unstack()
        
    all_sw_with_metrics_oneti_with_inh_change=copy.copy(all_sw_with_metrics.loc[ti])
    all_sw_with_metrics_oneti_with_inh_change['inh_fr_change'] = change_in_inh_fr_allsw
    if selected_inh_fr_trial_within_field is not None:
        change_in_selected_inh_fr_allsw = pd.Series(change_in_selected_inh_fr_allsw)
        all_sw_with_metrics_oneti_with_inh_change['selected_inh_fr_change'] = change_in_selected_inh_fr_allsw
    all_sw_with_metrics_oneti_with_inh_change = pd.concat([all_sw_with_metrics_oneti_with_inh_change,per_field_metrics_allsw],axis=1)

    return all_sw_with_metrics_oneti_with_inh_change

import pynapple as nap

def get_binned_spk_presleep(spike_trains,mergepoints):
    pre_sleep_int = mergepoints['timestamps'][0]
    intset = nap.interval_set.IntervalSet(start=pre_sleep_int[[0]],end=pre_sleep_int[[1]])
    binned_spk = spike_trains.count(0.1,ep=intset)        
    return binned_spk

import tqdm
def sweep_fit_glm_sleep_all_uid(spike_trains,cell_cols_d,time_bin=0.1,pre_sleep_int=None,mergepoints=None,**kwargs):
    if pre_sleep_int is None:
        pre_sleep_int = mergepoints['timestamps'][0]
    intset = nap.interval_set.IntervalSet(start=pre_sleep_int[[0]],end=pre_sleep_int[[1]])
    binned_spk = spike_trains.count(time_bin,ep=intset)

    spk_beh_df_onetask = binned_spk
    cell_cols_exog = cell_cols_d['int']
    glm_res_df_all = {}
    for uid_endog in tqdm.tqdm(cell_cols_d['pyr']):
        glm_res,glm_res_df=fit_glm(spk_beh_df_onetask,uid_endog,cell_cols_exog,trials_to_exclude=None,
                    **kwargs)
        glm_res_df_all[uid_endog] = glm_res_df
    glm_res_df_all = pd.concat(glm_res_df_all,axis=0)
    return glm_res_df_all

def get_all_int_within_field_all_pyr(all_fields_one_sess,fr_map_trial_df_int_combined):
    '''
    within_field_int_trial_df : ti,tt,uid,fieldid,int_id x trial
    '''
    gpb = fr_map_trial_df_int_combined.groupby(level=(0,1))
    within_field_int_trial_df = {}
    for (ti,tt),val in gpb:
        all_fields_one_tt = all_fields_one_sess.loc[(ti,tt),:]
        val = val.loc[(ti,tt),:].dropna(axis=1,how='all')
        for (uid,field_id),row in all_fields_one_tt.iterrows():
            st,ed = row['start'],row['end']
            within_field_int_trial_onepyr=val.loc[(slice(None),slice(st,ed)),:].groupby(level=0).mean()
            within_field_int_trial_df[(ti,tt,uid,field_id)] = within_field_int_trial_onepyr
    within_field_int_trial_df = pd.concat(within_field_int_trial_df,axis=0)
    return within_field_int_trial_df
            
def agg_int_within_field_all_pyr(within_field_int_trial_df,mask,ma_level=[2,4],other_level=[0,1,3]):
    '''
    within_field_int_trial_df: ti,tt,uid,fieldid,int_id x trial
    '''
    
    if mask is None:
        within_field_int_trial_df_ma = within_field_int_trial_df
    else:
        permutation = np.array(ma_level+other_level)
        inverse_perm = []
        for i in range(len(permutation)):
            inverse_perm.append(np.nonzero(permutation==i)[0][0]) 
        within_field_int_trial_df_ma = within_field_int_trial_df.reorder_levels(permutation).loc[mask].reorder_levels(inverse_perm) # to make mask work, has to reorder levels to make mask levels in the front
        
    mean_within_field_int_trial_df= within_field_int_trial_df_ma.groupby(level=(0,1,2,3)).mean()
    return mean_within_field_int_trial_df

        
