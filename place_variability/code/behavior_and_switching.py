import numpy as np
import scipy
import matplotlib.pyplot as plt
import sys,os,copy,pdb,importlib
from importlib import reload
import pandas as pd

sys.path.append('/mnt/home/szheng/projects/util_code')
import data_prep_pyn as dpp

def get_categorical_var(df,var_bin_d):
    for k,val in var_bin_d.items():
        k_cat = k+'_cat'
        df[k_cat] = pd.cut(df[k],val,retbins=False,labels=False,include_lowest=True)
    return df


def get_corret_prev_trialwithin_bothtt(correct_per_trial,trial_index_to_index_within_df):
    '''
    get whether the previous trial within the same trialtype is correct
    
    correct_prev_trialwithin_bothtt: series, index: trial index 
    '''
    trial_index_to_index_within_df_reset=trial_index_to_index_within_df.reset_index(level=(0,1))
    trial_index_to_index_within_df_reset.columns=['task_ind','tt_ind','index_within']
    trial_index_to_index_within_df_reset['correct']= correct_per_trial
    gpb = trial_index_to_index_within_df_reset.groupby('tt_ind')
    correct_prev_trialwithin_bothtt = {}
    for k,val in gpb:
        correct_prev_trialwithin = val[['index_within','correct']].reset_index().set_index('index_within').sort_index()
        correct_prev_trialwithin['correct'] = correct_prev_trialwithin['correct'].shift(1).values
        correct_prev_trialwithin_bothtt[k]=correct_prev_trialwithin
    correct_prev_trialwithin_bothtt = pd.concat(correct_prev_trialwithin_bothtt,axis=0)
    correct_prev_trialwithin_bothtt= correct_prev_trialwithin_bothtt.set_index('trial_ind')
    correct_prev_trialwithin_bothtt = correct_prev_trialwithin_bothtt.sort_index()['correct']
    
    return correct_prev_trialwithin_bothtt

def get_correct_df_all(spk_beh_df_all,ti=0):
    '''
    get the correct per trial; including correct in previous trial/within trialtype
    '''
    spk_beh_df_all_sub = spk_beh_df_all.query('(task_index==@ti)&(directed_locomotion)')
    correct_df_all = {}
    for k,sbd in spk_beh_df_all_sub.groupby(level=(0,1)):
        sbd=sbd.loc[k]
        trial_index_to_index_within_df = dpp.trial_index_to_index_within_trialtype(sbd)
        correct_per_trial = sbd.groupby('trial')['correct'].median()
        correct_per_prev_trial = correct_per_trial.shift(1)
        correct_prev_trialwithin_bothtt = get_corret_prev_trialwithin_bothtt(correct_per_trial,trial_index_to_index_within_df)

        correct_df = {'correct':correct_per_trial,'correct_prev':correct_per_prev_trial,'correct_prev_trialwithin':correct_prev_trialwithin_bothtt}
        correct_df = pd.concat(correct_df,axis=1)
        correct_df_all[(*k,ti)] = correct_df
    correct_df_all = pd.concat(correct_df_all,axis=0)
    return correct_df_all

def add_corret_prev_seperate(var_per_trial_pos_all,pos_key='lin_binned_cat',pos_thresh=1):
    '''
    correct_prev_seperate: for central arm, prev is the trial before; for side, prev is within trialtype
    '''
    ma_central =  (var_per_trial_pos_all[pos_key]<=pos_thresh)
    ma_side =   var_per_trial_pos_all[pos_key]>pos_thresh

    var_per_trial_pos_all['correct_prev_seperate'] = np.nan
    var_per_trial_pos_all.loc[ma_central,'correct_prev_seperate'] = var_per_trial_pos_all.loc[ma_central,'correct_prev']
    var_per_trial_pos_all.loc[ma_side,'correct_prev_seperate'] = var_per_trial_pos_all.loc[ma_side,'correct_prev_trialwithin']
    return var_per_trial_pos_all



def combine_correct_with_var_per_trial_pos_all(var_per_trial_pos_all,correct_df_all,pos_key='lin_binned_cat',pos_thresh=1):
    '''
    combine correct_all_df
    and
    var_per_trial_pos_all
    '''
    gpb=var_per_trial_pos_all.set_index('trial',append=True).groupby(level=(0,1,2,3),sort=False)
    correct_val_l = []
    for k,val in gpb:
        row=correct_df_all.loc[val.index[0],:]
        nrows = val.shape[0]
        correct_val = np.tile(row.values,(nrows,1))
        correct_val_l.append(correct_val)
    correct_val_l = np.concatenate(correct_val_l,axis=0)
    correct_val_l = pd.DataFrame(correct_val_l,columns=correct_df_all.columns)
    inds = var_per_trial_pos_all.index
    var_per_trial_pos_all=  pd.concat([var_per_trial_pos_all.reset_index(drop=True),correct_val_l],axis=1)
    var_per_trial_pos_all.index= inds
    
    var_per_trial_pos_all = add_corret_prev_seperate(var_per_trial_pos_all,pos_key=pos_key,pos_thresh=pos_thresh)

    return var_per_trial_pos_all

def get_n_fied_per_pos_per_trial_all(all_fields_recombined_all,trial_index_to_index_within_df_all,
                                     var_bin_d={'lin_binned':np.array([0,15,74,111,185,202,222])/2.2},
                                        combine_first_last_section=True,
                                     pos_key = 'peak',task_index=0,
                                    ):
    '''
    trial: each trial type have different place fields
    get: ani,sess,ti (only 0),tt, trial_index - n_field
    '''
    if isinstance(trial_index_to_index_within_df_all,dict):
        trial_index_to_index_within_df_all = pd.concat(trial_index_to_index_within_df_all,axis=0)
    lin_binned_cat_max = lin_binned_cat_max = len(var_bin_d['lin_binned'])-2
    n_field_per_pos_one_tt_per_trial_all = []
    for k,val in trial_index_to_index_within_df_all.groupby(level=(0,1,2,3)):
        ani,sess,ti,tt=k
        if ti==task_index: # only do familiar t maze for now
            if 'both' in all_fields_recombined_all.loc[(ani,sess,ti),:].index.get_level_values(0):
                tt_l = [tt,'both']
            else:
                tt_l = tt
            pf_loc=all_fields_recombined_all.loc[(ani,sess,ti,tt_l),:][pos_key]
            pf_loc_count = pf_loc.value_counts(sort=False).sort_index().reset_index()
            pf_loc_count.columns=['lin_binned','count']
            pf_loc_count=get_categorical_var(pf_loc_count,var_bin_d)
            if combine_first_last_section:
                pf_loc_count.loc[pf_loc_count['lin_binned_cat']==lin_binned_cat_max] = 0

            n_field_per_pos_one_tt = pf_loc_count.groupby('lin_binned_cat')['count'].sum()
            ntrials = len(val)
            n_field_per_pos_one_tt_per_trial=pd.DataFrame(np.tile(n_field_per_pos_one_tt,(ntrials,1)),index=val.index)
            n_field_per_pos_one_tt_per_trial = n_field_per_pos_one_tt_per_trial.stack()
    #         n_field_per_pos_one_tt_per_trial_all[ani,sess,ti,tt] = n_field_per_pos_one_tt_per_trial
            n_field_per_pos_one_tt_per_trial_all.append(n_field_per_pos_one_tt_per_trial)
    n_field_per_pos_one_tt_per_trial_all = pd.concat(n_field_per_pos_one_tt_per_trial_all,axis=0)
    return n_field_per_pos_one_tt_per_trial_all



def prep_regression(spk_beh_df_all,all_fields_recombined_all,all_sw_d_all,trial_index_to_index_within_df_all,cell_cols_d_all=None,**kwargs):
    '''
    spk_beh_df_all: (ani,sess,ind) x []
    assuming all task_index==0 share the same maze structure

    optional:
    combine_first_last_section
    '''

    beh_var_l = ['task_index','trial_type','trial','time','lin','lin_binned','speed_gauss','v_gauss','off_track_event','pause_event','directed_locomotion']
    beh_df_all = spk_beh_df_all[beh_var_l]
    task_index = kwargs.get('task_index',0)
    # filter df, only keep task_index==0, and directed_locomotion
    beh_df_all_fam = beh_df_all.query('(task_index==@task_index) & (directed_locomotion==1)')

    # assign categorical variables
    # var_bin_d_default = {'lin_binned':np.array([0,15,44.5,74,92.5,111,148,185,202,222])/2.2}
    var_bin_d_default = {'lin_binned':np.array([0,15,74,111,185,202,222])/2.2}
    var_bin_d = kwargs.get('var_bin_d',var_bin_d_default)
    lin_binned_cat_max = len(var_bin_d['lin_binned'])-2
    beh_df_all_fam = get_categorical_var(beh_df_all_fam,var_bin_d)

    combine_first_last_section = kwargs.get('combine_first_last_section',True)
    # update lin cat, combine two home region
    if combine_first_last_section:
        beh_df_all_fam.loc[beh_df_all_fam['lin_binned_cat']==lin_binned_cat_max,'lin_binned_cat'] = 0
    
    # get E I activities and ratio
    if cell_cols_d_all is not None:
        do_neural=True
    else:
        do_neural=False
    if do_neural:
        ei_res_df_all = get_ei_var_spk_beh_df_all(spk_beh_df_all,cell_cols_d_all)
        beh_df_all_fam = pd.concat([beh_df_all_fam,ei_res_df_all.loc[beh_df_all_fam.index]],axis=1)
        neural_cols =ei_res_df_all.columns 



    # aggregate per trial, lin_cat
    # ind_var_per_trial_pos_all=beh_df_all_fam.groupby(level=(0,1)).apply(lambda x:x.groupby(['task_index','trial','lin_binned_cat']).agg({'speed_gauss':'mean'}))
    gpb = beh_df_all_fam.set_index(['task_index','trial','lin_binned_cat'],append=True).groupby(level=(0,1,3,4,5))
    ind_var_per_trial_pos_all = pd.concat({'speed_gauss_mean':gpb['speed_gauss'].mean(),'speed_gauss_std':gpb['speed_gauss'].std()},axis=1)
    ind_var_per_trial_pos_all['speed_gauss_cv'] = ind_var_per_trial_pos_all['speed_gauss_std'] / ind_var_per_trial_pos_all['speed_gauss_mean']
    if do_neural:
        neural_var = gpb[neural_cols].mean()
        ind_var_per_trial_pos_all = pd.concat([ind_var_per_trial_pos_all,neural_var],axis=1)
    # ind_var_per_trial_pos_all=beh_df_all_fam.groupby(level=(0,1)).apply(lambda x:x.groupby(['task_index','trial','lin_binned_cat']).agg({'speed_gauss':['mean','std']}))
    # pdb.set_trace()
    # get time bin and time per trial and pos!!!!
    n_timebin_per_trial_pos_all = beh_df_all_fam.groupby(level=(0,1)).apply(lambda x:x.groupby(['task_index','trial','lin_binned_cat']).count().iloc[:,0])
    dt_per_ani_sess_task =beh_df_all.set_index('task_index',append=True).groupby(level=(0,1,3)).apply(lambda x:np.median(np.diff(x['time'])))
    time_per_trial_pos_all = n_timebin_per_trial_pos_all.groupby(level=(0,1,2)).apply(lambda x:x * dt_per_ani_sess_task.loc[x.name])
    ind_var_per_trial_pos_all['occupancy'] = time_per_trial_pos_all
    
    lin_bin_d = {'lin_binned':var_bin_d['lin_binned']}
    pos_key = kwargs.get('pos_key','peak')
    # get n field redone
    # # get n fields per pos!!!
    # pos_key = kwargs.get('pos_key','peak')
    # lin_bin_d = {'lin_binned':var_bin_d['lin_binned']}
    # n_field_per_ani_sess_task_pos = all_fields_recombined_all.groupby(level=(0,1,2)).apply(lambda x:x[pos_key].astype(int).value_counts(sort=False).sort_index(level=3)) # assuming pos_key column is int
    # n_field_per_ani_sess_task_pos = pd.DataFrame(n_field_per_ani_sess_task_pos.values,index=n_field_per_ani_sess_task_pos.index,columns=['n_field'])
    # n_field_per_ani_sess_task_pos = n_field_per_ani_sess_task_pos.reset_index(level=3).rename({'level_3':'lin_binned'},axis=1)
    # n_field_per_ani_sess_task_pos = get_categorical_var(n_field_per_ani_sess_task_pos,lin_bin_d) 
    #     # combine two sections for home
    # if combine_first_last_section:
    #     n_field_per_ani_sess_task_pos.loc[n_field_per_ani_sess_task_pos['lin_binned_cat']==lin_binned_cat_max,'lin_binned_cat'] = 0
    
    # n_field_per_ani_sess_task_pos = n_field_per_ani_sess_task_pos.set_index('lin_binned_cat',append=True)
    # n_field_per_ani_sess_task_pos = n_field_per_ani_sess_task_pos.groupby(level=(0,1,2,3))['n_field'].sum()    
    #     # assign nfield to each trial
    # gpb=ind_var_per_trial_pos_all.groupby(level=(0,1,2,3))
    # n_field_d = {}
    # for k,val in gpb:
    #     ani,sess,task,tr=k
    #     n_field_d[k] = n_field_per_ani_sess_task_pos.loc[ani,sess,task]
    # n_field_d=pd.concat(n_field_d)
    # ind_var_per_trial_pos_all['n_field'] = n_field_d

    all_sw_d_all_copy = copy.copy(all_sw_d_all)
    all_sw_d_all_copy['lin_binned'] = all_sw_d_all_copy['field_pos']
    all_sw_d_all_copy=get_categorical_var(all_sw_d_all_copy,lin_bin_d)
    if combine_first_last_section: # NB!!!! missed this before!!
        all_sw_d_all_copy.loc[all_sw_d_all_copy['lin_binned_cat']==lin_binned_cat_max,'lin_binned_cat']=0

    get_sw_per_trial_pos_func = lambda x:x.groupby(['switch','trial_index','lin_binned_cat']).count().iloc[:,0]
    sw_per_trial_pos_all = all_sw_d_all_copy.groupby(level=(0,1,2)).apply(get_sw_per_trial_pos_func)

    onoff_str_d={1:'on',-1:'off'}
    for onoff in [1,-1]:
        sw_per_trial_pos_all_onoff=sw_per_trial_pos_all.loc[(slice(None),slice(None),slice(None),onoff)]
        onoff_str = onoff_str_d[onoff]
        ind_var_per_trial_pos_all[f'sw_{onoff_str}'] = sw_per_trial_pos_all_onoff
        ind_var_per_trial_pos_all[f'sw_{onoff_str}'] = ind_var_per_trial_pos_all[f'sw_{onoff_str}'].fillna(0)

    var_per_trial_pos_all = ind_var_per_trial_pos_all
    var_per_trial_pos_all = var_per_trial_pos_all.reset_index(['trial','lin_binned_cat'])
    var_per_trial_pos_all['animal'] = var_per_trial_pos_all.index.get_level_values(0)
    var_per_trial_pos_all['session'] = var_per_trial_pos_all.index.get_level_values(1)

    # redo the ratio, cuz ratio in individual bin can be noisy and bias things
    if do_neural:
        eps=1e-10
        var_per_trial_pos_all['E_I_mean'] = var_per_trial_pos_all['E_mean'] / (var_per_trial_pos_all['I_mean'] + eps)
        var_per_trial_pos_all['E_I_frac_active'] = var_per_trial_pos_all['E_frac_active'] / (var_per_trial_pos_all['I_frac_active'] + eps)

    # zscore speed, within session
    # var_per_trial_pos_all = var_per_trial_pos_all.groupby(level=1).apply(lambda x:x.assign(speed_z=scipy.stats.zscore(x['speed_gauss'])))
    var_per_trial_pos_all = var_per_trial_pos_all.groupby(level=1).apply(lambda x:x.assign(speed_z=scipy.stats.zscore(x['speed_gauss_mean'])))
    var_per_trial_pos_all = var_per_trial_pos_all.groupby(level=1).apply(lambda x:x.assign(speed_std_z=scipy.stats.zscore(x['speed_gauss_std'])))
    var_per_trial_pos_all = var_per_trial_pos_all.groupby(level=1).apply(lambda x:x.assign(speed_cv_z=scipy.stats.zscore(x['speed_gauss_cv'])))
    # zscore trial, within session
    var_per_trial_pos_all = var_per_trial_pos_all.groupby(level=1).apply(lambda x:x.assign(trial_z=scipy.stats.zscore(x['trial'])))
    # var_per_trial_pos_all = var_per_trial_pos_all.groupby('session').apply(lambda x:x.assign(trial_z=sklearn.preprocessing.quantile_transform(x[['trial']].values)[:,0]))
    
    # filter out initial trials
    ge_than_trial = kwargs.get('ge_than_trial',4)
    var_per_trial_pos_all = var_per_trial_pos_all.query('trial>=@ge_than_trial')

    # redo nfield
    n_field_per_pos_one_tt_per_trial_all=get_n_fied_per_pos_per_trial_all(all_fields_recombined_all,trial_index_to_index_within_df_all,
                                     var_bin_d=var_bin_d,
                                        combine_first_last_section=combine_first_last_section,
                                     pos_key = pos_key,task_index=task_index
                                    )
    var_per_trial_pos_all_re_nfield=var_per_trial_pos_all.set_index(['trial','lin_binned_cat'],append=True)
    n_field_per_pos_one_tt_per_trial_all_droplevel=n_field_per_pos_one_tt_per_trial_all.droplevel(3)
    inds_intersect=n_field_per_pos_one_tt_per_trial_all_droplevel.index.intersection(var_per_trial_pos_all_re_nfield.index) # drop trials that were excluded in trial_index_to_ind_within_df_all, which were excluded from preprocessing 
    var_per_trial_pos_all_re_nfield.loc[inds_intersect,'n_field']  = n_field_per_pos_one_tt_per_trial_all_droplevel.loc[inds_intersect].values
    # pdb.set_trace()
    var_per_trial_pos_all = var_per_trial_pos_all_re_nfield.reset_index(['trial','lin_binned_cat']) # cannot dropna, for linearmaze correc would give nan
    



    # div n_field
    for y_key in ['sw_on','sw_off']:
        var_per_trial_pos_all[f'{y_key}_div_n_field'] = var_per_trial_pos_all[y_key] / var_per_trial_pos_all['n_field']
    var_per_trial_pos_all = var_per_trial_pos_all.fillna(0) # if n_field is 0, n_sw has to be 0
    
    # add in correct
    correct_df_all = get_correct_df_all(spk_beh_df_all,ti=task_index)
    var_per_trial_pos_all = combine_correct_with_var_per_trial_pos_all(var_per_trial_pos_all,correct_df_all)    

    



    return var_per_trial_pos_all

import sklearn
from sklearn.model_selection import StratifiedShuffleSplit
from patsy import dmatrices
import statsmodels.formula.api as smf
import statsmodels.api as sm

def build_formula_one(key_l,cat_l,y_key):
    term_l = []
    for k,is_c in zip(key_l,cat_l):
        if is_c:
            term = f"C({k})"
        else:
            term = k
        term_l.append(term)
    formula = "+".join(term_l)
    formula = y_key + '~' + formula
    return formula
from itertools import combinations
def build_formula_multiple(key_l,cat_l,y_key,combo_l=None):
    n_keys = len(key_l)
    if combo_l is None:
        combo_l = []
        for r in range(n_keys):
            combo_l.extend(list(combinations(range(n_keys),r)))
            
    formula_l = []
    for combo in combo_l:
        if len(combo)>0:
            combo = list(combo)
            formula = build_formula_one(key_l[combo],cat_l[combo],y_key)
        else:
            formula=f'{y_key}~1'
        formula_l.append(formula)
    return formula_l

def cv_poisson_glm_one_formula(X,formula,n_splits=5,group_level=0):
    '''
    which level of X's multi index to use for stratified shuffle split
    '''
    endog,exog=dmatrices(formula,data=X)
    endog = np.squeeze(endog)
    kf = StratifiedShuffleSplit(n_splits=n_splits,random_state=None)
    sc_l=[]
    clf=sklearn.linear_model.PoissonRegressor(alpha=0,fit_intercept=False,max_iter=10000)
    grp = X.index.get_level_values(group_level)
    for train_index, test_index in kf.split(X,grp):
        X_train,y_train = exog[train_index],endog[train_index]
        X_test,y_test = exog[test_index],endog[test_index]
        clf.fit(X_train,y_train)
        sc=clf.score(X_test,y_test)
        sc_l.append(sc)
    
    model=smf.glm(formula=formula,data=X,family=sm.families.Poisson(sm.families.links.Log()))
    res=model.fit()
    
    sc_l = np.array(sc_l)

    return sc_l,res


def cv_poisson_glm_multi_formula(X,formula_l,n_splits=5,group_level=0,random_state=1):
    '''
    
    '''
    kf = StratifiedShuffleSplit(n_splits=n_splits,random_state=random_state)
    sc_l_all=[]
    grp = X.index.get_level_values(group_level)
    res_all = []
    for formula in formula_l:
        # pdb.set_trace()
        endog,exog=dmatrices(formula,data=X)
        endog = np.squeeze(endog)
        sc_l_one = []
        for train_index, test_index in kf.split(X,grp):
            clf=sklearn.linear_model.PoissonRegressor(alpha=0,fit_intercept=False,max_iter=10000)
            X_train,y_train = exog[train_index],endog[train_index]
            X_test,y_test = exog[test_index],endog[test_index]
            clf.fit(X_train,y_train)
            sc=clf.score(X_test,y_test)
            sc_l_one.append(sc)
        sc_l_one = np.array(sc_l_one)
        model=smf.glm(formula=formula,data=X,family=sm.families.Poisson(sm.families.links.Log()))
        res=model.fit()
        res_all.append(res)
        sc_l_all.append(sc_l_one)
    sc_l_all = np.array(sc_l_all)
    return sc_l_all,res_all



        

#=====aggregating switches=====#

def agg_within_trial(var_per_trial_pos_all):
    '''
    var_per_trial_pos_all from prep_regression, variable per animal, session, task (only one), trial, position category
    turn into: variable per animal, session, task (only one), trial
    '''
    y_key_l = ['sw_on','sw_off','sw_on_div_n_field','sw_off_div_n_field','n_field']
    var_per_trial=var_per_trial_pos_all.groupby(level=(0,1,2)).apply(lambda x:x.groupby('trial')[y_key_l].sum())
    for onoff in ['on','off']:
        var_per_trial[f'sw_{onoff}_div_n_field'] = var_per_trial[f'sw_{onoff}'] / var_per_trial['n_field'] # sum n field across pos within trial, divide that; the sw_on_div_n_field from the first sum is not good, because each is n_sw / n_field_within_segment 
    var_per_trial = var_per_trial.fillna(0) # if n_field =0, n_sw must be 0
    to_add= var_per_trial_pos_all.groupby(level=(0,1,2)).apply(lambda x:x.groupby('trial')[['trial_z','speed_z']].mean())
    var_per_trial =pd.concat([var_per_trial ,to_add],axis=1)
    return var_per_trial




def shuffle_f_test_with_posthoc(data,shuffle_groupby_level='session',group_key='lin_cat',val_key = 'sw_on',div_by=['n_field','duration'],n_repeats = 200):
    '''
    shuffle_groupby_level: groupby first to shuffle each subgroup, i.e. session
    group_key: group for anova
    val_key: value to comapre
    div_by: can normalize the value; these won't be shuffled together with the value


    '''
    if div_by is not None:
        val_key_div = val_key + '_div_' + '_'.join(div_by)
        div_by_prod = data[div_by].prod(axis=1).values 
        data[val_key_div] = data[val_key] / div_by_prod
    else:
        val_key_div=val_key

    gpb = data.groupby(group_key)[val_key_div]
    mean_per_group = gpb.mean()
    mean_per_group_diff = mean_per_group.values[:,None] - mean_per_group.values[None,:]
    f_data,p_data=scipy.stats.f_oneway(*[x.dropna(axis=0) for x in dict(list(gpb)).values()])

    # key = 'sw_on_div_n_field_duration'
    # label = 'lin_cat'

    val_sh_l = []
    for i in range(n_repeats):
        val_sh= data.groupby(level=shuffle_groupby_level)[val_key].sample(frac=1,replace=False).values
        if div_by is not None:
            val_sh = val_sh / div_by_prod
        val_sh_l.append(val_sh)
        
    val_sh_l_df = pd.DataFrame(np.array(val_sh_l).T)
    val_sh_l_df[group_key] = data[group_key].values
    mean_per_group_sh = val_sh_l_df.groupby(group_key).mean()
    sh_cols = np.arange(n_repeats)

    gpb = val_sh_l_df.groupby(group_key)
    n_groups = len(gpb)
    n_within_group = gpb.count()
    MSB = np.sum(n_within_group*(mean_per_group_sh - val_sh_l_df[sh_cols].mean(axis=0))**2,axis=0) / (n_groups-1)
    MSW = gpb[sh_cols].apply(lambda x:np.sum((x.dropna() - x.dropna().mean(axis=0).values[None,:])**2,axis=0)).sum(axis=0) / (val_sh_l_df[0].dropna().shape[0] - n_groups)
    f_sh = MSB / MSW
    mean_per_group_diff_sh = np.stack(mean_per_group_sh.T.apply(lambda x:x.values[:,None]-x.values[None,:],axis=1).values)


    shuffle_f_test_res = {'f_data':f_data,'p_data':p_data,'f_shuffle':f_sh,
                        'mean_per_group_data':mean_per_group,'mean_per_group_diff_data':mean_per_group_diff,
                        'mean_per_group_shuffle':mean_per_group_sh,
                        'mean_per_group_diff_shuffle':mean_per_group_diff_sh,
                        }
                    
    return shuffle_f_test_res

def post_hoc_test(mean_per_group_diff_data,mean_per_group_diff_shuffle,alpha=0.05,do_bonf=True):
    n_groups = mean_per_group_diff_data.shape[0]
    n_tests = n_groups * (n_groups-1) / 2
    if do_bonf:
        alpha = alpha / n_tests
    
    sig_thresh = np.quantile(np.abs(mean_per_group_diff_shuffle),1-alpha,axis=0)
    pval = np.mean(np.abs(mean_per_group_diff_data) <= np.abs(mean_per_group_diff_shuffle),axis=0)
    sig = np.abs(mean_per_group_diff_data) > sig_thresh
    sig_inds = np.nonzero(sig)
    pval_selected = pval[sig_inds]
    sig_inds = np.array(sig_inds).T

    keep_one_pair_ma = np.squeeze(np.diff(sig_inds,axis=1) > 0)
    sig_inds = sig_inds[keep_one_pair_ma]
    pval_selected = pval_selected[keep_one_pair_ma]
    return sig, pval, sig_inds, pval_selected

# prep for neural 
def get_ei_var_spk_beh_df(spk_beh_df,cell_cols_d):
    eps = 1e-10 
    pyr_df = spk_beh_df[cell_cols_d['pyr']]
    int_df = spk_beh_df[cell_cols_d['int']]
    res_df={}
    res_df['E_mean'] = pyr_df.mean(axis=1)
    res_df['I_mean'] = int_df.mean(axis=1)
    
    res_df['E_I_mean'] = res_df['E_mean'] / (res_df['I_mean'] + eps)
    res_df['E_frac_active'] = (pyr_df>0).mean(axis=1)
    res_df['I_frac_active'] = (int_df>0).mean(axis=1)
    
    res_df['E_I_frac_active'] = res_df['E_frac_active']/(res_df['I_frac_active']+eps)
    res_df = pd.concat(res_df,axis=1)
    
    return res_df

def get_ei_var_spk_beh_df_all(spk_beh_df_all,cell_cols_d_all):
    ei_res_df_all = []
    for k,sbd in spk_beh_df_all.groupby(level=(0,1)):
        ei_res_df=get_ei_var_spk_beh_df(sbd,cell_cols_d_all[k])
        ei_res_df_all.append(ei_res_df)
    ei_res_df_all  = pd.concat(ei_res_df_all,axis=0)
    return ei_res_df_all