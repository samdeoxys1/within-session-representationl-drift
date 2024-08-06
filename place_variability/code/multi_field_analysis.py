import numpy as np
import pandas as pd
import os,sys,pdb,copy
import scipy

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from pandarallel import pandarallel


def get_whether_otherrows_sw_all_row(val,cols=['on','off']):
    otherrows_sw_l = []#[]#{}
    for i,row in val.iterrows():
        
        otherrows=val.loc[val.index!=i]
        otherrows_sw = otherrows[cols].any(axis=0)
        otherrows_sw = otherrows_sw.to_frame().T
        otherrows_sw.index=[i[-1]]
        
        # otherrows_sw_l[i]=otherrows_sw
        otherrows_sw_l.append(otherrows_sw)
    
    otherrows_sw_l = pd.concat(otherrows_sw_l,axis=0)
    # otherrows_sw_l = otherrows_sw_l.droplevel(-1)
    return otherrows_sw_l

def get_whether_other_field_sw(has_sw_df,cols=['on','off']):
    pandarallel.initialize()
    otherrows_sw_all = []
    gpb = has_sw_df.groupby(level=(0,1,2,3,4),sort=False)
    otherrows_sw_all=gpb.parallel_apply(get_whether_otherrows_sw_all_row,cols=cols)
    # for k,val in has_sw_df.groupby(level=(0,1,2,3,4),sort=False):
    #     for i,row in val.iterrows():
    #         otherrows=val.loc[val.index!=i]
    #         otherrows_sw = otherrows[cols].any(axis=0)
    #         otherrows_sw = otherrows_sw.to_frame().T
    #         otherrows_sw_all.append(otherrows_sw)
    
    # otherrows_sw_all = pd.concat(otherrows_sw_all,axis=0).reset_index(drop=True)
    otherrows_sw_all.columns=['other_'+x for x in cols]
    # otherrows_sw_all.index=has_sw_df.index
    return otherrows_sw_all

def shuffle_has_sw_df(has_sw_df,all_fields_recombined_all,do_combine_tt_with_both=True,nrepeats=100):
    index=has_sw_df.index
    if do_combine_tt_with_both:
        has_sw_df_combined=combine_trial_type_with_both(has_sw_df,all_fields_recombined_all)
        # multiple_field_l = has_sw_df_combined['multiple_field']
        multiple_field_l=get_multiple_field_l(has_sw_df_combined,level=(0,1,2,3,4))
    else:
        multiple_field_l = has_sw_df['multiple_field']

    resample_one_df_l = {}
    for r in range(nrepeats):
        resample_one_d = {}
        for k in ['on','off']:
            resample_one=has_sw_df[k].sample(frac=1,replace=False)
            resample_one.index=index
            resample_one_d[k] = resample_one
        resample_one_d = pd.concat(resample_one_d,axis=1)
        if do_combine_tt_with_both:
            resample_one_d = combine_trial_type_with_both(resample_one_d,all_fields_recombined_all)
        resample_one_d['multiple_field']=multiple_field_l
        otherrows_sw_all = get_whether_other_field_sw(resample_one_d,cols=['on','off'])
        resample_one_df = pd.concat([resample_one_d,otherrows_sw_all],axis=1)
        
        resample_one_df_l[r]=resample_one_df
    resample_one_df_l = pd.concat(resample_one_df_l,axis=1)
    
    return resample_one_df_l
    
def combine_trial_type_with_both(df_all,all_fields_recombined_all,trial_type_l=[0,1]):
    '''
    df_all: ani,sess,task_ind,tt,uid,field_id
    can also adjust level if df is for individual sessions (currently manually adjust slice(None) and level )
    NB!! 'both' fields, index based on tt=0, would duplicate when combined with tt==1, luckily all_fields has the 'other_field_index' to give the index for combining with tt==1!! 
    '''
    df_merge_tt_both_d = {}
    trial_type_l=[0,1]
    for tt in trial_type_l:
        if tt==0:
            df_merge_tt_both=df_all.loc[(slice(None),slice(None),slice(None),[tt,'both']),:].reset_index(level=3,drop=True)
        else:
            df_both_part=df_all.loc[(slice(None),slice(None),slice(None),'both'),:]
            df_tt_part = df_all.loc[(slice(None),slice(None),slice(None),tt),:]
            inds=df_both_part.index
            field_index_for_both_from_another_tt=all_fields_recombined_all.loc[inds]['other_field_index']
            df_both_part['other_field_index']=field_index_for_both_from_another_tt
            df_both_part=df_both_part.droplevel(-1)
            df_both_part=df_both_part.set_index('other_field_index',append=True)
            df_both_part=df_both_part.rename_axis([None]*6)
            df_merge_tt_both = pd.concat([df_tt_part,df_both_part],axis=0).reset_index(level=3,drop=True)
        df_merge_tt_both['trialtype']=tt
        # tt_l=np.ones(df_merge_tt_both.shape[0])*tt
        # df_merge_tt_both.reindex(tt_l,level=3)
        df_merge_tt_both.set_index('trialtype',append=True)
        df_merge_tt_both_d[tt] = df_merge_tt_both
    df_merge_tt_both_d = pd.concat(df_merge_tt_both_d,axis=0)
    df_merge_tt_both_d = df_merge_tt_both_d.swaplevel(0,1).swaplevel(1,2).swaplevel(2,3)
    df_merge_tt_both_d = df_merge_tt_both_d.drop('trialtype',axis=1)
    return df_merge_tt_both_d

def get_multiple_field_l(has_sw_df,level=(0,1,2,3,4)):
    '''
    (the lesson: when using groupby, careful about order and indices of the result! best to include indices)
    '''
    multiple_field_l = []
    for k,val in has_sw_df.groupby(level=(0,1,2,3,4),sort=False):
    #     has_sw_df.loc[val.index,'multiple_field']=val.count()>1
        count_val=np.ones(val.shape[0],dtype=bool)*(val['on'].count()>1)
        count_df=pd.DataFrame(count_val,index=val.index)
        multiple_field_l.append(count_df)
    multiple_field_l = pd.concat(multiple_field_l)
    multiple_field_l = multiple_field_l[0]
    return multiple_field_l

def get_sw_multifield_info(changes_df_all,all_fields_recombined_all,level=(0,1,2,3,4)):
    '''
    from changes_df, get whether each field has sw on/off, whether it belongs to a neuron with multiple fields
    whether any other field has on/off
    level: get down to uid level
    '''
    has_sw_on = (changes_df_all==1).any(axis=1)
    has_sw_off = (changes_df_all==-1).any(axis=1)

    has_sw_df=pd.concat({'on':has_sw_on,'off':has_sw_off},axis=1)
    has_sw_df = combine_trial_type_with_both(has_sw_df,all_fields_recombined_all)
    
    # multiple_field_l = []
    multiple_field_l=get_multiple_field_l(has_sw_df,level=(0,1,2,3,4))
    # for k,val in has_sw_df.groupby(level=level,sort=False):
    # #     has_sw_df.loc[val.index,'multiple_field']=val.count()>1
    #     multiple_field_l.append(np.ones(val.shape[0],dtype=bool)*(val['on'].count()>1))
    # multiple_field_l = np.concatenate(multiple_field_l)
    has_sw_df['multiple_field'] = multiple_field_l

    # get whether other field sw
    otherrows_sw_all=get_whether_other_field_sw(has_sw_df)
    has_sw_df = pd.concat([has_sw_df,otherrows_sw_all],axis=1)
    
    

    return has_sw_df

def get_odds_ratio(df,x,y):
    '''
    x: variable
    y: event to predict
    both binary
    '''

    df[[x,y]]=df[[x,y]].astype(int)
    gpb=df.groupby([x,y])
    counts=gpb.count().iloc[:,0]
    odds_d = {}
    for var in [1,0]:
        odds=counts.loc[var].loc[1] / counts.loc[var].loc[0]
        odds_d[var]=odds
    odds_ratio = odds_d[1] / odds_d[0]
    return odds_ratio