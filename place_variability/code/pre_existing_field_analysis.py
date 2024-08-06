import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import sys,os,copy,pdb,importlib,pickle
from importlib import reload
import pandas as pd

sys.path.append('/mnt/home/szheng/projects/place_variability/code')

import place_field_detection_thomas as pfdt
reload(pfdt)

# this script works on the aggregated data level

def get_npos_per_sess(fr_map_trial_df_all):
    # get n pos per sess
    gpb=fr_map_trial_df_all.groupby(level=(0,1,2))
    npos_per_sess = {}
    for k,val in gpb:
        npos = val.index.get_level_values(-1).unique().max()+1
        npos_per_sess[k] = npos
    npos_per_sess=pd.Series(npos_per_sess)

    return npos_per_sess # series;  (ani,sess,task_index)

def extend_field_range_all(all_fields_recombined_all,fr_map_trial_df_all):
    '''
    the core is pfdt.get_window_outside; using the default arguments (extending 10% of the track length)
    '''

    gpb=all_fields_recombined_all.groupby(level=(0,1,2,4))
    all_fields_recombined_all_windowextended = []#{}
    npos_per_sess = get_npos_per_sess(fr_map_trial_df_all)
    for k,val in gpb: # loop over cells

        val = val.droplevel((0,1,2))
        nfields_orig = val.shape[0]
        npos = npos_per_sess.loc[k[:-1]]
        val_win_extended_both = []
        for tt in [0,1]: # over trialtypes, plus both
            
            # the fields contain ones that occur in either both or tt 
            if (tt in val.index.get_level_values(0)) and ('both' in val.index.get_level_values(0)):
                val_onett_and_both=val.loc[(['both',tt]),:]
            elif (tt in val.index.get_level_values(0)) and ('both' not in val.index.get_level_values(0)):
                val_onett_and_both=val.loc[([tt]),:]
            elif (tt not in val.index.get_level_values(0)) and ('both' in val.index.get_level_values(0)):
                val_onett_and_both=val.loc[(['both']),:]
            else:
                val_onett_and_both=None
            if val_onett_and_both is not None: # extend fields

                val_reset=val_onett_and_both.sort_values('peak').reset_index()
                val_win_extended=pfdt.get_window_outside(val_reset,npos=npos)
                val_win_extended=val_win_extended.set_index(['level_0','level_1','field_index'])
                val_win_extended_both.append(val_win_extended)

        
        if len(val_win_extended_both) > 0: # if there's any field
            val_win_extended_both_df = pd.concat(val_win_extended_both,axis=0) # combine the extended results for both trial types
            # then need to trim the Both fields that are double counted
            # for the Both fields, contract the windows obtained from excluding fields from 0 and 1 trialtypes
            if 'both' in val_win_extended_both_df.index.get_level_values(0):
                gpb_for_both=val_win_extended_both_df.loc[['both']].groupby(level='field_index') # group by field index for the both fields
                remaining_both = [] 
                for _,valval in gpb_for_both: # for each both fields, select the max of the start and min of the end
                    row=valval.iloc[0]
                    row['window_start']=valval['window_start'].max()
                    row['window_end']=valval['window_end'].min()
                    remaining_both.append(row.to_frame().T)
                remaining_both = pd.concat(remaining_both,axis=0)
                # then put the both fields back together with the non both fields.
                not_both_ma=val_win_extended_both_df.index.get_level_values(0)!='both' 
                val_win_extended_both_df_processed = pd.concat([val_win_extended_both_df.loc[not_both_ma],remaining_both],axis=0)
            else: # if no both fields, don't have to do the contraction and reselection
                val_win_extended_both_df_processed = val_win_extended_both_df
        
        nfields_after = val_win_extended_both_df_processed.shape[0]
        if nfields_after!=nfields_orig:
            break
            
        # update the index
        current_index = val_win_extended_both_df_processed.index
        new_tuples = [k[:3] + idx for idx in current_index]
        new_index = pd.MultiIndex.from_tuples(new_tuples)
        val_win_extended_both_df_processed.index = new_index
    #     all_fields_recombined_all_windowextended[k[:3]] = val_win_extended_both_df_processed

        all_fields_recombined_all_windowextended.append(val_win_extended_both_df_processed)
        
    all_fields_recombined_all_windowextended = pd.concat(all_fields_recombined_all_windowextended,axis=0)
    return all_fields_recombined_all_windowextended




