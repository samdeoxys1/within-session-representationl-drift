import numpy as np
import scipy
import matplotlib.pyplot as plt
import sys,os,copy,pdb,importlib
from importlib import reload
import traceback
import pandas as pd
sys.path.append('/mnt/home/szheng/projects/util_code/')
sys.path.append('/mnt/home/szheng/projects/nmf_analysis')
sys.path.append('/mnt/home/szheng/projects/nmf_analysis/scripts')
sys.path.append('/mnt/home/szheng/projects/place_variability/code/thomas_18_data/prep_thomas_one_region.py')

import misc
import database 
import data_prep_pyn as dpp
import place_cell_analysis as pa
import place_field_analysis as pf
import prep_thomas_one_region as ptor

import pickle

import fr_map_one_session as fmos

ROOT = "/mnt/home/szheng/ceph/ad/thomas_data/180301_DG_CA3_CA1"
do_transient_mask = True#False#True#False
# SAVE_FN_ONE = 'fr_map.p'
# SAVE_FN_ALL = 'fr_map_all.p'
# if do_transient_mask==False:
SAVE_FN_ONE = f'fr_map_mask_{do_transient_mask}.p'
SAVE_FN_ALL = f'fr_map_all_mask_{do_transient_mask}.p'

def get_fr_map_one_day(ddf,force_reload=False,load_only=False,dosave=True,**kwargs):
    try:
        bin_size=kwargs.get('bin_size',0.026 * 2) # don't know why used to be 0.011, 
        gauss_width = kwargs.get("gauss_width",1.25) # 2.5
        save_fn_one = kwargs.get('save_fn_one',SAVE_FN_ONE)
        save_dir = ddf
        save_fn_one, fr_map_res = misc.get_res(save_dir,save_fn_one,force_reload)
        if (fr_map_res is not None) or load_only: # load only would skip the computation that follows
            return fr_map_res

        # prep_fn = os.path.join(ddf,'preprocessed.p')
        prep_fn = os.path.join(ddf,ptor.SAVE_FN)
        prep_res = pickle.load(open(prep_fn,'rb'))

        spk_beh_df = prep_res['spk_beh_df']
        spk_beh_df,_=dpp.add_lin_binned(spk_beh_df,bin_size=bin_size,nbins=None)
        prep_res['spk_beh_df'] = spk_beh_df
        # pdb.set_trace()
        fr_map_res=fmos.analyze_data(prep_res,gauss_width=gauss_width)
        misc.save_res(save_fn_one,fr_map_res,dosave=dosave)

        return fr_map_res

    except Exception as e:
        tb_str = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
        tb_str.insert(0, f"Error in session: {ddf}\n")
        sys.stderr.writelines(tb_str)

def rectify_df(df):
    df_v = df.values
    df_v[df_v<0]=0.
    df = pd.DataFrame(df_v,index=df.index,columns=df.columns)
    return df


def main(force_reload=False,load_only=False,dosave=True,**kwargs):
    
    save_dir = ROOT
    save_fn_all = kwargs.get('save_fn_all',SAVE_FN_ALL)
    save_fn_all, fr_map_res_all = misc.get_res(save_dir,save_fn_all,force_reload)
    if (fr_map_res_all is not None) or load_only: # load only would skip the computation that follows
        return fr_map_res_all

    thomas_db=database.thomas_18_db
    # thomas_db=database.get_thomas_180301_DG_CA3_CA1_db(dosave=False)

    fr_map_d = {}
    fr_map_trial_d = {}
    fr_map_trial_df_d = {}
    fr_map_trial_df_pyr_combined_d = {}
    occu_map_d = {}

    for _,row in thomas_db.iterrows():
        ddf = row['data_dir_full']
        
        fr_map_res = get_fr_map_one_day(ddf,
                        force_reload=force_reload,load_only=load_only,dosave=dosave,**kwargs
                            )
        
        if fr_map_res is not None:
            region = row['region']
            # exp_ind = row['exp_ind']
            # day_ind = row['day_ind']
            exp_ind = int(row['exp_ind'])
            day_ind = int(row['day_ind'])
            key = (region, exp_ind, day_ind)
            fr_map_d[key] = fr_map_res['fr_map']
            occu_map_d[key] = fr_map_res['occu_map']
            fr_map_trial_d[key] = fr_map_res['fr_map_trial']
            fr_map_trial_df_d[key] = fr_map_res['fr_map_trial_df']
            fr_map_trial_df_pyr_combined_d[key] = fr_map_res['fr_map_trial_df_pyr_combined']
    fr_map_d = pd.concat(fr_map_d,axis=0)
    fr_map_d = rectify_df(fr_map_d)
    fr_map_trial_df_d = pd.concat(fr_map_trial_df_d,axis=0)
    fr_map_trial_df_d = rectify_df(fr_map_trial_df_d)
    occu_map_d = pd.concat(occu_map_d,axis=0).unstack()
    fr_map_trial_df_pyr_combined_d = pd.concat(fr_map_trial_df_pyr_combined_d,axis=0)
    fr_map_trial_df_pyr_combined_d = rectify_df(fr_map_trial_df_pyr_combined_d)

    res = {'fr_map_all':fr_map_d,
            'fr_map_trial_all':fr_map_trial_d,
            'fr_map_trial_df_all':fr_map_trial_df_d,
            'occu_map_all':occu_map_d,
            'fr_map_trial_df_pyr_combined_all':fr_map_trial_df_pyr_combined_d,
            }
    
    misc.save_res(save_fn_all,res,dosave=dosave)


    return res
    
