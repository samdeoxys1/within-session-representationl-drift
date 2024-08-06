'''
for one exp, do the nmf recon error vs shuffle, for each neuron, across novel-fam
'''

import numpy as np
import scipy
import matplotlib.pyplot as plt
import sys,os,copy,pdb,importlib,pickle
from importlib import reload
import pandas as pd


import get_thomas_cell_metrics as gtcm
import reuse_block_analysis as rba

sys.path.append('/mnt/home/szheng/projects/place_variability/code')
sys.path.append('/mnt/home/szheng/projects/util_code')
import traceback
import misc
import tqdm
import database 
db = database.thomas_18_db
db_grouped = db.groupby(['region','exp_ind']).mean().index # group on the level of region and exp

ROOT = "/mnt/home/szheng/ceph/ad/thomas_data/180301_DG_CA3_CA1"
do_transient_mask = True#False#True#False
# SAVE_FN_ONE = 'fr_map.p'
# SAVE_FN_ALL = 'fr_map_all.p'
# if do_transient_mask==False:
FRMAP_FN = f'fr_map_mask_{do_transient_mask}_smallbin.p'


FORCE_RELOAD = True
N_ITER_MAX = 100 # for nmf fitting


SAVE_FN = f'unmf_withinoutsidediff_{do_transient_mask}.p'

def load_data():
    '''
    load fr_map
    '''
    fn = os.path.join(ROOT,FRMAP_FN)
    fr_map_all=pickle.load(open(fn,'rb'))
    fr_map_trial_df_all = fr_map_all['fr_map_trial_df_all']
    frmap = fr_map_all['fr_map_all']
    fr_map_trial_all = fr_map_all['fr_map_trial_all']
    occu_map = fr_map_all['occu_map_all']

    # reshape fr_map_trial_df_all to make it easier to work with
    fr_map_trial_df_all_day =fr_map_trial_df_all.unstack(level=2).swaplevel(0,1,axis=1).sort_index(axis=1) # day in column
    # fr_map_trial_df_all_per_uid=fr_map_trial_df_all.unstack(level=(2,3)).swaplevel(0,1,axis=1).swaplevel(1,2,axis=1).sort_index(axis=1).dropna(axis=1,how='all')
    frmap_all_day = frmap.unstack(level=2).swaplevel(0,1,axis=1).sort_index(level=0,axis=1)

    load_res={'fr_map_trial_df_all_day':fr_map_trial_df_all_day, 
    'frmap_all_day':fr_map_trial_df_all_day}
    return load_res



def main(i,test_mode=False,
            save_dir = None,
            save_fn = SAVE_FN,
            force_reload=FORCE_RELOAD,dosave=True,load_only=False,n_iter_max=N_ITER_MAX,
            ):
    try:
        # select sub data
        region,exp = db_grouped[i]
        ddf_day = db.loc[(db['region']==region)&(db['exp_ind']==exp)]['data_dir_full'].iloc[0]
        session_path=ddf_exp = os.path.dirname(ddf_day) # exp level
        exp=int(exp)

        if save_dir is None:
            save_dir = ddf_exp
        save_fn, res = misc.get_res(save_dir,save_fn,force_reload)
        if (res is not None) or load_only: # load only would skip the computation that follows
            return res

        # load data
        load_res = load_data()
        fr_map_trial_df_all_day =load_res['fr_map_trial_df_all_day']
        frmap_all_day = load_res['frmap_all_day']
        fr_map_trial_df_all_day_sub = fr_map_trial_df_all_day.loc[(region,exp),:]
        if test_mode:
            n_max = 3
        else:
            n_max = 1000000

        # do analysis
        gpb = fr_map_trial_df_all_day_sub.groupby(level=(0,1)) # isnovel, uid
        res_all = {}
        pre_day_activation_d_all = {}
        W_hat_d_all={}
        H_hat_d_all={}
        field_range_d_all={}
        field_range_ma_d_all={}
        for ii,(k,val) in enumerate(tqdm.tqdm(gpb)):
            if ii<n_max:
                X_df_=val.loc[k].dropna(axis=1,how='all')
                try:
                    pre_day_activation_d,W_hat_d,H_hat_d,field_range_d,field_range_ma_d = rba.fit_unmf_and_get_activation_all_comparisons_onecell(X_df_)
                    pre_day_activation_d_all[k] = pre_day_activation_d
                    W_hat_d_all[k] = W_hat_d
                    H_hat_d_all[k] = H_hat_d
                    field_range_d_all[k] = field_range_d
                    field_range_ma_d_all[k] = field_range_ma_d
                except:
                    pass
        
        pre_day_activation_d_all = pd.concat(pre_day_activation_d_all,axis=0)
        W_hat_d_all = pd.concat(W_hat_d_all,axis=0)
        H_hat_d_all = pd.concat(H_hat_d_all,axis=0)
        field_range_d_all = pd.concat(field_range_d_all,axis=0)
        field_range_ma_d_all = pd.concat(field_range_ma_d_all,axis=0)

        res={'pre_day_activation':pre_day_activation_d_all,
            'W':W_hat_d_all,'H':H_hat_d_all,'field_range':field_range_d_all,
            'field_range_ma':field_range_ma_d_all
        }

        # save
        misc.save_res(save_fn,res,dosave=dosave)

        return res
    except Exception as e:
        tb_str = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
        tb_str.insert(0, f"Error in session: {session_path}\n")
        sys.stderr.writelines(tb_str)


if __name__ == "__main__":
    region_exp_ind = int(sys.argv[1])
    test_mode = bool(int(sys.argv[2]))
    
    print(region_exp_ind)
    print(test_mode)
    main(region_exp_ind, test_mode=test_mode)


