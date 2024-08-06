'''
for getting the across session within session difference in correlation for each cell
'''

import numpy as np
import scipy
import matplotlib.pyplot as plt
import sys,os,copy,pdb,importlib,pickle
from importlib import reload
import pandas as pd



import get_thomas_cell_metrics as gtcm

sys.path.append('/mnt/home/szheng/projects/place_variability/code')
sys.path.append('/mnt/home/szheng/projects/util_code')
import traceback
import misc
import database 
db = database.thomas_18_db
db_grouped = db.groupby(['region','exp_ind']).mean().index # group on the level of region and exp

ROOT = "/mnt/home/szheng/ceph/ad/thomas_data/180301_DG_CA3_CA1"
do_transient_mask = True#False#True#False
# SAVE_FN_ONE = 'fr_map.p'
# SAVE_FN_ALL = 'fr_map_all.p'
# if do_transient_mask==False:
FRMAP_FN = f'fr_map_all_mask_{do_transient_mask}.p'

N_TR = 5
NREPEATS = 400
FORCE_RELOAD = True


SAVE_FN = f'across_within_diff_per_cell_mask_{do_transient_mask}.p'

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



def main(i,n_tr=N_TR,n_roll_min=2,nrepeats=NREPEATS,test_mode=False,
            save_dir = None,
            save_fn = SAVE_FN,
            force_reload=FORCE_RELOAD,dosave=True,load_only=False
            ):
    try:
        # select sub data
        region,exp = db_grouped[i]
        ddf_day = db.loc[(db['region']==region)&(db['exp_ind']==exp)]['data_dir_full'].iloc[0]
        ddf_exp = os.path.dirname(ddf_day) # exp level
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

        if test_mode:
            n_max = 3
        else:
            n_max = 1000000

        # do analysis
        beg_end_corr_diff_df_famnov = {}
        pval_d_famnov = {}
        diff_l_sh_l_all_famnov = {}
        for isnovel in [0,1]:
            fr_map_trial_df_all_day_sub = fr_map_trial_df_all_day.loc[(region,exp,isnovel),:].dropna(axis=1,how='all')
            beg_end_corr_diff_df = gtcm.get_end_beg_diff_minus_beg_end_same_all(fr_map_trial_df_all_day_sub,cell_level=0,n_tr = n_tr)
            beg_end_corr_diff_df_famnov[isnovel] = beg_end_corr_diff_df
            pval_d, diff_l_sh_l_all = gtcm.shuffle_test_end_beg_diff_minus_beg_end_same_all(fr_map_trial_df_all_day_sub,cell_level=0,n_tr = n_tr,n_roll_min=n_roll_min,nrepeats=NREPEATS,n_max=n_max)
            pval_d_famnov[isnovel] = pval_d
            diff_l_sh_l_all_famnov[isnovel] = diff_l_sh_l_all
        beg_end_corr_diff_df_famnov = pd.concat(beg_end_corr_diff_df_famnov,axis=0)
        pval_d_famnov = pd.concat(pval_d_famnov,axis=0)
        diff_l_sh_l_all_famnov = pd.concat(diff_l_sh_l_all_famnov,axis=0)
        res = {'beg_end_corr_diff':beg_end_corr_diff_df_famnov,
                'pval':pval_d_famnov,
                'diff_shuffle':diff_l_sh_l_all_famnov
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


