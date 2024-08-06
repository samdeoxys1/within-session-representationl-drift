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
# FRMAP_FN = f'fr_map_mask_{do_transient_mask}_smallbin.p'
FRMAP_FN = f'fr_map_mask_{do_transient_mask}_smallbin_smallgauss.p'

NREPEATS = 1000
FORCE_RELOAD = True
N_ITER_MAX = 200 # for nmf fitting
N_ROLL_MIN = 10

SAVE_FN = f'shuffle_nmf_error_mask_{do_transient_mask}.p'

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



def main(i,n_roll_min=N_ROLL_MIN,nrepeats=NREPEATS,test_mode=False,
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
        recon_null_l_all = {}
        for ii,(k,val) in tqdm.tqdm(enumerate(gpb)):
            if ii<n_max:
                X_df=val.loc[k].dropna(axis=1,how='all')
                res_one,recon_null_l=rba.nmf_pick_rank_shuffle_one_neuron(X_df,rank_l=range(1,5),nrepeats=nrepeats,alpha=0.05,n_roll_min=n_roll_min,n_iter_max=n_iter_max)
                recon_null_l_all[k] = recon_null_l
                res_all[k] = res_one
        res_all = pd.concat(res_all,axis=0)
        recon_null_l_all = pd.DataFrame(recon_null_l_all)
        res={'summary':res_all,'recon_null':recon_null_l_all}

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


