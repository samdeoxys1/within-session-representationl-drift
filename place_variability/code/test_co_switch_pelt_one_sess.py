import os
import sys
import traceback
import numpy as np
import scipy.io as sio
# Import other required libraries
import pandas as pd
sys.path.append('/mnt/home/szheng/projects/util_code')
sys.path.append('/mnt/home/szheng/projects/nmf_analysis')
import copy,pdb,importlib
import matplotlib.pyplot as plt
import misc
import database 
import data_prep_pyn as dpp
import place_cell_analysis as pa
import place_field_analysis as pf
import test_change_point as tcp
importlib.reload(tcp)

import pf_recombine_central as pfrc
from database import db
import switch_detection_one_session as sdos
import test_co_switch as tcs
import switch_detection_sweep_penalty_one_session as sdspos
# filter db
# subdb = database.db.query("owner=='roman'").sort_values('n_pyr_putative',ascending=False)
subdb = database.db.sort_values('n_pyr_putative',ascending=False)
# db = database.db
# subdb = db.loc[['linearMaze' in x for x in db['behavior']]]
# subdb = db.loc[['[' in x for x in db['behavior']]]



# pf_save_fn = 'place_field_afterheadscan.p'
# pf_shuffle_fn = 'frmap_null_afterheadscan.p'
force_reload = True
PF_FR_KEY = 'fr_peak'#'fr_mean'#
SAVE_DIR=''
SAVE_FN=f'penalty_co_sw_res_afterheadscan_{PF_FR_KEY}.p'#'sw_res_afterheadscan.p'#f'sw_res_afterheadscan_{PF_FR_KEY}.p'
n_shuffle = 1000
# n_change_pts_max_MAX = 5
# MIN_SIZE = 2

def load_preprocess_data(session_path):
    sw_res = sdspos.main(session_path,force_reload=False,load_only=True)
    data={'changes_df_all_pen':{k:sw_res[k]['changes_df'] for k in sw_res.keys()}} # k is penalty, e.g. 0.3 
    
    return data

def analyze_data(data,*args,**kwargs):
    n_shuffle = kwargs.get('n_shuffle',1000)
    min_size = kwargs.get('min_size',sdos.MIN_SIZE)
    # Perform your main analysis
    pval_all_pen = {}
    n_pair_all_n_co_d_all_pen= {}
    n_pair_all_n_co_d_l_all_pen = {}
    changes_df_all_pen = data['changes_df_all_pen']
    for pen,changes_df_all_task in changes_df_all_pen.items():
        pval_all_task, n_pair_all_n_co_d_all_task, n_pair_all_n_co_d_l_all_task =tcs.test_co_switch_all_task(changes_df_all_task,changes_df_shuffle_all_task=None,window_l=[0,1,2],n_co_l=[2,3,4],shuffle_kwargs={'nrepeats':n_shuffle,'min_cpd_win':min_size})
        pval_all_pen[pen] = pval_all_task
        n_pair_all_n_co_d_all_pen[pen] = n_pair_all_n_co_d_all_task
        n_pair_all_n_co_d_l_all_pen[pen]=n_pair_all_n_co_d_l_all_task
    res = {'co_sw_pval':pval_all_pen,
            'n_co_sw':n_pair_all_n_co_d_all_pen,
            'n_co_sw_shuffle':n_pair_all_n_co_d_l_all_pen
            }
    return res

# def save_results(results, session_path, output_folder):
#     # Save your results to a file
#     pass

def main(session_path,test_mode=False,
        analysis_args=[],
        analysis_kwargs={'n_shuffle':n_shuffle},
        dosave=True, save_dir=SAVE_DIR,save_fn=SAVE_FN, force_reload=force_reload,load_only=False,
    ):

    try:
        # create subdir
        save_dir = misc.get_or_create_subdir(session_path,'py_data',save_dir)
        save_fn, res = misc.get_res(save_dir,save_fn,force_reload)
        if (res is not None) or load_only: # load only would skip the computation that follows
            return res
        data = load_preprocess_data(session_path)
        if test_mode:
            # UPDATE SOME PARAMS!!!
            analysis_kwargs['n_shuffle']=2
            dosave=False
            # pass
        
        res = analyze_data(data,*analysis_args,**analysis_kwargs)
        misc.save_res(save_fn,res,dosave=dosave)
        return res
        
    except Exception as e:
        tb_str = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
        tb_str.insert(0, f"Error in session: {session_path}\n")
        sys.stderr.writelines(tb_str)

if __name__ == "__main__":
    sess_ind = int(sys.argv[1])
    test_mode = bool(int(sys.argv[2]))
    session_path = subdb['data_dir_full'][sess_ind]
    print(sess_ind)
    print(test_mode)
    main(session_path, test_mode=test_mode)
