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
import switch_detection_one_session as sdos
importlib.reload(tcp)
import test_co_switch as tcs

import pf_recombine_central as pfrc
from database import db
# filter db
# subdb = database.db.query("owner=='roman'").sort_values('n_pyr_putative',ascending=False)
subdb = database.db.sort_values('n_pyr_putative',ascending=False)
# db = database.db
# subdb = db.loc[['linearMaze' in x for x in db['behavior']]]
# subdb = db.loc[['[' in x for x in db['behavior']]]



# pf_save_fn = 'place_field_afterheadscan.p'
# pf_shuffle_fn = 'frmap_null_afterheadscan.p'
force_reload = True
# pf_fr_key = 'fr_peak'
SAVE_DIR=''
SAVE_FN='shuffle_all_sw_afterheadscan.p'#'shuffle_all_sw_afterheadscan_fr_mean.p'#'sw_res_afterheadscan.p'
n_shuffle = 1000


def load_preprocess_data(session_path):
    sw_res = sdos.main(session_path,force_reload=False,load_only=True)
    
    changes_df = sw_res['changes_df']
    pf_res_recombine = pfrc.main(session_path,force_reload=False,load_only=True)
    pf_params_recombined = pf_res_recombine['params_recombined']
    res=dpp.load_spk_beh_df(session_path,load_only=True)
    spk_beh_df = res['spk_beh_df']

    data = {'changes_df':changes_df,'spk_beh_df':spk_beh_df,'pf_params_recombined':pf_params_recombined}
    return data

def analyze_data(data,*args,**kwargs):
    nrepeats = kwargs.get('nrepeats',n_shuffle)
    n_change_pts_max_MAX = kwargs.get('n_change_pts_max_MAX',5)
    min_cpd_win = kwargs.get('min_cpd_win',sdos.MIN_SIZE)
    # Perform your main analysis
    changes_df = data['changes_df']
    pf_params_recombined=data['pf_params_recombined']
    spk_beh_df = data['spk_beh_df']
    res = tcs.gen_circular_shuffle_trialtype_seperated_get_all_sw(changes_df,pf_params_recombined,spk_beh_df,nrepeats=nrepeats,min_cpd_win=min_cpd_win)
    
    return res

# def save_results(results, session_path, output_folder):
#     # Save your results to a file
#     pass

def main(session_path,test_mode=False,
        analysis_args=[],
        analysis_kwargs={'nrepeats':n_shuffle},
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
            analysis_kwargs['nrepeats']=2
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
