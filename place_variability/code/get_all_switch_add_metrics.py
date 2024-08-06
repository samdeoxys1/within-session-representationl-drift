import os
import sys
import traceback
import numpy as np
import scipy.io as sio
# Import other required libraries
import pandas as pd
sys.path.append('/mnt/home/szheng/projects/util_code')
sys.path.append('/mnt/home/szheng/projects/nmf_analysis')
import copy,pdb
import matplotlib.pyplot as plt
import misc
import database 
import data_prep_pyn as dpp
import place_cell_analysis as pa
import place_field_analysis as pf

import switch_metrics as sm
from importlib import reload
reload(sm)

# filter db
# subdb = database.db.query("owner=='roman'").sort_values('n_pyr_putative',ascending=False)
subdb = database.db.sort_values('n_pyr_putative',ascending=False)


SAVE_DIR='switch_analysis'
PF_LOC_KEY = 'com'#'peak'#
PF_FR_KEY = 'fr_peak'#'fr_mean' # 'fr_peak
SAVE_FN='all_sw_info.p'#f'all_sw_info_{PF_FR_KEY}.p'#

import pf_recombine_central as pfrc
import switch_detection_one_session as sdos

def load_preprocess_data(session_path):
    prep_res = dpp.load_spk_beh_df(session_path,force_reload=False,extra_load=dict(sessionPulses='*SessionPulses.Events.mat',filtered='*thetaFiltered.lfp.mat'))
    spk_beh_df=prep_res['spk_beh_df']
    _,spk_beh_df = dpp.group_into_trialtype(spk_beh_df)
    # spk_beh_df,_=dpp.add_lin_binned(spk_beh_df,nbins=100)
    spk_beh_df,_=dpp.add_lin_binned(spk_beh_df,bin_size=2.2,nbins=None)
    cell_cols_d = prep_res['cell_cols_d']
    
    
    pf_res_recombine = pfrc.main(session_path,force_reload=False)

    all_fields_recombined=pf_res_recombine['all_fields_recombined']
    pf_params_recombined = pf_res_recombine['params_recombined']
    sw_res = sdos.main(session_path,force_reload=False,load_only=True)

    data = {'spk_beh_df':spk_beh_df,'cell_cols_d':cell_cols_d,'pf_params_recombined':pf_params_recombined,
            'sw_res':sw_res            
            }
    return data

def analyze_data(data,*args,**kwargs):
    # Perform your main analysis
    
    spk_beh_df = data['spk_beh_df']
    sw_res = data['sw_res']
    pf_params_recombined = data['pf_params_recombined']
    pf_loc_key = kwargs.get('pf_loc_key',PF_LOC_KEY)
    pf_fr_key = kwargs.get('pf_fr_key',PF_FR_KEY)

    all_sw_d, all_sw_with_metrics_d = sm.get_all_sw_add_metrics_all_tasks(sw_res,pf_params_recombined,spk_beh_df,pf_loc_key=pf_loc_key,pf_fr_key=pf_fr_key)

    res = {'all_sw_d':all_sw_d,'all_sw_with_metrics_d':all_sw_with_metrics_d}
    return res

# def save_results(results, session_path, output_folder):
#     # Save your results to a file
#     pass

def main(session_path,test_mode=False,
        analysis_args=[],
        analysis_kwargs={'pf_loc_key':PF_LOC_KEY,'pf_fr_key':PF_FR_KEY},
        dosave=True, save_dir=SAVE_DIR,save_fn=SAVE_FN, force_reload=False,load_only=False,
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
            pass
        
        res = analyze_data(data,*analysis_args,**analysis_kwargs)
        misc.save_res(save_fn,res,dosave=dosave)
        return res
        
    except Exception as e:
        tb_str = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
        tb_str.insert(0, f"Error in session: {session_path}\n")
        sys.stderr.writelines(tb_str)

if __name__ == "__main__":
    sess_ind = int(sys.argv[1])
    test_mode = bool(sys.argv[2])
    session_path = subdb['data_dir_full'][sess_ind]
    
    main(session_path, test_mode=test_mode)
