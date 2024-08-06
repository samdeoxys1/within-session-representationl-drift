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
import process_central_arm as pca
from importlib import reload
reload(pca)
from database import db
# filter db
# subdb = database.db.query("owner=='roman'").sort_values('n_pyr_putative',ascending=False)
subdb = database.db.sort_values('n_pyr_putative',ascending=False)
# subdb = db.loc[['[' in x for x in db['behavior']]]
# db = database.db
# subdb = db.loc[['linearMaze' in x for x in db['behavior']]]

SAVE_DIR=''
# SAVE_FN='fr_map.p'

pf_save_fn = 'place_field_afterheadscan.p'
pf_shuffle_fn = 'frmap_null_afterheadscan.p'
force_reload = True
SAVE_FN = 'place_field_afterheadscan_recombine.p' # make it consistent with pf_save_fn by adding_recombine


def load_preprocess_data(session_path):
    pf_res = pf.field_detection_avg_wrapper(session_path, dosave=False,force_reload=False, 
                                        save_fn = pf_save_fn, 
                                        shuffle_fn=pf_shuffle_fn,load_only=True)
    

    all_fields = pd.concat(pf_res['all_fields'],axis=0)
    pf_params = pd.concat(pf_res['params'],axis=0)

    prep_res = dpp.load_spk_beh_df(session_path,force_reload=False,extra_load=dict(sessionPulses='*SessionPulses.Events.mat',filtered='*thetaFiltered.lfp.mat'))
    spk_beh_df=prep_res['spk_beh_df']
    _,spk_beh_df = dpp.group_into_trialtype(spk_beh_df)
    # spk_beh_df,_=dpp.add_lin_binned(spk_beh_df,nbins=100)
    spk_beh_df,_=dpp.add_lin_binned(spk_beh_df,bin_size=2.2,nbins=None)
    cell_cols_d = prep_res['cell_cols_d']
    index_within_to_trial_index_df = dpp.index_within_to_trial_index(spk_beh_df)
    
    data = {'all_fields':all_fields,'pf_params':pf_params,'spk_beh_df':spk_beh_df,'cell_cols_d':cell_cols_d,'index_within_to_trial_index_df':index_within_to_trial_index_df}
    
    return data

def analyze_data(data,*args,**kwargs):
    # Perform your main analysis
    
    pf_params = data['pf_params']
    all_fields = data['all_fields']
    spk_beh_df = data['spk_beh_df']
    pf_par_recombined_alltask, all_fields_recombined_alltask = pca.combine_pf_res(pf_params,all_fields,beh_df=spk_beh_df)

    res = {'all_fields_recombined':all_fields_recombined_alltask,'params_recombined':pf_par_recombined_alltask,
            'all_fields':all_fields,
            'params':pf_params}
    
    return res

# def save_results(results, session_path, output_folder):
#     # Save your results to a file
#     pass

def main(session_path,test_mode=False,
        analysis_args=[],
        analysis_kwargs={},
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
