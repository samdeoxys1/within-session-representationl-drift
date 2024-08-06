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

# filter db
# subdb = database.db.query("owner=='roman'").sort_values('n_pyr_putative',ascending=False)
subdb = database.db.sort_values('n_pyr_putative',ascending=False)


SAVE_DIR=''
SAVE_FN='place_field_afterheadscan.p'
SHUFFLE_SAVE_FN = 'frmap_null_afterheadscan.p'


def load_preprocess_data(session_path):
    prep_res = dpp.load_spk_beh_df(session_path,force_reload=False,extra_load=dict(sessionPulses='*SessionPulses.Events.mat',filtered='*thetaFiltered.lfp.mat'))
    spk_beh_df=prep_res['spk_beh_df']
    _,spk_beh_df = dpp.group_into_trialtype(spk_beh_df)
    # spk_beh_df,_=dpp.add_lin_binned(spk_beh_df,nbins=100)
    spk_beh_df,_=dpp.add_lin_binned(spk_beh_df,bin_size=2.2,nbins=None)
    cell_cols_d = prep_res['cell_cols_d']
    
    data = {'spk_beh_df':spk_beh_df,'cell_cols_d':cell_cols_d}
    return data

def analyze_data(data,*args,**kwargs):
    # Perform your main analysis
    cell_cols = data['cell_cols_d']['pyr']
    spk_beh_df = data['spk_beh_df']
    speed_thresh=0.5
    fr_map_dict=pa.get_fr_map_trial(spk_beh_df,cell_cols,gauss_width=2.5,trialtype_key='trial_type',speed_key='directed_locomotion',speed_thresh=speed_thresh,order=['smooth','divide','average'])
    fr_map_d={k:val[0] for k,val in fr_map_dict.items()}
    fr_map_df_all = pd.concat(fr_map_d,axis=0)

    fr_map_trial_dict=pa.get_fr_map_trial(spk_beh_df,cell_cols,gauss_width=2.5,trialtype_key='trial_type',speed_key='directed_locomotion',speed_thresh=speed_thresh,order=['smooth','divide'])
    fr_map_trial_d = {k:val[0] for k,val in fr_map_trial_dict.items()}
    fr_map_trial_df_all=pd.concat({k:pf.fr_map_trial_to_df(fr_map_trial_d[k],cell_cols) for k in fr_map_dict.keys()},axis=0)


    res = {'fr_map':fr_map_df_all,'fr_map_trial':fr_map_trial_d,'fr_map_trial_df':fr_map_trial_df_all}
    return res

# def save_results(results, session_path, output_folder):
#     # Save your results to a file
#     pass

def main(session_path,test_mode=False,
        analysis_args=[],
        analysis_kwargs={},
        dosave=True, save_dir=SAVE_DIR,save_fn=SAVE_FN, shuffle_fn=SHUFFLE_SAVE_FN,force_reload=True,load_only=False,
        shuffle_force_reload = True,
    ):

    try:
        # create subdir
        save_dir = misc.get_or_create_subdir(session_path,'py_data',save_dir)
        save_fn, res = misc.get_res(save_dir,save_fn,force_reload)
        if (res is not None) or load_only: # load only would skip the computation that follows
            return res
        # data = load_preprocess_data(session_path)
        if test_mode:
            # UPDATE SOME PARAMS!!!
            pass
        
        res=pf.field_detection_avg_wrapper(session_path, dosave=dosave,force_reload=force_reload,nbins = 100, bin_size=2.2,
                                        save_fn = save_fn, 
                                        shuffle_fn=shuffle_fn,
                                        shuffle_force_reload=shuffle_force_reload,
                                        smth_in_bin=2.5, speed_thresh=0.5,speed_key='directed_locomotion',load_only=load_only,            
                                        )
        # res = analyze_data(data,*analysis_args,**analysis_kwargs)
        # misc.save_res(save_fn,res,dosave=dosave)
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
