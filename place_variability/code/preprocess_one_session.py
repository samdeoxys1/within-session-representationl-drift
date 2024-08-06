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

# filter db
# subdb = database.db.query("owner=='roman'").sort_values('n_pyr_putative',ascending=False)
subdb = database.db.sort_values('n_pyr_putative',ascending=False)


SAVE_DIR='[replace]'
SAVE_FN='[replace].p'


def load_preprocess_data(session_path):
    
    pass

def analyze_data(data,*args,**kwargs):
    # Perform your main analysis
    pass

def save_results(results, session_path, output_folder):
    # Save your results to a file
    pass

def main(session_path,test_mode=False,
        analysis_args=[],
        analysis_kwargs={},
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
        # results = analyze_data(data,*analysis_args,**analysis_kwargs)
        # misc.save_res(save_fn,res,dosave=dosave)
        prep_res = dpp.load_spk_beh_df(session_path,force_reload=True,extra_load=dict(
                                                                            sessionPulses='*SessionPulses.Events.mat',
                                                                            filtered='*thetaFiltered.lfp.mat'))
    except Exception as e:
        tb_str = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
        tb_str.insert(0, f"Error in session: {session_path}\n")
        sys.stderr.writelines(tb_str)

if __name__ == "__main__":
    sess_ind = int(sys.argv[1])
    test_mode = bool(sys.argv[2])
    session_path = subdb['data_dir_full'][sess_ind]
    
    main(session_path, test_mode=test_mode)
