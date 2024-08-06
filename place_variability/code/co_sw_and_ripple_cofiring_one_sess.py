import os
import sys
import traceback
import numpy as np
import scipy.io as sio
# Import other required libraries
import pandas as pd
sys.path.append('/mnt/home/szheng/projects/util_code')
sys.path.append('/mnt/home/szheng/projects/nmf_analysis')
sys.path.append('/mnt/home/szheng/projects/place_variability/code')
import copy,pdb,importlib
from importlib import reload
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
import misc
import database 
import data_prep_pyn as dpp
import place_cell_analysis as pa
import place_field_analysis as pf
import test_change_point as tcp
importlib.reload(tcp)
import pairwise_analysis as pwa

import pf_recombine_central as pfrc
import fr_map_one_session as fmos
import switch_detection_one_session as sdos
reload(sdos)
import get_all_switch_add_metrics as gasam
reload(gasam)

import pairwise_analysis as pwa
reload(pwa)
import switch_metrics as sm
reload(sm)
import test_co_switch as tcs
reload(tcs)
import shuffle_changes_df_one_session as scdos
import preprocess as prep

import preprocess_one_session as prepos
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
# PF_FR_KEY = 'fr_mean'#'fr_peak'
SAVE_DIR=''
# SAVE_FN=f'sw_res_afterheadscan_{PF_FR_KEY}.p'
FIG_SAVE_FN = 'ripple_cofiring_co_sw_minus_no_co_sw_vs_shuffle.pdf'
# n_shuffle = 1000
# n_change_pts_max_MAX = 5
# MIN_SIZE = 2
task_index = 0
DATA_OR_FIG_DIR = 'py_figures'

def load_preprocess_data(data_dir_full):
    
    pf_res_recombine = pfrc.main(data_dir_full,force_reload=False,load_only=True)
    sw_res = sdos.main(data_dir_full,force_reload=False,load_only=True)
    changes_df_one = sw_res['changes_df'].loc[task_index]
    sw_info_res=gasam.main(data_dir_full,force_reload=False,load_only=True)
    all_sw_d_one = sw_info_res['all_sw_d'].loc[task_index]
    pf_params_recombined = pf_res_recombine['params_recombined']
    
    res=dpp.load_spk_beh_df(data_dir_full,load_only=True)
    spk_beh_df = res['spk_beh_df']
    trial_index_to_index_within_df = dpp.trial_index_to_index_within_trialtype(spk_beh_df)
    
    shuffle_all_sw_one=scdos.main(data_dir_full,test_mode=False,dosave=False,force_reload=False,load_only=True)

    mat_to_return=prep.load_stuff(data_dir_full)
    ripples = mat_to_return['ripples']
    cell_metrics=mat_to_return['cell_metrics']
    mergepoints = mat_to_return['mergepoints']

    res_all_epochs_d, ripple_time_ints_epochs_d = prep.get_spike_count_rate_participation_in_ripple_all(cell_metrics,
                                                                                                    ripples,
                                                                                                    mergepoints,
                                                                                                   )
    ripple_sim_d = pwa.get_all_ripple_pairwise_sim(cell_metrics,ripples,mergepoints)

    data = {'changes_df_one':changes_df_one,'all_sw_d_one':all_sw_d_one,'spk_beh_df':spk_beh_df,
            'trial_index_to_index_within_df':trial_index_to_index_within_df,
            'shuffle_all_sw_one':shuffle_all_sw_one,
            'ripple_sim_d':ripple_sim_d,'mergepoints':mergepoints
            }
    return data

def analyze_data(data,*args,**kwargs):
    onoff_str_d = {1:'on',-1:'off','both':'both'}
    
    # Perform your main analysis
    changes_df_one = data['changes_df_one']
    all_sw_d_one = data['all_sw_d_one']
    shuffle_all_sw_one = data['shuffle_all_sw_one']
    n_shuffle = kwargs.get('n_shuffle',len(shuffle_all_sw_one))
    mergepoints = data['mergepoints']
    ripple_sim_d = data['ripple_sim_d']

    sw_sim_allconfig_d = {}
    fig_save_fn_full = kwargs.get('fig_save_fn_full',FIG_SAVE_FN)
    with PdfPages(fig_save_fn_full) as pdf:
        # outer loop: different coswitch definitiion category
        for diff_key,win_l in zip(['trial_index','time'],[[0,1,2],[1,30,60]]):
            sw_sim_allconfig_d[diff_key] = {}
            sw_sim_allonoff_shuffle = []
            # diff_key = 'trial_index'#'time'#
            # win_l = [0,1,2]#[1,30]#[0,1,2]
            sw_sim_allonoff,time_diff = pwa.get_sw_sim(all_sw_d_one,diff_key=diff_key,win_l=win_l,decay_rate_l=[])
            sw_sim_allconfig_d[diff_key]['data'] = sw_sim_allonoff

            for ii,allsw in enumerate(shuffle_all_sw_one[:n_shuffle]):
                ss, td=pwa.get_sw_sim(allsw.loc[task_index],diff_key=diff_key,win_l=win_l,decay_rate_l=[])
            #     sw_sim_allonoff_shuffle[ii]=sw_sim_allonoff
                sw_sim_allonoff_shuffle.append(ss)
            sw_sim_allconfig_d[diff_key]['shuffle'] = sw_sim_allonoff_shuffle

            # inner loop: different window
            for win in win_l:
                sw_key=f'within_{win}'
                nplots = mergepoints['timestamps'].shape[0]
                fig,axs = plt.subplots(3,nplots,figsize=(6*nplots,4*3))
                # each figure: 
                for ii,onoff in enumerate([1,-1,'both']):
                    onoff_str=onoff_str_d[onoff]
                    epoch_name = ['pre-sleep ripples','behavior ripples','post-sleep ripples']
                #     for jj,epoch in enumerate([0,1,2]):
                    for jj,epoch in enumerate(range(nplots)):
                        label_sim_one = ripple_sim_d.loc['ripple_only','count_in_interval',epoch]
                        co_sw_sim_one = sw_sim_allonoff.loc[onoff,sw_key]
                    
                        ax=axs[ii,jj]
                        diff_data,diff_data_sh_l,pval,fig,ax=pwa.shuffle_test_label_switch_diff_plot(label_sim_one,co_sw_sim_one,sw_sim_allonoff_shuffle,onoff=onoff,sw_key=sw_key,
                                                            fig=fig,ax=ax,doplot=True
                                                        )
                        if jj <=len(epoch_name)-1:
                            title = epoch_name[jj]
                        else:
                            title = f"epoch_{jj}"
                        ax.set_title(title)
                        if jj==0:
                            ax.set_ylabel(onoff_str)
                suptitle = f'{diff_key}, window={win}'
                fig.suptitle(suptitle,fontsize=20)
                plt.tight_layout()
                pdf.savefig(fig,bbox_inches='tight')
                plt.close(fig)
    print(f'{fig_save_fn_full} saved!')
        
    # return res

# def save_results(results, session_path, output_folder):
#     # Save your results to a file
#     pass

def main(session_path,test_mode=False,
        analysis_args=[],
        analysis_kwargs={},
        dosave=True, save_dir=SAVE_DIR,save_fn=FIG_SAVE_FN, force_reload=force_reload,load_only=False,
    ):

    try:
        # create subdir
        save_dir = misc.get_or_create_subdir(session_path,DATA_OR_FIG_DIR,save_dir)
        save_fn, res = misc.get_res(save_dir,save_fn,force_reload)
        if (res is not None) or load_only: # load only would skip the computation that follows
            return res
        data = load_preprocess_data(session_path)
        analysis_kwargs['fig_save_fn_full']=save_fn
        if test_mode:
            # UPDATE SOME PARAMS!!!
            analysis_kwargs['n_shuffle']=5
            dosave=False
            # pass
        
        res = analyze_data(data,*analysis_args,**analysis_kwargs)
        # misc.save_res(save_fn,res,dosave=dosave)
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
