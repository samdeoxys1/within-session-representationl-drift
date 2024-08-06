import numpy as np
import scipy
from scipy.ndimage import gaussian_filter1d
import sklearn
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture,BayesianGaussianMixture

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['image.cmap'] = 'Greys'

import seaborn as sns

import sys,os,pdb,copy,pickle
from importlib import reload
import pynapple as nap

sys.path.append('/mnt/home/szheng/projects/seq_detection2/code')
sys.path.append('/mnt/home/szheng/projects/util_code')
sys.path.append('/mnt/home/szheng/projects/nmf_analysis')
sys.path.append('/mnt/home/szheng/projects/nmf_analysis/scripts')
sys.path.append('/mnt/home/szheng/projects/cluster_spikes')
import data_prep_new as dpn
import place_cell_analysis as pa
import plot_helper as ph
import preprocess as prep
import nmf_analysis as na
import nmf_plot as nmfp
reload(na)

import raster_new as rn
import data_prep_pyn as dpp
import database


import change_point_analysis as cpa
import change_point_plot as cpp
import change_point_post_analysis as cppa

import place_field_analysis as pf
import pingouin as pg
import misc
from collections import OrderedDict

db = database.db
sess_to_plot = db.query("owner=='roman'").sort_values('n_pyr_putative',ascending=False)
db_sorted= sess_to_plot
print(f'{sess_to_plot.shape[0]} sessions!')


PLACE_FIELD_FN = 'place_field_res_speed_residual.p' #'place_field_avg_and_trial_vthresh.p'
SHUFFLE_FN = 'fr_map_null_trialtype_vthresh.p'
TRIAL_TYPE_L = [(0,0),(0,1)]
FR_KEY = 'fr_mean' # peak or mean to use for detection
# SAVE_FN = lambda fr_key:f'shuffle_detection_vthresh_switch_res_switch_magnitude_only_{fr_key}.p'
SAVE_FN = lambda fr_key:f'shuffle_detection_vthresh_switch_res_switch_magnitude_only_{fr_key}_highrepeats.p'
# SAVE_FN = lambda fr_key:f'resid_switch_res_switch_magnitude_only_{fr_key}_highrepeats.p'
force_reload = True

def main(i,testmode=False):
    data_dir_full = sess_to_plot.iloc[i]['data_dir_full']
    if testmode:
        nrepeats = 2
    else:
        nrepeats = 10000
    switch_analysis_one_session(data_dir_full,force_reload=force_reload,nrepeats=nrepeats)

def switch_analysis_one_session(data_dir_full,place_field_res=None,force_reload=force_reload,nrepeats=1000,place_field_fn = PLACE_FIELD_FN, 
                                        shuffle_fn=SHUFFLE_FN,dosave=True,save_fn=SAVE_FN(FR_KEY),load_only=False,speed_key='speed_kalman'):
    res_to_save_dir = os.path.join(data_dir_full,'py_data','switch_analysis')
    if not os.path.exists(res_to_save_dir):
        os.makedirs(res_to_save_dir)
        print(f'{res_to_save_dir} made!',flush=True)
    # deal with force reload
    save_dir =misc.get_or_create_subdir(data_dir_full,'py_data','switch_analysis')
    # save_fn =SAVE_FN(FR_KEY)
    save_fn, res = misc.get_res(save_dir,save_fn,force_reload)
    # plt.close('all')
    if (res is not None) or load_only: # load only would skip the computation that follows
        return res
    if place_field_res is None:
        place_field_res=pf.field_detection_both_avg_trial_wrapper(data_dir_full, dosave=True,force_reload=False,nbins = 100, 
                                        save_fn = place_field_fn, 
                                        shuffle_fn=shuffle_fn,
                                        smth_in_bin=2.5, speed_thresh=1.,speed_key=speed_key,
                                        )
    
    X_all = {}
    X_all_norm = {}
    com_agg_all = {}
    fr_key = FR_KEY
    trial_type_l = TRIAL_TYPE_L

    detection_type_l =[d for d in place_field_res.keys() if 'kwargs' not in d]

    X_all_detection = {k:{} for k in detection_type_l}
    X_all_norm_detection = {k:{} for k in detection_type_l}
    # com_agg_all_detection = {k:{} for k in detection_type_l}

    for detection in detection_type_l:
        X_all = {}
        X_all_norm = {}
        com_agg_all = {}
        for tt in trial_type_l:
            X = place_field_res[detection]['params'][tt].loc[fr_key]
            # com_agg = place_field_res[detection]['all_fields'][tt].loc[:,'com']
            X_all_detection[detection][tt] = X
            X_norm = X/X.max(axis=1).values[:,None]
            X_all_norm_detection[detection][tt] = X_norm
            # com_agg_all_detection[detection][tt] = com_agg
    
    
    X_to_be_analyzed_detection = {}
    for k,val in X_all_norm_detection.items():
        # X_to_be_analyzed=cpa.turn_X_into_pwc_sweep(val,pen_l=[0.3,0.5])
        X_to_be_analyzed=cpa.turn_X_into_pwc_sweep(val,pen_l=[0.3])
        X_to_be_analyzed_detection[k] = X_to_be_analyzed
    
    
    # plt.ioff()
    # tosweep_key_l = ['switch_magnitude','high_thresh']
    # tosweep_val_l = [np.arange(0,0.7,0.1).round(1),np.arange(0.3,0.8,0.1).round(1)]
    # kwargs_l = [dict(low_thresh=1,high_thresh=0),
    #     dict(low_thresh=0.2,switch_magnitude = 0),
    # ]
    tosweep_key_l = ['switch_magnitude']
    # tosweep_val_l = [np.arange(0,0.7,0.2).round(1)]
    tosweep_val_l = [[0.4]]
    kwargs_l = [dict(low_thresh=1,high_thresh=0)
    ]
    min_size = 1

    res_to_save_data_detection = {}
    for detection,X_to_be_analyzed in X_to_be_analyzed_detection.items():
        cdf_alltrialtype,sig_alltrialtype,sr_alltrialtype,changes_df_alltrialtype,fig_alltrialtype = cpa.sweep_test_switch_ratio_multisweep_alltrialtype_multipreprocess(X_to_be_analyzed,min_size,tosweep_key_l,tosweep_val_l,kwargs_l,detect_func=cpa.detect_switch_pwc,alpha=0.05,do_bonf=True,doplots=True,nrepeats=nrepeats)    

        res_to_save_data = dict(
            X = X_to_be_analyzed,
            cdf=cdf_alltrialtype,
            sig=sig_alltrialtype,
            sr=sr_alltrialtype,
            changes_df=changes_df_alltrialtype,
            # fig=fig_alltrialtype
        )

        res_to_save_data_detection[detection] = res_to_save_data
    
    
    if dosave:
        pickle.dump(res_to_save_data_detection,open(save_fn,'wb'))
        print(f'{save_fn} saved!', flush=True)
    # plt.close('all')
    return res_to_save_data_detection
    

def load_switch_analysis_res_allsess(save_fn_one = SAVE_FN,force_reload=False,dosave=True,load_only=False,n_pyr_thresh=50,save_dir='/mnt/home/szheng/ceph/place_variability/data'):
    
    save_fn_all = 'all_'+save_fn_one
    save_dir =misc.get_or_create_subdir(save_dir)
    save_fn_all, res = misc.get_res(save_dir,save_fn_all,force_reload)
    # plt.close('all')
    if (res is not None) or load_only: # load only would skip the computation that follows
        return res
    data_dir_and_sess=list(db_sorted[['data_dir_full','animal_name.1','sess_name']].itertuples(index=False,name=None))
    res_to_save_data_detection_l = OrderedDict()
    
    for ddf,ani, sess in data_dir_and_sess:
        res_to_save_data_detection_l[(ani,sess)] = switch_analysis_one_session(ddf,place_field_res=None,force_reload=False,nrepeats=1000,dosave=False,save_fn=save_fn_one,load_only=True)
    
    plt.close('all')

    sess_selected = db_sorted.loc[db_sorted['n_pyr_putative'] >= n_pyr_thresh,'sess_name']
    switch_detection_res_allsess = cppa.reshape_switch_detection_result_all_sess(res_to_save_data_detection_l,sess_selected)

    misc.save_res(save_fn_all,switch_detection_res_allsess,dosave=dosave)
    return switch_detection_res_allsess

if __name__ == '__main__':
    args = sys.argv[1:]
    print(args)
    main(int(args[0]))
