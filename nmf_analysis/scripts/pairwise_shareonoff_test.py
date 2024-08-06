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
import switch_analysis_one_session as saos

db = database.db
sess_to_plot = db.query("owner=='roman'").sort_values('n_pyr_putative',ascending=False)
db_sorted= sess_to_plot
print(f'{sess_to_plot.shape[0]} sessions!')

SW_RES_FN = 'shuffle_detection_vthresh_switch_res_switch_magnitude_only_fr_mean_highrepeats.p'
FORCE_RELOAD = True
SAVE_FN = 'pairwise_test_shuffle_detection_vthresh_switch_res_switch_magnitude_only_fr_mean_highrepeats.p'


def main(i,testmode=False):


    data_dir_full = db_sorted.iloc[i]['data_dir_full']
    if testmode:
        nrepeats = 2
    else:
        nrepeats = 10000
    
    test_res_d=pairwise_shareonoff_test_one_session(data_dir_full,SW_RES_FN, save_fn=SAVE_FN,dosave=True,force_reload=FORCE_RELOAD,
                                        nrepeats=nrepeats,load_only=False
                                        )
    return test_res_d
    

def pairwise_shareonoff_test_one_session(data_dir_full,sw_res_fn, save_fn=SAVE_FN,dosave=True,force_reload=FORCE_RELOAD,
                                        nrepeats=100,load_only=False
                                        ):
    
    res_to_save_dir = os.path.join(data_dir_full,'py_data','switch_analysis')
    if not os.path.exists(res_to_save_dir):
        os.makedirs(res_to_save_dir)
        print(f'{res_to_save_dir} made!',flush=True)
    # deal with force reload
    save_dir =misc.get_or_create_subdir(data_dir_full,'py_data','switch_analysis')
    save_fn, res = misc.get_res(save_dir,save_fn,force_reload)
    # plt.close('all')
    if (res is not None) or load_only: # load only would skip the computation that follows
        return res
    try:
        sw_res_one=saos.switch_analysis_one_session(data_dir_full,place_field_res=None,force_reload=False,nrepeats=1000,dosave=False,save_fn=sw_res_fn,load_only=True)
        plt.close('all')
        test_res_d = {}
        for detection in sw_res_one.keys():
            gpb = sw_res_one[detection]['changes_df'].groupby(level=(0,1,2,3,4))
            test_res = {}
            for key, val in gpb:
                test_res[key],_=cpa.shuffle_test_pair_share_onoff(val.dropna(axis=1),nrepeats=nrepeats,alpha=0.025)
            test_res = pd.concat(test_res,axis=0)
            test_res_d[detection] = test_res.unstack(level=(-2,-1))
        misc.save_res(save_fn,test_res_d,dosave)
        return test_res_d
    except Exception as e:
        print(e)
        return 
    


if __name__ == '__main__':
    args = sys.argv[1:]
    print(args)
    main(int(args[0]))
