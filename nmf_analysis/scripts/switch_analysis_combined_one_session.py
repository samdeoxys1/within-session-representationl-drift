import numpy as np
import scipy
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['image.cmap'] = 'Greys'

import seaborn as sns

import sys,os,pdb,copy,pickle
from importlib import reload


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
import place_field_analysis as pf
import pingouin as pg
import misc
import database
from collections import OrderedDict

import change_point_analysis_central_arm_seperate as cpacas

db = database.db
sess_to_plot = db.query("owner=='roman'").sort_values('n_pyr_putative',ascending=False)
db_sorted= sess_to_plot
print(f'{sess_to_plot.shape[0]} sessions!')




def main(i,test_mode=0):
    data_dir_full = db_sorted.iloc[i]['data_dir_full']
    if test_mode:
        nrepeats_sw = 2
    else:
        nrepeats_sw = 1000
    cpacas.sweep_test_coswitch_wrapper(data_dir_full,
                                pf_res_save_fn='place_field_avg_and_trial_vthresh.p',
                                pf_shuffle_fn = 'fr_map_null_trialtype_vthresh.p',
                                speed_key='v',fr_key='fr_peak',
                                bin_size=2.2,
                                switch_res_query=(slice(None),0.3,'switch_magnitude',0.4),
                                nrepeats_sw = nrepeats_sw,
                                edges = None,
                                save_fn = 'switch_res_window.p',
                                load_only=False,
                                dosave=True,force_reload=True,
                                task_ind = 0,
                                prep_force_reload=False,
                                )

if __name__ == '__main__':
    args = sys.argv[1:]
    print(args)
    main(int(args[0]))