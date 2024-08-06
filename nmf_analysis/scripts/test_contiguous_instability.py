import sys,os,pickle,copy, itertools
sys.path.append('/mnt/home/szheng/projects/seq_detection2/code')
sys.path.append('/mnt/home/szheng/projects/util_code')
sys.path.append('/mnt/home/szheng/projects/nmf_analysis')
import numpy as np

import pandas as pd
from importlib import reload
import seaborn as sns

import database



import matplotlib as mpl
mpl.rcParams['image.cmap'] = 'Greys'
import matplotlib.pyplot as plt

# import nmf_analysis as na
# import nmf_plot as nmfp
# reload(na)
print(sys)
# import data_prep_pyn as dpp

import pynapple as nap
from sklearn.cluster import KMeans

from matplotlib.backends.backend_pdf import PdfPages

import submitit

# note force_reload default False; so if want to update remember to change it back
FORCE_RELOAD = True

DETECTION = 'avg'
SAVE_FN = f'contiguous_instability_{DETECTION}.p'

n_change_pts_max_MAX = 6 # adding a cap to the max number of change points to compare
def test_contiguous_instability(data_dir_full, save_dir_name='instability',
        detection=DETECTION,n_shuffle=500,force_reload=FORCE_RELOAD,dosave=True,load_only=False,save_fn=SAVE_FN):
    '''
    test whether a field time series across trials has change points, instead of just 
    fluctuates up and down
    right now depending on the fact that nmf has already been run, using nmf_one_session.py
    '''

    # upgly; for submitit imports have to be done within the function when append is needed 
    sys.path.append('/mnt/home/szheng/projects/seq_detection2/code')
    sys.path.append('/mnt/home/szheng/projects/util_code')
    sys.path.append('/mnt/home/szheng/projects/nmf_analysis')
    import data_prep_pyn as dpp
    import nmf_analysis as na
    import nmf_plot as nmfp
    import nmf_test as nt
    import place_field_analysis as pf
    import misc

    # deal with force reload
    save_dir =misc.get_or_create_subdir(data_dir_full,'py_data',save_dir_name)
    save_fn, res = misc.get_res(save_dir,save_fn,force_reload)
    # plt.close('all')
    if (res is not None) or load_only: # load only would skip the computation that follows
        return res

    # load data
    pf_res = pf.field_detection_both_avg_trial_wrapper(data_dir_full, dosave=True,force_reload=False,nbins = 100, 
                                        save_fn = 'place_field_avg_and_trial_vthresh.p', 
                                        shuffle_fn='fr_map_null_trialtype_vthresh.p',
                                        smth_in_bin=2.5, speed_thresh=1.,speed_key='v',load_only=True,
                                        )


    pf_alltrialtype = pf_res[detection]['params']

    instability_df_d= {}
    ratio_d = {}
    fr_key = 'fr_mean'
    for tt, pfr in pf_alltrialtype.items():
        fr = pfr.loc[fr_key]
        ntrials = fr.shape[1]
        n_change_pts_max = np.minimum(int(ntrials // 4),n_change_pts_max_MAX) # 4 here is kinda arbitrary
        n_change_pts_l = np.arange(1,n_change_pts_max+1)

        signal = fr.values.T
        instability_df = nt.test_contiguity_independent_multidim(signal,n_shuffle=n_shuffle,sig_thresh=0.05,n_change_pts_l=n_change_pts_l)
        instability_df.index = fr.index
        instability_df_d[tt] = instability_df
        # get ratio of neurons with at least one peak issig=True
        any_field_sig = instability_df.groupby(level=0,axis=0)['opt_issig'].any()
        ratio_of_pyr_has_unstable_field = any_field_sig.sum()/len(any_field_sig)
        ratio_d[tt] = ratio_of_pyr_has_unstable_field
    
    # save
    
    res_to_save = {'instability_df_d':instability_df_d,'ratio_d':ratio_d}
    misc.save_res(save_fn,res_to_save,dosave=dosave)
    
    return res_to_save


def main():
    log_folder = os.path.join('.', 'slurm_jobs', 'instability/%j')
    ex = submitit.AutoExecutor(folder=log_folder)
    if ex.cluster == 'slurm':
        print('submitit executor will schedule jobs on slurm!')
    else:
        print(f"!!! Slurm executable `srun` not found. Will execute jobs on '{ex.cluster}'")
    ex.update_parameters(
        slurm_job_name='nmf',
        nodes=1,
        slurm_partition="genx", 
        cpus_per_task=2,
        mem_gb=16,  # 32gb for train mode, 8gb for eval mode
        timeout_min=1440,
        slurm_array_parallelism=25,
    )
    db = database.db
    # subdb = db.query('owner=="roman"|owner=="ipshita"')
    # subdb = db.query('animal_name=="e13_26m1"')
    subdb = db.query('owner=="roman"').sort_values('n_pyr_putative',ascending=False)
    jobs = []
    with ex.batch():
        for data_dir_full in subdb['data_dir_full']:
            job = ex.submit(test_contiguous_instability,data_dir_full)
            jobs.append(job)
    idx=0
    for data_dir_full in subdb['data_dir_full']:
        print(f'{jobs[idx].job_id} === {data_dir_full}')
        idx += 1
    
if __name__=='__main__':
    main()
