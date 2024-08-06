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
n_change_pts_max_MAX = 6 # adding a cap to the max number of change points to compare
def test_contiguous_instability(data_dir_full,n_shuffle=500,force_reload=FORCE_RELOAD):
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

    # check existence
    res_to_save_dir = os.path.join(data_dir_full,'py_data','instability')
    if not os.path.exists(res_to_save_dir):
        os.makedirs(res_to_save_dir)
        print(f'{res_to_save_dir} made!',flush=True)
    
    res_to_save_name = f'contiguous_instability'
    res_to_save_fn  = os.path.join(res_to_save_dir,res_to_save_name+'.p')
    if os.path.exists(res_to_save_fn) and not force_reload:
        res_to_save_data = pickle.load(open(res_to_save_fn,'rb'))
        print(f"{res_to_save_fn} exists, loading--!",flush=True)
        return res_to_save_data

    res = pickle.load(open(os.path.join(data_dir_full,'py_data','nmf','nmf_4.p'),'rb')) # might change, depending on nmf_one_session.py

    instability_df_d= {}
    ratio_d = {}
    for key in res['W_df_peaks_only_d'].keys(): 
        peak_inds=res['W_df_peaks_only_d'][key].index # note the peak here is from W, not from the original X
        X_normed_restacked_df = res['X_normed_restacked_df_d'][key]
        X_normed_restacked_df_peaks_only = X_normed_restacked_df.loc[peak_inds]
        ntrials = X_normed_restacked_df.shape[1]
        n_change_pts_max = np.minimum(int(ntrials // 4),n_change_pts_max_MAX) # 4 here is kinda arbitrary
        n_change_pts_l = np.arange(1,n_change_pts_max+1)

        signal = X_normed_restacked_df_peaks_only.values.T
        instability_df = nt.test_contiguity_independent_multidim(signal,n_shuffle=n_shuffle,sig_thresh=0.05,n_change_pts_l=n_change_pts_l)
        instability_df.index = X_normed_restacked_df_peaks_only.index
        instability_df_d[key] = instability_df
        # get ratio of neurons with at least one peak issig=True
        any_field_sig = instability_df.groupby(level=0,axis=0)['opt_issig'].any()
        ratio_of_pyr_has_unstable_field = any_field_sig.sum()/len(any_field_sig)
        ratio_d[key] = ratio_of_pyr_has_unstable_field
    
    # save
    res_to_save_data = {'instability_df_d':instability_df_d,'ratio_d':ratio_d}
    pickle.dump(res_to_save_data,open(res_to_save_fn,'wb'))
    print(f"{res_to_save_fn} saved!",flush=True)
    return res_to_save_data


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
        cpus_per_task=1,
        mem_gb=8,  # 32gb for train mode, 8gb for eval mode
        timeout_min=1440
    )
    db = database.db
    # subdb = db.query('owner=="roman"|owner=="ipshita"')
    subdb = db.query('animal_name=="e13_26m1"')
    # subdb = db.query('owner=="ipshita"')
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
