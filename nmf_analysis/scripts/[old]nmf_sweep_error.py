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


def nmf_sweep_error(data_dir_full):
    # upgly; for submitit imports have to be done within the function when append is needed 
    sys.path.append('/mnt/home/szheng/projects/seq_detection2/code')
    sys.path.append('/mnt/home/szheng/projects/util_code')
    sys.path.append('/mnt/home/szheng/projects/nmf_analysis')
    import data_prep_pyn as dpp
    import nmf_analysis as na
    import nmf_plot as nmfp
    TRIALTYPE_KEY_DICT = dpp.TRIALTYPE_KEY_DICT

    # load data
    reload(na)
    prep_res = dpp.load_spk_beh_df(data_dir_full,force_reload=False,extra_load={})
    spk_beh_df=prep_res['spk_beh_df']
    cell_cols_d = prep_res['cell_cols_d']
    beh_df = prep_res['beh_df']
    fr_pyr = spk_beh_df.loc[:,list(cell_cols_d['pyr'])+list(beh_df.columns)]
    # convert to rate (Hz):
    dt = np.median(np.diff(beh_df.index))
    fr_pyr.loc[:,cell_cols_d['pyr']] = fr_pyr.loc[:,cell_cols_d['pyr']]/dt

    speed_thresh = 1.
    nbins = 10
    max_fr_thresh = 1.

    # bin coarsely by lin
    fr_filtered = copy.copy(fr_pyr.query("(speed>=@speed_thresh)"))
    fr_filtered['lin_binned'] = pd.cut(fr_filtered['lin'],nbins,retbins=False,labels=False)
    # fr_mean_trial_type = fr_filtered.groupby(['task_index','visitedArm','trial','lin_binned']).mean().loc[:,cell_cols_d['pyr']]

    trialtype_key_dict = dpp.TRIALTYPE_KEY_DICT
    task_index_to_task_name = dpp.get_task_index_to_task_name(spk_beh_df)

    error_ratio_l_d = {}
    do_normalize = False

    for task_type, fr_filtered_task in fr_filtered.groupby('task_index'):
        trialtype_key=trialtype_key_dict[task_index_to_task_name[task_type]] # visitedArm or direction
        for trial_type, fr_filtered_task_trialtype in fr_filtered_task.groupby(trialtype_key):
            fr_to_be_nmfed_one_trialtype = fr_filtered_task_trialtype.groupby(['trial','lin_binned'])[cell_cols_d['pyr']].mean()
            fr_peak_only = na.get_peaks_in_fr(fr_to_be_nmfed_one_trialtype,max_fr_thresh=max_fr_thresh)
            
            sweep_l = None
            error_ratio_l_d[(task_type,trial_type)]={}
            for model in ['nmf','pca']:
                error_ratio_l = na.sweep_nmf_get_error(fr_to_be_nmfed_one_trialtype,sweep_l=sweep_l,model=model,do_normalize=do_normalize)
                error_ratio_l_d[(task_type,trial_type)][model] = error_ratio_l
                
    res_to_save_dir = os.path.join(data_dir_full,'py_data','nmf')
    if not os.path.exists(res_to_save_dir):
        os.makedirs(res_to_save_dir)
        print(f'{res_to_save_dir} made!',flush=True)
    
    res_to_save_name = f'nmf_error'
    res_to_save_fn  = os.path.join(res_to_save_dir,res_to_save_name+'.p')
    pickle.dump(error_ratio_l_d,open(res_to_save_fn,'wb'))
    print(f"{res_to_save_fn} saved!",flush=True)

    # plot
    fig,axs = plt.subplots(1,len(error_ratio_l_d))
    for ii,(k,val) in enumerate(error_ratio_l_d.items()):
        for kk,valval in val.items():
            axs[ii].plot(valval,label=kk)
        axs[ii].set_title(k)
        axs[ii].legend()
    fig_to_save_dir = os.path.join(data_dir_full,'py_figures','nmf')
    if not os.path.exists(fig_to_save_dir):
        os.makedirs(fig_to_save_dir)
        print(f'{fig_to_save_dir} made!',flush=True)
    fig_to_save_fn = os.path.join(fig_to_save_dir,f'{res_to_save_name}.pdf')
    fig.savefig(fig_to_save_fn,bbox_inches='tight')
    return error_ratio_l_d

def main():
    log_folder = os.path.join('.', 'slurm_jobs', 'logs/%j')
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
        timeout_min=240
    )
    db = database.db
    subdb = db.query('owner=="roman"|owner=="ipshita"')
    # subdb = db.query('owner=="ipshita"')
    jobs = []
    with ex.batch():
        for data_dir_full in subdb['data_dir_full']:
            job = ex.submit(nmf_sweep_error,data_dir_full)
            jobs.append(job)
    idx=0
    for data_dir_full in subdb['data_dir_full']:
        print(f'{jobs[idx].job_id} === {data_dir_full}')
        idx += 1
    
if __name__=='__main__':
    main()



