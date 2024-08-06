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

# TRIAL_TYPE_L = [(0,0),(0,1)]

N_SHUFFLE = 500

def nmf_sweep_error(data_dir_full,detection='avg',fr_key='fr_mean',n_shuffle=N_SHUFFLE,force_reload=False,save_dir_name='nmf_error',dosave=True,save_fn='nmf_sweep_error_shuffle.p',load_only=False,do_plot=True):
    # upgly; for submitit imports have to be done within the function when append is needed 
    sys.path.append('/mnt/home/szheng/projects/seq_detection2/code')
    sys.path.append('/mnt/home/szheng/projects/util_code')
    sys.path.append('/mnt/home/szheng/projects/nmf_analysis')
    import data_prep_pyn as dpp
    import nmf_analysis as na
    import nmf_plot as nmfp
    import place_field_analysis as pf
    import change_point_analysis as cpa
    import misc

    TRIALTYPE_KEY_DICT = dpp.TRIALTYPE_KEY_DICT

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
    error_ratio_l_d = {}
    error_ratio_l_d_shuffle = {}
    err_difference_d = {}
    fig_d = {}
    do_normalize = True


    # for tt in TRIAL_TYPE_L:
    #     pfr = TRIAL_TYPE_L[tt]
    for tt, pfr in pf_alltrialtype.items():
        fr = pfr.loc[fr_key]
        sweep_l = None
        model = 'nmf'
        
        error_ratio_l = na.sweep_nmf_get_error(fr,sweep_l=sweep_l,model=model,do_normalize=do_normalize,prep_for_concat_bin=False)
        error_ratio_l_d[tt] = error_ratio_l

        fr_shuffle_l = cpa.gen_circular_shuffle(fr,nrepeats=n_shuffle,new_start_inds=None)
        error_ratio_l_shuffle_all = []
        for fr_shuffle in fr_shuffle_l:
            error_ratio_l_shuffle_one = na.sweep_nmf_get_error(fr_shuffle,sweep_l=sweep_l,model=model,do_normalize=do_normalize)
            error_ratio_l_shuffle_all.append(error_ratio_l_shuffle_one)
        error_ratio_l_shuffle_all = pd.DataFrame(error_ratio_l_shuffle_all)
        error_ratio_l_d_shuffle[tt] = error_ratio_l_shuffle_all

        n_compo_range = (2,np.minimum(8,fr.shape[1]-2)) 
        err_difference = np.mean(np.mean(error_ratio_l_d_shuffle[tt].loc[:,n_compo_range[0]:n_compo_range[1]] - error_ratio_l.loc[n_compo_range[0]:n_compo_range[1]]))
        err_difference_d[tt] = err_difference

    res_to_save = {'error':error_ratio_l_d,'shuffle_error':error_ratio_l_d_shuffle,'err_difference':err_difference_d}

    if do_plot:
        ax_d={}
        for tt in pf_alltrialtype.keys():
            fig,ax=plt.subplots()
            ax.plot(error_ratio_l_d[tt])
            ax=sns.lineplot(data=error_ratio_l_d_shuffle[tt].melt(),x='variable',y='value',ax=ax)
            ax.set(ylabel='Reconstruction Error', xlabel='N Components')
            ax.legend(['Data', 'Shuffle'])
            ax_d[tt] = ax
        

    misc.save_res(save_fn,res_to_save,dosave=dosave)


    # # plot
    # fig,axs = plt.subplots(1,len(error_ratio_l_d))
    # for ii,(k,val) in enumerate(error_ratio_l_d.items()):
    #     for kk,valval in val.items():
    #         axs[ii].plot(valval,label=kk)
    #     axs[ii].set_title(k)
    #     axs[ii].legend()
    # fig_to_save_dir = os.path.join(data_dir_full,'py_figures','nmf')
    # if not os.path.exists(fig_to_save_dir):
    #     os.makedirs(fig_to_save_dir)
    #     print(f'{fig_to_save_dir} made!',flush=True)
    # fig_to_save_fn = os.path.join(fig_to_save_dir,f'{res_to_save_name}.pdf')
    # fig.savefig(fig_to_save_fn,bbox_inches='tight')
    # return error_ratio_l_d
    return res_to_save

def main():
    log_folder = os.path.join('.', 'slurm_jobs','nmf_error', 'logs/%j')
    ex = submitit.AutoExecutor(folder=log_folder)
    if ex.cluster == 'slurm':
        print('submitit executor will schedule jobs on slurm!')
    else:
        print(f"!!! Slurm executable `srun` not found. Will execute jobs on '{ex.cluster}'")
    ex.update_parameters(
        slurm_job_name='nmf_error',
        nodes=1,
        slurm_partition="genx", 
        cpus_per_task=2,
        mem_gb=16,  # 32gb for train mode, 8gb for eval mode
        timeout_min=240,
        slurm_array_parallelism=10,
    )
    db = database.db
    # subdb = db.query('owner=="roman"|owner=="ipshita"')
    subdb = db.query('owner=="roman"').sort_values('n_pyr_putative',ascending=False)
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



