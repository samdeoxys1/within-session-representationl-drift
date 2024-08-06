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

'''
For getting spk_beh_df, doing nmf, save all the results, 
figures of nmf as well as example factors
'''


KWARGS = {
    'nbins': 10, # number of position bins for dividing spikes within one trial, for nmf
    'speed_thresh': 1., # min speed to be considered to be moving
    'max_fr_thresh': 1., # min fr to be considered a peak in the rate map
    'n_compo': 4,
    'n_clust': 4,
    'doplots': True,
    'dosave': True,
    'force_reload':True,
    'do_normalize':True,
    'israte':True,
    'res_to_save_name':None,
}

def nmf_one_session(data_dir_full,kwargs_={}):
    # upgly; for submitit imports have to be done within the function when append is needed 
    sys.path.append('/mnt/home/szheng/projects/seq_detection2/code')
    sys.path.append('/mnt/home/szheng/projects/util_code')
    sys.path.append('/mnt/home/szheng/projects/nmf_analysis')
    import data_prep_pyn as dpp
    import nmf_analysis as na
    import nmf_plot as nmfp
    TRIALTYPE_KEY_DICT = dpp.TRIALTYPE_KEY_DICT

    kwargs = KWARGS
    kwargs.update(kwargs_)

    # unwrap args
    nbins = kwargs['nbins']
    speed_thresh = kwargs['speed_thresh']
    max_fr_thresh = kwargs['max_fr_thresh']
    n_compo = kwargs['n_compo']
    n_clust = kwargs['n_clust']
    doplots = kwargs['doplots']
    dosave = kwargs['dosave']
    force_reload = kwargs['force_reload']
    do_normalize = kwargs['do_normalize']
    israte = kwargs['israte'] # on firing rate or spike count
    res_to_save_name = kwargs['res_to_save_name']

    # save locations
    res_to_save_dir = os.path.join(data_dir_full,'py_data','nmf')
    if not os.path.exists(res_to_save_dir):
        os.makedirs(res_to_save_dir)
        print(f'{res_to_save_dir} made!',flush=True)
    if res_to_save_name is None:
        res_to_save_name = f'nmf_{n_compo}'
    res_to_save_fn  = os.path.join(res_to_save_dir,res_to_save_name+'.p')

    fig_to_save_dir = os.path.join(data_dir_full,'py_figures','nmf')
    if not os.path.exists(fig_to_save_dir):
        os.makedirs(fig_to_save_dir)
        print(f'{fig_to_save_dir} made!',flush=True)
    fig_to_save_fn = os.path.join(fig_to_save_dir,f'{res_to_save_name}.pdf')

    if os.path.exists(res_to_save_fn) and not force_reload:
        if (doplots and os.path.exists(fig_to_save_fn)) or not doplots: # if doplots then whether plots exist need also be checked
            res = pickle.load(open(res_to_save_fn,'rb'))
            print(f'{res_to_save_fn} already exists! Loading--')
            return res
    
    # load data
    prep_res = dpp.load_spk_beh_df(data_dir_full,force_reload=False,extra_load={})
    spk_beh_df=prep_res['spk_beh_df']
    cell_cols_d = prep_res['cell_cols_d']
    beh_df = prep_res['beh_df']
    fr_pyr = spk_beh_df.loc[:,list(cell_cols_d['pyr'])+list(beh_df.columns)]
    # convert to rate (Hz):
    dt = np.median(np.diff(beh_df.index))
    if israte:
        fr_pyr.loc[:,cell_cols_d['pyr']] = fr_pyr.loc[:,cell_cols_d['pyr']]/dt

    # bin coarsely by lin
    fr_filtered = copy.copy(fr_pyr.query("(speed>=@speed_thresh)"))
    fr_filtered['lin_binned'] = pd.cut(fr_filtered['lin'],nbins,retbins=False,labels=False) # should i bin first then do threshold???
    # fr_mean_trial_type = fr_filtered.groupby(['task_index','visitedArm','trial','lin_binned']).mean().loc[:,cell_cols_d['pyr']]

    # nmf
    W_df_d,W_sorted_d,H_sorted_d,W_df_peaks_only_d,W_df_original_d,X_normed_restacked_df_d = {},{},{},{},{},{}

    trialtype_key_dict = TRIALTYPE_KEY_DICT
    task_index_to_task_name = dpp.get_task_index_to_task_name(spk_beh_df)

    for task_type, fr_filtered_task in fr_filtered.groupby('task_index'):
        trialtype_key=trialtype_key_dict[task_index_to_task_name[task_type]] # visitedArm or direction
        for trial_type, fr_filtered_task_trialtype in fr_filtered_task.groupby(trialtype_key):
            if israte:
                fr_to_be_nmfed_one_trialtype = fr_filtered_task_trialtype.groupby(['trial','lin_binned'])[cell_cols_d['pyr']].mean()
                fr_peak_only = na.get_peaks_in_fr(fr_to_be_nmfed_one_trialtype,max_fr_thresh=max_fr_thresh)
            else:
                fr_to_be_nmfed_one_trialtype = fr_filtered_task_trialtype.groupby(['trial','lin_binned'])[cell_cols_d['pyr']].sum()
                fr_to_be_nmfed_one_trialtype_rate = fr_filtered_task_trialtype.groupby(['trial','lin_binned'])[cell_cols_d['pyr']].mean()
                fr_peak_only = na.get_peaks_in_fr(fr_to_be_nmfed_one_trialtype_rate,max_fr_thresh=max_fr_thresh)

            W_df, W_sorted,W_inds, factor_assignment, H_sorted, X_sorted, X_recon_sorted,X_normed_restacked_df = na.nmf_sort_with_position(fr_to_be_nmfed_one_trialtype,n_compo, model=None,do_normalize=do_normalize)
            
            W_df_original = copy.copy(W_df)
            W_df_peaks_only = W_df.loc[fr_peak_only.index]
            key = (task_type,trial_type)
            W_df_original_d[key] = W_df_original
            W_sorted_d[key] = W_sorted
            W_df_d[key] = W_df
            H_sorted_d[key] = H_sorted
            W_df_peaks_only_d[key] = W_df_peaks_only
            X_normed_restacked_df_d[key] =X_normed_restacked_df

    # post nmf, cluster, add metrics, sort
    sort_func = lambda x:na.hierarchical_sort(x,to_cut_keys=[f'skew_{x.name}','skew'],to_cut_nbins=[4,4],final_sort_key=x.name,ascending=False)
    W_df_peaks_only_post_sorted_d = {}
    for key,W in W_df_peaks_only_d.items():
        W_df_original = W_df_original_d[key]
        X_normed_restacked_df=X_normed_restacked_df_d[key] 

        # kmeans clustering
        res=KMeans(n_clust).fit(W)
        clust = res.labels_
        centroid = res.cluster_centers_
        dist = res.transform(W)
        dist_to_own_centroid = np.diag(dist[:,clust])

        # add metrics
        W_df = na.add_metrics_to_W(W, n_compo, pd_kwargs={})
        W_df['clust'] = clust
        W_df['factor'] = W_df.loc[:,0:n_compo-1].idxmax(axis=1)
        W_df['neg_dist_to_centroid'] = -dist_to_own_centroid # for easiness of sorting; all features the higher the better

        # sort
        W_df_sorted = W_df.groupby('factor').apply(sort_func)
        W_df_peaks_only_post_sorted_d[key] = W_df_sorted
    
    res_to_save = dict(
        W_df_peaks_only_post_sorted_d = W_df_peaks_only_post_sorted_d,
        X_normed_restacked_df_d = X_normed_restacked_df_d,
        kwargs = kwargs,
        W_sorted_d = W_sorted_d,
        H_sorted_d = H_sorted_d,
        W_df_peaks_only_d=W_df_peaks_only_d,
        W_df_original_d = W_df_original_d,
    )
    
    if dosave:
        pickle.dump(res_to_save,open(res_to_save_fn,'wb'))
        print(f"{res_to_save_fn} saved!",flush=True)

    # PLOTTING
    if doplots:
        
        with PdfPages(fig_to_save_fn) as pdf:
            for key, W_df_peaks_only_post_sorted in W_df_peaks_only_post_sorted_d.items():
                
                # plot wh
                W_sorted = W_sorted_d[key]
                H_sorted = H_sorted_d[key]
                fig,ax=nmfp.plot_wh(W_sorted, H_sorted,factor_neuron_ratio=300,spacing=2,trial_ticklabels=H_sorted.columns)
                fig.suptitle(key)
                pdf.savefig(figure=fig,bbox_inches='tight')
                plt.close(fig=fig)

                # plot cluster and W post sorting
                fig,axs = plt.subplots(2,1,figsize=(1*6,2*4))
                W_df_peaks_only_post_sorted.groupby('clust').mean().loc[:,0:n_compo-1].T.plot(figure=fig,ax=axs[0])
                axs[0].set_xticks(range(n_compo))
                sns.heatmap(W_df_peaks_only_post_sorted.loc[:,0:n_compo-1],ax=axs[1],figure=fig)
                pdf.savefig(figure=fig,bbox_inches='tight')
                plt.close(fig=fig)

                # plot example ratemaps
                clust_inds = W_df_peaks_only_post_sorted.index.get_level_values(level=0).unique()
                clust_name = clust_inds.name
                for clust in clust_inds:
                    nunits_per_page = 12
                    npages = 2 #W_df_peaks_only_post_sorted.loc[clust].shape[0] // nunits_per_page
                    for ii in range(npages):
                        rank_start,rank_end = (nunits_per_page*ii, nunits_per_page * (ii+1) )
                        sample_inds = W_df_peaks_only_post_sorted.loc[clust].index[rank_start:rank_end]
                        if len(sample_inds)>0:
                            fig,axs=nmfp.plot_example_W_and_ratemaps(W_df_original_d[key], sample_inds,X_normed_restacked_df_d[key],n_compo=n_compo)
                            fig.suptitle(f'{key},{clust_name}={clust}\nrank={rank_start}-{rank_end-1}',fontsize=20)
                            plt.tight_layout()
                            pdf.savefig(figure=fig,bbox_inches='tight')
                            plt.close(fig=fig)

        print(f'{fig_to_save_fn} saved!',flush=True)

    return res_to_save

def main():
    log_folder = os.path.join('.', 'slurm_jobs', 'nmf/%j')
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
    # subdb = db.query('owner=="roman"')
    subdb = db.query('owner=="roman"|owner=="ipshita"')
    jobs = []
    with ex.batch():
        for data_dir_full in subdb['data_dir_full']:
            job = ex.submit(nmf_one_session,data_dir_full)
            jobs.append(job)
    idx=0
    for data_dir_full in subdb['data_dir_full']:
        print(f'{jobs[idx].job_id} === {data_dir_full}')
        idx += 1
    
if __name__=='__main__':
    main()



