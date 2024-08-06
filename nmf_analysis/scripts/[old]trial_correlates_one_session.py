import sys,os,pickle,copy, itertools
sys.path.append('/mnt/home/szheng/projects/seq_detection2/code')
sys.path.append('/mnt/home/szheng/projects/util_code')
sys.path.append('/mnt/home/szheng/projects/nmf_analysis')
import numpy as np
import scipy
import scipy.stats

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
import nmf_one_session as nos

import pynapple as nap
from sklearn.cluster import KMeans

from matplotlib.backends.backend_pdf import PdfPages

import submitit
N_COMPO = 4
def trial_correlates_one_session(data_dir_full,speed_thresh = 1,speed_key='v',doplots=True,force_reload=False):
    # upgly; for submitit imports have to be done within the function when append is needed 
    sys.path.append('/mnt/home/szheng/projects/seq_detection2/code')
    sys.path.append('/mnt/home/szheng/projects/util_code')
    sys.path.append('/mnt/home/szheng/projects/nmf_analysis')
    import data_prep_pyn as dpp
    import nmf_analysis as na
    import nmf_plot as nmfp
    import nmf_test as nt
    import trial_correlates as tc
    import preprocess as prep
    import plot_helper as ph
    

    # check existence
    res_to_save_dir = os.path.join(data_dir_full,'py_data','trial_correlates')
    if not os.path.exists(res_to_save_dir):
        os.makedirs(res_to_save_dir)
        print(f'{res_to_save_dir} made!',flush=True)

    res_to_save_name = f'trial_correlates'
    res_to_save_fn  = os.path.join(res_to_save_dir,res_to_save_name+'.p')
    if os.path.exists(res_to_save_fn) and not force_reload:
        res_to_save_data = pickle.load(open(res_to_save_fn,'rb'))
        print(f"{res_to_save_fn} exists, loading--!",flush=True)
        # return res_to_save_data
    
    else:
        mat_to_return=prep.load_stuff(data_dir_full,sessionPulses = '*SessionPulses.Events.mat')
        # sessionPulses=mat_to_return['sessionPulses']
        behavior=mat_to_return['behavior']
        

        prep_res = dpp.load_spk_beh_df(data_dir_full,force_reload=False,extra_load={})
        spk_beh_df=prep_res['spk_beh_df']
        cell_cols_d = prep_res['cell_cols_d']
        beh_df = prep_res['beh_df']
        beh_df=pd.DataFrame(beh_df)

        beh_df_d,beh_df = dpp.group_into_trialtype(beh_df)
        
        # get the trial correlates
        speed_stats = tc.get_variable_statistics_per_trial(beh_df,key=speed_key)
        # if 'v' in beh_df.columns:
        #     v_stats=tc.get_variable_statistics_per_trial(beh_df,key='v')

        time_related=tc.get_time_related_per_trial(beh_df)
        firing_related=tc.get_firing_related_per_trial(spk_beh_df,cell_cols_d)
        reward_related=tc.get_reward_related_per_trial(beh_df)
        try:
            ripples=mat_to_return['ripples']
            ripples_related=tc.get_ripples_related_per_trial(ripples,beh_df)
            has_ripples=True
        except:
            has_ripples = False
            pass
        
        # concat trial correlates with H from nmf
        # res = pickle.load(open(os.path.join(data_dir_full,'py_data','nmf','nmf_4.p'),'rb')) # might change, depending on nmf_one_session.py
        nmf_res = nos.nmf_one_session(data_dir_full,kwargs_={'n_compo':N_COMPO,'dosave':False,'doplots':False,'force_reload':False})
        H_sorted_d = nmf_res['H_sorted_d']
        df_d = {}
        res_to_save_data = {}
        for key, H in H_sorted_d.items():
            df=H.T
            df=pd.concat([df,speed_stats.loc[key]],axis=1)
            # df=pd.concat([df,v_stats.loc[key]],axis=1)
            df=pd.concat([df,time_related.loc[key]],axis=1)
            df=pd.concat([df,firing_related.loc[key]],axis=1)
            try:
                df=pd.concat([df,reward_related.loc[key]],axis=1) # for linear maze reward_related won't exist
            except:
                pass
            if has_ripples:
                df=pd.concat([df,ripples_related.loc[key]],axis=1)
            df.index = df.index.astype(int)
            df_d[key] = df
        res_to_save_data['df_d'] = df_d
        pickle.dump(res_to_save_data,open(res_to_save_fn,'wb'))
        print(f'{res_to_save_fn} saved!', flush=True)
    
    # plots
    fig_to_save_dir = os.path.join(data_dir_full,'py_figures','trial_correlates')
    if not os.path.exists(fig_to_save_dir):
        os.makedirs(fig_to_save_dir)
        print(f'{fig_to_save_dir} made!',flush=True)
    fig_to_save_fn = os.path.join(fig_to_save_dir,f'{res_to_save_name}.pdf')

    if doplots:
        df_d = res_to_save_data['df_d']
        with PdfPages(fig_to_save_fn) as pdf:
            for key, df in df_d.items():
                # plot H
                fig,ax=plt.subplots()
                ax.imshow(df.loc[:,0:N_COMPO-1].T,aspect='auto')
                ax.set_title(f'{key} H')
                ax.set_xticks(np.arange(df.shape[0]))
                ax.set_xticklabels(df.index)
                pdf.savefig(figure=fig,bbox_inches='tight')
                plt.close(fig=fig)

                # plot correlation
                corr = df.corr()
                mask = np.triu(np.ones_like(corr, dtype=bool))
                # corr
                fig,axs=plt.subplots(1,2,figsize=(20,10))
                sns.heatmap(corr,mask=mask,annot=corr.round(1),cmap='vlag',ax=axs[0])
                axs[0].set_title(f'{key} correlations')
                

                # sort and cluster variables and replot correlation 
                df_z=df.apply(lambda x:scipy.stats.zscore(x,nan_policy='omit'))
                df_z=df_z.dropna(axis=1,how='all').fillna(0) # drop columns that don't change; fill the rest na with 0
                corr_sorted,idx = tc.cluster_corr(df_z.corr())
                df_z_sorted=df_z.iloc[:,idx]
                mask = np.triu(np.ones_like(corr_sorted, dtype=bool))
                sns.heatmap(corr_sorted,mask=mask,cmap='vlag',ax=axs[1])
                axs[1].set_title(f'{key} sorted')
                pdf.savefig(figure=fig,bbox_inches='tight')
                plt.close(fig=fig)

                # plot the zscored values
                fig,axs=plt.subplots(1,2,figsize=(20,10))
                sns.heatmap(df_z,cmap='vlag',ax=axs[0])
                axs[0].set_title(f'{key} H and trial correlates')
                sns.heatmap(df_z_sorted,cmap='vlag',ax=axs[1])
                axs[1].set_title(f'{key} sorted')
                pdf.savefig(figure=fig,bbox_inches='tight')
                plt.close(fig=fig)

                # plot line plots
                nplots = len(df.columns)
                fig,axs = ph.subplots_wrapper(nplots,return_axs=True)    
                for ii,col in enumerate(df.columns):
                    df[[col]].plot(xticks=df.index,ax=axs.ravel()[ii])
                pdf.savefig(figure=fig,bbox_inches='tight')
                plt.close(fig=fig)
    return res_to_save_data

def main():
    log_folder = os.path.join('.', 'slurm_jobs', 'trial_correlates/%j')
    ex = submitit.AutoExecutor(folder=log_folder)
    if ex.cluster == 'slurm':
        print('submitit executor will schedule jobs on slurm!')
    else:
        print(f"!!! Slurm executable `srun` not found. Will execute jobs on '{ex.cluster}'")
    ex.update_parameters(
        slurm_job_name='trial_correlates',
        nodes=1,
        slurm_partition="genx", 
        cpus_per_task=1,
        mem_gb=8,  # 32gb for train mode, 8gb for eval mode
        timeout_min=1440
    )
    db = database.db
    subdb = db.query('owner=="roman"|owner=="ipshita"')
    # subdb = db.query('owner=="ipshita"')
    jobs = []
    with ex.batch():
        for data_dir_full in subdb['data_dir_full']:
            job = ex.submit(trial_correlates_one_session,data_dir_full)
            jobs.append(job)
    idx=0
    for data_dir_full in subdb['data_dir_full']:
        print(f'{jobs[idx].job_id} === {data_dir_full}')
        idx += 1
    
if __name__=='__main__':
    main()
