import numpy as np

import sys,os,pdb,copy,pickle
from importlib import reload
# import pynapple as nap

sys.path.append('/mnt/home/szheng/projects/seq_detection2/code')
sys.path.append('/mnt/home/szheng/projects/util_code')
sys.path.append('/mnt/home/szheng/projects/nmf_analysis')
sys.path.append('/mnt/home/szheng/projects/nmf_analysis/scripts')
sys.path.append('/mnt/home/szheng/projects/cluster_spikes')
sys.path.append('/mnt/home/szheng/projects/place_variability/code')
import data_prep_new as dpn
import data_prep_pyn as dpp

import pf_recombine_central as pfrc
import fr_map_one_session as fmos
import switch_detection_one_session as sdos
import get_all_switch_add_metrics as gasam

import preprocess_one_session as prepos
import preprocess as prep


import data_prep_pyn

import neo
import neo.rawio
from quantities import mV
import pynapple as nap
import tqdm

import database
db = database.db
import misc

subdb = db.query('owner=="roman"').sort_values('n_pyr_putative',ascending=False)
subdb = subdb.query('ntrials>=20&n_neurons>=50')
to_exclude_sess = ['e16_3m2_211211']
subdb=subdb.loc[~subdb['sess_name'].isin(to_exclude_sess)]


def main():


    figdir = '/mnt/home/szheng/ceph/place_variability/fig/paper/suppfigure_waveform'
    misc.get_or_create_subdir(figdir)
    suppfigdir =misc.get_or_create_subdir(figdir,'supp')
    data_dir = '/mnt/home/szheng/ceph/place_variability/data/paper'
    misc.get_or_create_subdir(data_dir)


    ### load
    fn = 'prepped_data_agg.p'
    fn_full = os.path.join(data_dir,fn)
    prepped_data_agg = pickle.load(open(fn_full,'rb'))

    all_fields_recombined_all = prepped_data_agg['all_fields_recombined_all']
    pf_params_recombined_all=  prepped_data_agg['pf_params_recombined_all']
    fr_map_trial_df_all= prepped_data_agg['fr_map_trial_df_all']
    occu_map_all = prepped_data_agg['occu_map_all']
    fr_map_all = prepped_data_agg['fr_map_all']
    fr_map_trial_df_pyr_combined_all = prepped_data_agg['fr_map_trial_df_pyr_combined_all']

    pval_all = prepped_data_agg['pval_all']
    X_pwc_all = prepped_data_agg['X_pwc_all']
    X_raw_all = prepped_data_agg['X_raw_all']
    changes_df_all = prepped_data_agg['changes_df_all']
    var_res_all_test = prepped_data_agg['var_res_all']
    # corr_all = prepped_data_agg['corr_all']
    all_sw_d_all = prepped_data_agg['all_sw_d_all']
    all_sw_with_metrics_d_all = prepped_data_agg['all_sw_with_metrics_d_all']
    best_n_all_test = prepped_data_agg['best_n_all']
    spk_beh_df_all = prepped_data_agg['spk_beh_df_all']
    pf_params_all = prepped_data_agg['pf_params_all']

    ntrials_per_tt=fr_map_trial_df_all.groupby(level=(0,1,2,3)).apply(lambda x:x.dropna(axis=1,how='all').shape[1])
    ntrials_mask = ntrials_per_tt.groupby(level=(0,1,2)).apply(lambda x:(x>=7).all()) 

    trial_index_to_index_within_df_all=prepped_data_agg['trial_index_to_index_within_df_all']

    row = subdb.iloc[0]
    data_dir_full = row['data_dir_full']
    sess_name=row['sess_name']

    mat_to_return=prep.load_stuff(data_dir_full,sessionPulses='*SessionPulses.Events.mat')
    # sessionPulses=mat_to_return['sessionPulses']
    # filtered = mat_to_return['filtered']
    behavior=mat_to_return['behavior']
    # ripples = mat_to_return['ripples']
    cell_metrics = mat_to_return['cell_metrics']

    prep_res = dpp.load_spk_beh_df(data_dir_full,force_reload=False,extra_load={})
    # prep_res = dpp.load_spk_beh_df(data_dir_full,force_reload=True,extra_load=dict(sessionPulses='*SessionPulses.Events.mat',filtered='*thetaFiltered.lfp.mat'))
    spk_beh_df=prep_res['spk_beh_df']
    _,spk_beh_df = dpp.group_into_trialtype(spk_beh_df)
    # spk_beh_df,_=dpp.add_lin_binned(spk_beh_df,nbins=100)
    spk_beh_df,_=dpp.add_lin_binned(spk_beh_df,bin_size=2.2,nbins=None)
    cell_cols_d = prep_res['cell_cols_d']
    beh_df = prep_res['beh_df']
    beh_df_d,beh_df=dpp.group_into_trialtype(beh_df)
    spike_trains = prep_res['spike_trains']

    ani,sess,ti = row['animal_name.1'],row['sess_name'],0
    all_sw_onesess=all_sw_with_metrics_d_all.loc[ani,sess,ti]


    
    dat_fn = os.path.join(data_dir_full,f'{sess_name}.dat')
    reader = neo.io.RawBinarySignalIO(dat_fn,dtype='int16',
                                  nb_channel=128,
                                  sampling_rate=30000,
                                 )
    data_lazy=reader.read(lazy=True)
    anasigproxy=data_lazy[0].segments[0].analogsignals[0]


    gpb=all_sw_onesess.groupby('switch')
    waveform_allspk_alluid_ep_allsw={}
    waveform_avgspk_alluid_ep_allsw={}
    for onoff,all_sw_onesess_onoff in gpb:
        waveform_allspk_alluid_ep_allsw[onoff] = {}
        waveform_avgspk_alluid_ep_allsw[onoff]={}
        for sw_id,sw_row in tqdm.tqdm(all_sw_onesess_onoff.iterrows(),total=all_sw_onesess_onoff.shape[0]):
            uid,sw_time=sw_row['uid'],sw_row['time']
            waveform_allspk_alluid_ep, waveform_avgspk_alluid_ep = get_waveform_pre_post(anasigproxy,uid,sw_time, 
                            cell_metrics,
                            spike_trains,
                            win_sz_half=20,
                            spk_win_sz_half=0.001
                            )
            waveform_allspk_alluid_ep_allsw[onoff][sw_id] = waveform_allspk_alluid_ep
            waveform_avgspk_alluid_ep_allsw[onoff][sw_id] = waveform_avgspk_alluid_ep
    
    res = {'all':waveform_allspk_alluid_ep_allsw,'avg':waveform_avgspk_alluid_ep_allsw}
    save_fn_full = os.path.join(data_dir_full,'py_data','waveform_peri_sw.p')
    pickle.dump(res,open(save_fn_full,'wb'))
    return save_fn_full         


def get_waveform_pre_post(anasigproxy,uid,sw_time, 
                          cell_metrics,
                          spike_trains,
                          win_sz_half=20,
                          spk_win_sz_half=0.001,
                          to_microV_factor=0.195
                         ):
    '''
    win_sz_half: window peri switch
    spk_win_sz_half: window peri spike
    
    waveform_allspk_alluid_ep: {'pre':{uid:(n_spks x n_bin_in_spk_win)}, 'post':}
    waveform_avgspk_alluid_ep: {'pre':{uid: (n_bin_in_spk_win, )}, 'post':}
    '''



    # get cells with the same shank
    shank_id = cell_metrics['shankID'][uid-1]
    uid_with_same_shank = cell_metrics['UID'][cell_metrics['shankID']==shank_id]
    # get window of spike around the switch
    
    peri_switch_epoch = nap.IntervalSet(start=sw_time+0.01,end=sw_time+win_sz_half)
    pre_post_epoch = {
        'pre':nap.IntervalSet(start=sw_time-win_sz_half,end=sw_time-0.01),
        'post':nap.IntervalSet(start=sw_time+0.01,end=sw_time+win_sz_half)
    }
    
    # get waveform for spikes in the windows
    waveform_allspk_alluid_ep = {}
    waveform_avgspk_alluid_ep={}
    for prepost,ep in pre_post_epoch.items():
        peri_switch_epoch =pre_post_epoch[prepost]
        spike_trains_restrict=spike_trains.restrict(peri_switch_epoch)[uid_with_same_shank]
        waveform_allspk_alluid={}
        waveform_avgspk_alluid = {}
        for k,val in spike_trains_restrict.items():
            ch=int(cell_metrics['maxWaveformCh'][k-1])
            waveform_allspk_oneuid=[]

            for t in val.index:
                loaded_data=anasigproxy.load(time_slice=(t-0.001,t+0.001))
                loaded_data = loaded_data*to_microV_factor
                waveform_one=loaded_data[:,ch]
                waveform_allspk_oneuid.append(waveform_one)
            if len(waveform_allspk_oneuid)>0:
                waveform_allspk_oneuid=np.array(waveform_allspk_oneuid)[:,:,0]
                waveform_allspk_alluid[k] = waveform_allspk_oneuid
                waveform_avgspk_alluid[k]= waveform_allspk_oneuid.mean(axis=0)
        waveform_allspk_alluid_ep[prepost] = waveform_allspk_alluid
        waveform_avgspk_alluid_ep[prepost] = waveform_avgspk_alluid
    common_uid=np.array([k for k in waveform_allspk_alluid_ep['pre'].keys() if k in waveform_allspk_alluid_ep['post'].keys()])
    for prepost in waveform_allspk_alluid_ep.keys():
        waveform_allspk_alluid_ep[prepost] = {k:waveform_allspk_alluid_ep[prepost][k] for k in common_uid}
        waveform_avgspk_alluid_ep[prepost] = {k:waveform_avgspk_alluid_ep[prepost][k] for k in common_uid}
    
    return waveform_allspk_alluid_ep, waveform_avgspk_alluid_ep


if __name__ =="__main__":
    main()