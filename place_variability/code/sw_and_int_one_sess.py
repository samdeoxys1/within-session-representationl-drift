import os
import sys
import traceback
import numpy as np
import scipy.io as sio
# Import other required libraries
import pandas as pd
sys.path.append('/mnt/home/szheng/projects/util_code')
sys.path.append('/mnt/home/szheng/projects/nmf_analysis')
import copy,pdb,pickle
import matplotlib.pyplot as plt
import misc
import database 
import data_prep_pyn as dpp
import place_cell_analysis as pa
import place_field_analysis as pf
import fr_map_one_session as fmos
import pf_recombine_central as pfrc
import fr_map_one_session as fmos
import switch_detection_one_session as sdos
import peer_prediction as pp
import get_all_switch_add_metrics as gasam
import get_all_switch_add_metrics_pen as gasamp

# filter db
# subdb = database.db.query("owner=='roman'").sort_values('n_pyr_putative',ascending=False)
subdb = database.db.sort_values('n_pyr_putative',ascending=False)


SAVE_DIR=''
# SAVE_FN='fr_map.p'
SAVE_FN='sw_and_int.p'
force_reload=True
N_LASTING_TRIAL=4
TASK_INDEX=TI=0
DO_NORM=True
PEN=0.5 # None # None means shuffle test, number means using pelt
if PEN is not None:
    SAVE_FN=f'sw_and_int_pen_{PEN:1.0e}.p'

def load_preprocess_data(session_path):
    fr_map_res=fmos.main(session_path,force_reload=False,load_only=True)
    fr_map_trial_df_int_combined = fr_map_res['fr_map_trial_df_int_combined']

    pf_res_recombine = pfrc.main(session_path,force_reload=False,load_only=True)
    all_fields_recombined=pf_res_recombine['all_fields_recombined']
    all_fields_one_sess=all_fields_recombined
    
    sw_res = sdos.main(session_path,force_reload=False,load_only=True)
    X_raw = sw_res['X_raw']
    
    prep_res = dpp.load_spk_beh_df(session_path,force_reload=False,extra_load=None)
    spk_beh_df=prep_res['spk_beh_df']
    _,spk_beh_df = dpp.group_into_trialtype(spk_beh_df)
    spk_beh_df,_=dpp.add_lin_binned(spk_beh_df,bin_size=2.2,nbins=None)
    cell_cols_d = prep_res['cell_cols_d']
    index_within_to_trial_index_df = dpp.index_within_to_trial_index(spk_beh_df)
    if PEN is None:
        sw_info_res=gasam.main(session_path,force_reload=False,load_only=True)
        if sw_info_res is None:
            sw_info_res=gasam.main(session_path,force_reload=True,load_only=False)
        all_sw_with_metrics = sw_info_res['all_sw_with_metrics_d']
    else: # the penalty case
        sw_info_res=gasamp.main(session_path,force_reload=False,load_only=True)
        if sw_info_res is None:
            sw_info_res=gasamp.main(session_path,force_reload=True,load_only=False)
        all_sw_with_metrics = sw_info_res['all_sw_with_metrics_d_pen'][PEN]


    pf_res_recombine = pfrc.main(session_path,force_reload=False,load_only=True)
    all_fields_one_sess=pf_res_recombine['all_fields_recombined']

    data_dir = '/mnt/home/szheng/ceph/place_variability/data/paper'
    savefn_full = os.path.join(data_dir,'per_field_metrics_shtest_with_1cp.p')
    # pickle.dump(per_field_metrics_all,open(savefn_full,'wb'))
    per_field_metrics_all=pickle.load(open(savefn_full,'rb'))
    ani,sess=subdb.loc[subdb['data_dir_full'] == session_path][['animal_name.1','sess_name']].values[0]
    per_field_metrics_one = per_field_metrics_all.loc[(ani,sess),:]
    

    data = {'fr_map_trial_df_int_combined':fr_map_trial_df_int_combined,
            'all_fields_one_sess':all_fields_one_sess,
            'X_raw':X_raw,'spk_beh_df':spk_beh_df,'all_sw_with_metrics':all_sw_with_metrics,
            'all_fields_one_sess':all_fields_one_sess,
            'index_within_to_trial_index_df':index_within_to_trial_index_df,
            'fr_map_trial_df_int_combined':fr_map_trial_df_int_combined,
            'cell_cols_d':cell_cols_d,'per_field_metrics_one':per_field_metrics_one
            }
    return data

def analyze_data(data,*args,**kwargs):
    '''
    consec_trial_fr_diff_all: (trialtype, uid, field_id, referece_trial)
    the values are changes relative to the reference trial, k trials after the reference trial
    '''
    X_raw = data['X_raw']
    all_fields_one_sess = data['all_fields_one_sess']
    fr_map_trial_df_int_combined = data['fr_map_trial_df_int_combined']
    spk_beh_df = data['spk_beh_df']
    all_sw_with_metrics = data['all_sw_with_metrics']
    all_fields_one_sess = data['all_fields_one_sess']
    index_within_to_trial_index_df = data['index_within_to_trial_index_df']
    fr_map_trial_df_int_combined = data['fr_map_trial_df_int_combined']                                    
    cell_cols_d = data['cell_cols_d']
    per_field_metrics_one = data['per_field_metrics_one']

    ti = kwargs.get('task_index',TI)
    consec_trial_fr_diff_all = {}
    do_norm=kwargs.get('do_norm',DO_NORM)
    n_lasting_trial = kwargs.get('n_lasting_trial',N_LASTING_TRIAL)
    inh_fr_trial_within_field_all={}
    for (tt,uid,field_id),row in all_fields_one_sess.loc[ti].iterrows():
        st,ed = row['start'],row['end']
        pyr_fr_trial_within_field = X_raw.loc[(ti,tt,uid,field_id),:].dropna().values
        if do_norm:
            pyr_fr_trial_within_field = pyr_fr_trial_within_field / np.max(pyr_fr_trial_within_field)
        pyr_fr_prior_l = []
        for ii in range(1,len(pyr_fr_trial_within_field)):
            pyr_fr_prior = np.mean(pyr_fr_trial_within_field[:ii])
            pyr_fr_prior_l.append(pyr_fr_prior)
        pyr_fr_prior_l.append(np.nan) # up till reference trial (the index value); the last is nan because all the pyr_k and int_k will be nan, so doesn't matter
        pyr_fr_prior_l = np.array(pyr_fr_prior_l)
        
        inh_fr_trial_within_field_ = fr_map_trial_df_int_combined.loc[(ti,tt),:].dropna(axis=1,how='all').loc[(slice(None),slice(st,ed)),:].groupby(level=0).mean().mean(axis=0)
        inh_fr_trial_within_field = inh_fr_trial_within_field_.values
        inh_fr_trial_within_field_all[(tt,uid,field_id)] = inh_fr_trial_within_field_
        inh_fr_trial_within_field_diff = np.diff(inh_fr_trial_within_field,append=np.nan)
        
        diff_l = {}
        diff_l['inh'] = inh_fr_trial_within_field_diff
        for kk in range(1,n_lasting_trial+1):
    #         pyr_diff = np.diff(pyr_fr_trial_within_field,n=kk,append=[np.nan]*kk)
            pyr_diff=pyr_fr_trial_within_field[kk:] - pyr_fr_trial_within_field[:-kk]
            pyr_diff=np.append(pyr_diff,[np.nan]*kk)    
            # pyr_diff=np.concatenate([[np.nan]*kk,pyr_diff])    
            diff_l[f'pyr_{kk}'] = pyr_diff

        for kk in range(1,n_lasting_trial+1): # need to group columns of pyr and int to be together for downstream analysis
            int_diff = inh_fr_trial_within_field[kk:] - inh_fr_trial_within_field[:-kk]
            int_diff = np.append(int_diff,[np.nan]*kk)
            # int_diff = np.concatenate([[np.nan]*kk,int_diff])
            diff_l[f'int_{kk}']=int_diff

        diff_l = pd.DataFrame(diff_l)
        diff_l['fr_prior'] = pyr_fr_prior_l
        consec_trial_fr_diff_all[(tt,uid,field_id)] = diff_l
        
    inh_fr_trial_within_field_all = pd.concat(inh_fr_trial_within_field_all,axis=0).unstack()
    # consec_trial_fr_diff_all = pd.concat(consec_trial_fr_diff_all,axis=0).dropna(axis=0)
    consec_trial_fr_diff_all = pd.concat(consec_trial_fr_diff_all,axis=0)
    test_res_d = post_tests(consec_trial_fr_diff_all,**kwargs)
    

    res = {'consec_trial_fr_diff_all':consec_trial_fr_diff_all}
    res.update(test_res_d)

    # do peer prediction first
    mean_within_field_pred_all,glm_res_df_all,r2_all = pp.sweep_fit_glm_predict_rate_change(spk_beh_df,all_sw_with_metrics,all_fields_one_sess,
                                    index_within_to_trial_index_df,
                                    fr_map_trial_df_int_combined,
                                    cell_cols_d,                                        
                                    do_inh_only=True,
                                    do_weighted_pred=False,
                                    pval_thresh=None,
                                    ti=ti)

    # reload(pp)
    
    all_sw_with_metrics_oneti_with_inh_change= pp.add_inh_fr_change_to_all_sw(all_sw_with_metrics,mean_within_field_pred_all,per_field_metrics_one,ti=0)

    res['all_int_mean_fr_within_field'] = inh_fr_trial_within_field_all
    res['selected_int_mean_fr_within_field'] = mean_within_field_pred_all
    res['glm_res_df'] = glm_res_df_all
    res['r2'] = r2_all
    res['all_sw_with_metrics_oneti_with_inh_change'] = all_sw_with_metrics_oneti_with_inh_change
    

    return res



import scipy
def post_tests(consec_trial_fr_diff_all,
                pyr_up_thresh_l = [0.,0.1,0.2,0.3,0.4,0.5],
                inh_thresh=0.,
                pyr_up_thresh_2=0.,
                n_lasting_trial=4,**kwargs
                    ):
    # test sustained change in pyr FR -- disinhibition; comparing change in PYR FR                    
    # given first (reference) trial pyr fr increase, compare pyr fr in later trials, as a function of the inh fr change in first trial  
    test_res_d_d = {}
    for pyr_up_thresh in pyr_up_thresh_l:
        test_res_d = {}
        for kk in range(1,n_lasting_trial+1):
            ma=(consec_trial_fr_diff_all['pyr_1'] > pyr_up_thresh) & (consec_trial_fr_diff_all['inh'] < inh_thresh)
            xx=consec_trial_fr_diff_all.loc[ma][f'pyr_{kk}'] # disinhibition
            ma=(consec_trial_fr_diff_all['pyr_1'] > pyr_up_thresh) & (consec_trial_fr_diff_all['inh'] > -inh_thresh)
            yy=consec_trial_fr_diff_all.loc[ma][f'pyr_{kk}'] # more inhibition
            w_stat,w_p = scipy.stats.ranksums(xx,yy)
            ks_res=scipy.stats.ks_2samp(xx,yy)
            ks_stat,ks_p,ks_loc,ks_sign = ks_res[0],ks_res[1],ks_res.statistic_location, ks_res.statistic_sign
            test_res=pd.Series({'w_stat':w_stat,'w_p':w_p,'ks_stat':ks_stat,'ks_p':ks_p,'ks_loc':ks_loc,'ks_sign':ks_sign})
            test_res_d[kk] = test_res
        test_res_d = pd.concat(test_res_d)
        test_res_d_d[pyr_up_thresh] = test_res_d
    test_res_df = pd.concat(test_res_d_d).unstack()
    sustained_pyr_test_res_df = copy.copy(test_res_df)

    # test transient change in pyr FR -- more inhibition; comparing change in INT FR
    # given first (reference) trial pyr fr increase, compare int fr in first trial, as a function of the pyr change in later trials
    test_res_d_d={}
    for pyr_up_thresh in pyr_up_thresh_l:
        test_res_d = {}
        for k in range(2,n_lasting_trial+1):
            ma=(consec_trial_fr_diff_all['pyr_1'] > pyr_up_thresh) &(consec_trial_fr_diff_all[f'pyr_{k}'] > pyr_up_thresh_2) 
            xx=consec_trial_fr_diff_all.loc[ma][f'inh']
            ma=(consec_trial_fr_diff_all['pyr_1'] > pyr_up_thresh) &(consec_trial_fr_diff_all[f'pyr_{k}'] < -pyr_up_thresh_2) 
            yy=consec_trial_fr_diff_all.loc[ma][f'inh']
            w_stat,w_p = scipy.stats.ranksums(xx,yy)
            ks_res=scipy.stats.ks_2samp(xx,yy)
            ks_stat,ks_p,ks_loc,ks_sign = ks_res[0],ks_res[1],ks_res.statistic_location, ks_res.statistic_sign
            test_res=pd.Series({'w_stat':w_stat,'w_p':w_p,'ks_stat':ks_stat,'ks_p':ks_p,'ks_loc':ks_loc,'ks_sign':ks_sign})
            test_res_d[kk] = test_res
        test_res_d = pd.concat(test_res_d)
        test_res_d_d[pyr_up_thresh] = test_res_d
    test_res_df = pd.concat(test_res_d_d).unstack()
    transient_pyr_test_res_df = copy.copy(test_res_df)

    test_res_d = {'sustained':sustained_pyr_test_res_df,
                'transient':transient_pyr_test_res_df
                    }

    return test_res_d

# def save_results(results, session_path, output_folder):
#     # Save your results to a file
#     pass

def main(session_path,test_mode=False,
        analysis_args=[],
        analysis_kwargs={'task_index':TI,
                        'do_norm':DO_NORM,
                        'n_lasting_trial':N_LASTING_TRIAL,
                            },
        dosave=True, save_dir=SAVE_DIR,save_fn=SAVE_FN, force_reload=force_reload,load_only=False,
    ):

    try:
        # create subdir
        save_dir = misc.get_or_create_subdir(session_path,'py_data',save_dir)
        save_fn, res = misc.get_res(save_dir,save_fn,force_reload)
        if (res is not None) or load_only: # load only would skip the computation that follows
            return res
        data = load_preprocess_data(session_path)
        if test_mode:
            # UPDATE SOME PARAMS!!!
            pass
        
        res = analyze_data(data,*analysis_args,**analysis_kwargs)
        misc.save_res(save_fn,res,dosave=dosave)
        return res
        
    except Exception as e:
        tb_str = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
        tb_str.insert(0, f"Error in session: {session_path}\n")
        sys.stderr.writelines(tb_str)

if __name__ == "__main__":
    sess_ind = int(sys.argv[1])
    test_mode = bool(sys.argv[2])
    session_path = subdb['data_dir_full'][sess_ind]
    
    main(session_path, test_mode=test_mode)
