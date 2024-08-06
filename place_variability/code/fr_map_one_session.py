import os
import sys
import traceback
import numpy as np
import scipy.io as sio
# Import other required libraries
import pandas as pd
sys.path.append('/mnt/home/szheng/projects/util_code')
sys.path.append('/mnt/home/szheng/projects/nmf_analysis')
import copy,pdb
import matplotlib.pyplot as plt
import misc
import database 
import data_prep_pyn as dpp
import place_cell_analysis as pa
import place_field_analysis as pf

# filter db
# subdb = database.db.query("owner=='roman'").sort_values('n_pyr_putative',ascending=False)
subdb = database.db.sort_values('n_pyr_putative',ascending=False)


SAVE_DIR=''
# SAVE_FN='fr_map.p'
SAVE_FN='fr_map_with_int.p'
force_reload=True


def load_preprocess_data(session_path):
    prep_res = dpp.load_spk_beh_df(session_path,force_reload=False,extra_load=dict(sessionPulses='*SessionPulses.Events.mat',filtered='*thetaFiltered.lfp.mat'))
    spk_beh_df=prep_res['spk_beh_df']
    _,spk_beh_df = dpp.group_into_trialtype(spk_beh_df)
    # spk_beh_df,_=dpp.add_lin_binned(spk_beh_df,nbins=100)
    spk_beh_df,_=dpp.add_lin_binned(spk_beh_df,bin_size=2.2,nbins=None)
    cell_cols_d = prep_res['cell_cols_d']
    
    data = {'spk_beh_df':spk_beh_df,'cell_cols_d':cell_cols_d}
    return data

def analyze_data(data,*args,**kwargs):
    speed_key=kwargs.get('speed_key','directed_locomotion')
    speed_thresh=kwargs.get('speed_thresh',0.5)
    gauss_width=kwargs.get('gauss_width',2.5)
    # Perform your main analysis
    cell_cols_d = data['cell_cols_d']
    cell_cols = data['cell_cols_d']['pyr']
    spk_beh_df = data['spk_beh_df']
    # speed_thresh=0.5
    fr_map_dict=pa.get_fr_map_trial(spk_beh_df,cell_cols,gauss_width=gauss_width,trialtype_key='trial_type',speed_key=speed_key,speed_thresh=speed_thresh,order=['smooth','divide','average'])
    fr_map_d={k:val[0] for k,val in fr_map_dict.items()}
    occu_map_d={k:val[2] for k,val in fr_map_dict.items()}
    fr_map_df_all = pd.concat(fr_map_d,axis=0)
    
    occu_d={}
    for k,val in fr_map_dict.items():
        occu_d[k]=pd.Series(np.squeeze(val[2]),index=fr_map_dict[k][0].columns)
    occu_d = pd.concat(occu_d,axis=0)

    fr_map_trial_dict=pa.get_fr_map_trial(spk_beh_df,cell_cols,gauss_width=gauss_width,trialtype_key='trial_type',speed_key=speed_key,speed_thresh=speed_thresh,order=['smooth','divide'])
    fr_map_trial_d = {k:val[0] for k,val in fr_map_trial_dict.items()}
    fr_map_trial_df_all=pd.concat({k:pf.fr_map_trial_to_df(fr_map_trial_d[k],cell_cols) for k in fr_map_dict.keys()},axis=0)

    index_within_to_trial_index_df=dpp.index_within_to_trial_index(spk_beh_df)

    
    cell_cols = cell_cols_d['pyr']
    speed_thresh=0.5
    fr_map_dict=pa.get_fr_map_trial(spk_beh_df,cell_cols,gauss_width=gauss_width,trialtype_key='trial_type',speed_key=speed_key,speed_thresh=speed_thresh,order=['smooth','divide','average'])
    fr_map_d={k:val[0] for k,val in fr_map_dict.items()}
    fr_map_df_all = pd.concat(fr_map_d,axis=0)
    
    fr_map_trial_dict=pa.get_fr_map_trial(spk_beh_df,cell_cols,gauss_width=gauss_width,trialtype_key='trial_type',speed_key=speed_key,speed_thresh=speed_thresh,order=['smooth','divide'])
    fr_map_trial_d = {k:val[0] for k,val in fr_map_trial_dict.items()}
    fr_map_trial_df_all=pd.concat({k:pf.fr_map_trial_to_df(fr_map_trial_d[k],cell_cols) for k in fr_map_dict.keys()},axis=0)
    
    trialtype_levels=fr_map_trial_df_all.index.nlevels - 2 # usually 4-2=2, for thomas (no trialtype), 3-2=1
    level=tuple(np.arange(trialtype_levels))
    gpb=fr_map_trial_df_all.groupby(level=level,sort=False,group_keys=False)
    fr_map_trial_df_all_=[]
    for k,val in gpb:
        # pdb.set_trace()
        fr_map_trial_df_all_.append(val[index_within_to_trial_index_df.loc[k].index])
    fr_map_trial_df_all_ = pd.concat(fr_map_trial_df_all_,axis=0)
    fr_map_trial_df_all = fr_map_trial_df_all_

    if 'int' in cell_cols_d.keys():
        cell_cols = cell_cols_d['int']
        speed_thresh=0.5
        fr_map_dict=pa.get_fr_map_trial(spk_beh_df,cell_cols,gauss_width=gauss_width,trialtype_key='trial_type',speed_key=speed_key,speed_thresh=speed_thresh,order=['smooth','divide','average'])
        fr_map_d={k:val[0] for k,val in fr_map_dict.items()}
        fr_map_df_all_int = pd.concat(fr_map_d,axis=0)

        fr_map_trial_dict=pa.get_fr_map_trial(spk_beh_df,cell_cols,gauss_width=gauss_width,trialtype_key='trial_type',speed_key=speed_key,speed_thresh=speed_thresh,order=['smooth','divide'])
        fr_map_trial_d = {k:val[0] for k,val in fr_map_trial_dict.items()}
        fr_map_trial_df_all_int=pd.concat({k:pf.fr_map_trial_to_df(fr_map_trial_d[k],cell_cols) for k in fr_map_dict.keys()},axis=0)

        gpb=fr_map_trial_df_all_int.groupby(level=(0,1),sort=False,group_keys=False)
        fr_map_trial_df_all_int_=[]
        for k,val in gpb:
            fr_map_trial_df_all_int_.append(val[index_within_to_trial_index_df.loc[k].index])
        fr_map_trial_df_all_int_ = pd.concat(fr_map_trial_df_all_int_,axis=0)
        fr_map_trial_df_all_int = fr_map_trial_df_all_int_
        fr_map_trial_df_all_both=pd.concat([fr_map_trial_df_all_int,fr_map_trial_df_all],axis=0).sort_index(level=2)

    # combine trialtypes
    cell_cols = cell_cols_d['pyr']
    speed_thresh=0.5
    fr_map_trial_dict=pa.get_fr_map_trial(spk_beh_df,cell_cols,gauss_width=gauss_width,trialtype_key='task_index',speed_key=speed_key,speed_thresh=speed_thresh,order=['smooth','divide'])
    fr_map_trial_d = {k:val[0] for k,val in fr_map_trial_dict.items()}
    fr_map_trial_df_bothtt_pyr=pd.concat({k:pf.fr_map_trial_to_df(fr_map_trial_d[k],cell_cols) for k in fr_map_trial_d.keys()},axis=0)

    val_l=[]
    for k,val in fr_map_trial_df_bothtt_pyr.groupby(level=0):

        val_l.append(val.loc[:,index_within_to_trial_index_df.loc[k].sort_values().values])

    fr_map_trial_df_bothtt_pyr = pd.concat(val_l,axis=0)

    if 'int' in cell_cols_d.keys():
        cell_cols = cell_cols_d['int']
        speed_thresh=0.5
        fr_map_trial_dict=pa.get_fr_map_trial(spk_beh_df,cell_cols,gauss_width=gauss_width,trialtype_key='task_index',speed_key=speed_key,speed_thresh=speed_thresh,order=['smooth','divide'])
        # fr_map_trial_dict=pa.get_fr_map_trial(spk_beh_df,cell_cols,gauss_width=gauss_width,trialtype_key='trial_type',speed_key=speed_key,speed_thresh=speed_thresh,order=['smooth','divide'])
        fr_map_trial_d = {k:val[0] for k,val in fr_map_trial_dict.items()}
        fr_map_trial_df_bothtt_int=pd.concat({k:pf.fr_map_trial_to_df(fr_map_trial_d[k],cell_cols) for k in fr_map_trial_d.keys()},axis=0)
        val_l=[]
        for k,val in fr_map_trial_df_bothtt_int.groupby(level=0):
            
            val_l.append(val.loc[:,index_within_to_trial_index_df.loc[k].sort_values().values])
        fr_map_trial_df_bothtt_int = pd.concat(val_l,axis=0)

    # combine trialtypes and both
    fr_map_trial_df_bothtt_pyr_ = copy.copy(fr_map_trial_df_bothtt_pyr)
    fr_map_trial_df_bothtt_pyr_['trialtype']='both'
    fr_map_trial_df_bothtt_pyr_=fr_map_trial_df_bothtt_pyr_.set_index('trialtype',append=True).swaplevel(3,2).swaplevel(2,1)
    fr_map_trial_df_pyr_combined=pd.concat([fr_map_trial_df_all,fr_map_trial_df_bothtt_pyr_])

    


    res = {'fr_map':fr_map_df_all,'fr_map_trial':fr_map_trial_d,'fr_map_trial_df':fr_map_trial_df_all,'occu_map':occu_d,
        'fr_map_trial_df_pyr_combined':fr_map_trial_df_pyr_combined,
    }
    if 'int' in cell_cols_d.keys():
        fr_map_trial_df_bothtt_int_ = copy.copy(fr_map_trial_df_bothtt_int)
        fr_map_trial_df_bothtt_int_['trialtype']='both'
        fr_map_trial_df_bothtt_int_=fr_map_trial_df_bothtt_int_.set_index('trialtype',append=True).swaplevel(3,2).swaplevel(2,1)
        fr_map_trial_df_int_combined=pd.concat([fr_map_trial_df_all_int,fr_map_trial_df_bothtt_int_])

        res['fr_map_trial_df_int_combined'] = fr_map_trial_df_int_combined
        res['fr_map_trial_df_pyr_int'] = fr_map_trial_df_all_both
    return res

# def save_results(results, session_path, output_folder):
#     # Save your results to a file
#     pass

def main(session_path,test_mode=False,
        analysis_args=[],
        analysis_kwargs={},
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
