# %%
import numpy as np
import scipy
import data_prep_new as dpn
import place_cell_analysis as pa
import plot_helper as ph
from importlib import reload
import itertools, sys, os, copy, pickle
import matplotlib.pyplot as plt
import pandas as pd
import submitit
reload(dpn)

# %%

DATABASE_LOC = '/mnt/home/szheng/ceph/ad/database.csv'
db = pd.read_csv(DATABASE_LOC,index_col=[0,1])

def main(db,i,sess_name=None,data_dir_full=None,dosave=True):
    if data_dir_full is None:
        if sess_name is None:
            sess_name = db.iloc[i]['sess_name']
        
        data_dir_full = db.query(f'sess_name=="{sess_name}"').loc[:,'data_dir_full'].iloc[0]
    
    cell_metric,behavior,spike_times,uid,fr,cell_type,mergepoints,behav_timestamps,position,\
                    rReward,lReward,endDelay,startPoint,visitedArm \
        = dpn.load_sess(sess_name=sess_name, data_dir=None, data_dir_full=data_dir_full)

    
    df_dict, pos_bins_dict,cell_cols_dict = dpn.get_fr_beh_df(spike_times,uid,behav_timestamps,cell_type,position,visitedArm,startPoint,n_pos_bins=100)
    
    dt = df_dict['pyr']['times'].iloc[2] - df_dict['pyr']['times'].iloc[1]

    all_trials_frmap_result = pa.get_fr_map_trial(df_dict['pyr'],cell_cols_dict['pyr'],gauss_width=2.5,order=['smooth','divide'])

    fr_map_all_trials_dict = {0:all_trials_frmap_result[0][0],1:all_trials_frmap_result[1][0]}
    count_map_all_trials_dict = {0:all_trials_frmap_result[0][1],1:all_trials_frmap_result[1][1]}
    occupancy_map_all_trials_dict = {0:all_trials_frmap_result[0][2] * dt,1:all_trials_frmap_result[1][2] * dt}

    frmap_result = pa.get_fr_map_trial(df_dict['pyr'],cell_cols_dict['pyr'],gauss_width=2.5,order=['smooth','divide','average'],n_lin_bins=100)

    fr_map_dict = {0:frmap_result[0][0],1:frmap_result[1][0]}
    count_map_dict = {0:frmap_result[0][1],1:frmap_result[1][1]}
    occupancy_map_dict = {0:frmap_result[0][2] *dt,1:frmap_result[1][2] * dt}

    res_fr = {'df':df_dict,'pos_bins':pos_bins_dict,'cell_cols':cell_cols_dict}
    res_fr_map = {'fr_map':fr_map_dict,'count_map':count_map_dict,'occupancy_map':occupancy_map_dict, \
                'fr_map_trial':fr_map_all_trials_dict, 'count_map_trial':count_map_all_trials_dict,'occupancy_map_trial':occupancy_map_all_trials_dict}

    if dosave:
        save_data_dir = os.path.join(data_dir_full,'py_data')
        if not os.path.exists(save_data_dir):
            os.mkdir(save_data_dir)
            print(f'{save_data_dir} made!')
        fn_full = os.path.join(save_data_dir,'fr_map.p')
        pickle.dump(res_fr_map,open(fn_full,'wb'))
        print(f'{fn_full} saved!')

        fn_full = os.path.join(save_data_dir,'fr.p')
        pickle.dump(res_fr,open(fn_full,'wb'))
        print(f'{fn_full} saved!')

        # np.savez(fn_full,**res,unit_names=unit_names)

    return res_fr, res_fr_map

# %%
from tqdm import tqdm
# %% 
# MAIN OPERATION!
if __name__=="__main__":
    failed_list=[]
    # db_sub = db.loc[['AZ10','AZ11','Naz1','Naz2']]
    db_sub = db.loc[['e13_26m1']]
    for i in tqdm(range(db_sub.shape[0])):
        try:
            main(db_sub,i,True)
        except:
            failed_list.append(i)
    # %% POST PROCESSING
    # fn="/mnt/home/szheng/ceph/ad/Chronic_H2/AZ10/AZ10_210315_sess1/py_data/fr.p"
    # res_fr=pickle.load(open(fn,'rb'))
    # %%
    db.iloc[failed_list].to_pickle('save_fr_failed_1.p')
    # %%
    # dbfailed_list =pd.read_pickle('save_fr_failed.p')


# %% EXAMINING THE FAILED SESSIONS
# sess_name = dbfailed_list['sess_name'].iloc[4]
# data_dir_full = db.query(f'sess_name=="{sess_name}"').loc[:,'data_dir_full'].iloc[0]

# cell_metric,behavior,spike_times,uid,fr,cell_type,mergepoints,behav_timestamps,position,\
#                 rReward,lReward,endDelay,startPoint,visitedArm \
#     = dpn.load_sess(sess_name=sess_name, data_dir=None, data_dir_full=data_dir_full)

# df_dict, pos_bins_dict,cell_cols_dict = dpn.get_fr_beh_df(spike_times,uid,behav_timestamps,cell_type,position,visitedArm,startPoint,n_pos_bins=100)
# dt = df_dict['pyr']['times'].iloc[2] - df_dict['pyr']['times'].iloc[1]

# all_trials_frmap_result = pa.get_fr_map_trial(df_dict['pyr'],cell_cols_dict['pyr'],gauss_width=2.5,order=['smooth','divide'])

# fr_map_all_trials_dict = {0:all_trials_frmap_result[0][0],1:all_trials_frmap_result[1][0]}
# count_map_all_trials_dict = {0:all_trials_frmap_result[0][1],1:all_trials_frmap_result[1][1]}
# occupancy_map_all_trials_dict = {0:all_trials_frmap_result[0][2] * dt,1:all_trials_frmap_result[1][2] * dt}

# frmap_result = pa.get_fr_map_trial(df_dict['pyr'],cell_cols_dict['pyr'],gauss_width=2.5,order=['smooth','divide','average'],n_lin_bins=100)

# fr_map_dict = {0:frmap_result[0][0],1:frmap_result[1][0]}
# count_map_dict = {0:frmap_result[0][1],1:frmap_result[1][1]}
# occupancy_map_dict = {0:frmap_result[0][2] *dt,1:frmap_result[1][2] * dt}

# res_fr = {'df':df_dict,'pos_bins':pos_bins_dict,'cell_cols':cell_cols_dict}
# res_fr_map = {'fr_map':fr_map_dict,'count_map':count_map_dict,'occupancy_map':occupancy_map_dict, \
#             'fr_map_trial':fr_map_all_trials_dict, 'count_map_trial':count_map_all_trials_dict,'occupancy_map_trial':occupancy_map_all_trials_dict}

# %%
