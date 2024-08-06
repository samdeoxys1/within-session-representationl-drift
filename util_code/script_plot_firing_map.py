import numpy as np
import scipy
import pandas as pd
import copy,os,sys
import matplotlib.pyplot as plt
import seaborn
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')

import data_prep_new as dpn
import place_cell_analysis as pa

sess_name='Naz2_210323_sess1' #'AZ12_210415_sess8'

animal_name = sess_name.split('_')[0]
data_dir = '/mnt/home/szheng/ceph/ad/Chronic_H2/'
figdir = os.path.join(data_dir,animal_name,sess_name,'py_fig','firing_map')
if not os.path.exists(figdir):
    os.makedirs(figdir)
    print(f'{figdir} created!')

def main():
    spike_times,uid,fr,cell_type,mergepoints,behav_timestamps,position,\
                rReward,lReward,endDelay,startPoint,visitedArm \
    = dpn.load_sess(sess_name, data_dir=data_dir)
    ncells=len(uid)
    fig_ax_l = [plt.subplots() for i in range(ncells)]
    df_dict, pos_bins_dict,cell_cols_dict = dpn.get_fr_beh_df(spike_times,uid,behav_timestamps,cell_type,position,visitedArm,startPoint,n_pos_bins=100)

    order_l = [['divide','smooth','average'],['average','divide','smooth'],['average','smooth','divide'],['smooth','divide','average']]
    linestyle_l = {1:'-',0:':'}
    color_l = [f'C{ii}' for ii in range(len(order_l))]
    for cel,df in df_dict.items():
        for oo,order in enumerate(order_l):
            fr_map_final_dict = pa.get_fr_map_trial(df,cell_cols_dict[cel],gauss_width=2.5,order=order)
            for unit_id in tqdm(cell_cols_dict[cel]):
                for choice,fr_map_final in fr_map_final_dict.items():
                    unit_id_int = int(unit_id.split('_')[1]) - 1
                    fig_ax_l[unit_id_int][1].plot(pos_bins_dict['lin'][1:],fr_map_final.loc[unit_id],linestyle=linestyle_l[choice],color=color_l[oo],label=f'{order};choice={choice}')
                fig_ax_l[unit_id_int][1].set_title(f'{unit_id},{cel}') # 1 for pyr
                fig_ax_l[unit_id_int][1].legend(bbox_to_anchor=[1,1.05])
                fig_ax_l[unit_id_int][0].savefig(os.path.join(figdir,f'{unit_id}.pdf'),bbox_inches='tight')
    

if __name__ == '__main__':
    main()



