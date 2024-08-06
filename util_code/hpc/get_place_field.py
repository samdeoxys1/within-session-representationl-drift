import numpy as np
import scipy
import pandas as pd
import sys,os,copy,itertools,pdb,pickle,tqdm
sys.path.append('/mnt/home/szheng/projects/util_code')
import place_field_analysis as pf
from scipy.ndimage import gaussian_filter1d

DATABASE_LOC = '/mnt/home/szheng/ceph/ad/database.csv'
db = pd.read_csv(DATABASE_LOC,index_col=[0,1])

sess_to_plot = db.query("owner=='roman'").sort_values('n_pyr_putative',ascending=False)
print(f'{sess_to_plot.shape[0]} sessions!')

# SAVE_FN = 'place_field_avg_and_trial.p'
SAVE_FN = 'place_field_avg_and_trial_vthresh.p'
# SHUFFLE_FN = 'fr_map_null_trialtype.p'
SHUFFLE_FN = 'fr_map_null_trialtype_vthresh.p'

def main(i,testmode=False):
    
    data_dir_full = sess_to_plot.iloc[i]['data_dir_full']
    
    place_field_res=pf.field_detection_both_avg_trial_wrapper(data_dir_full, dosave=True,force_reload=True,nbins = 100, 
                                        save_fn = SAVE_FN, 
                                        shuffle_fn=SHUFFLE_FN,
                                        smth_in_bin=2.5, speed_thresh=1.,
                                        load_only=False,
                                        )

    return place_field_res
    

if __name__ == '__main__':
    args = sys.argv[1:]
    print(args)
    main(int(args[0]))

