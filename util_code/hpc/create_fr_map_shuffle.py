import sys,os
sys.path.append('../')
sys.path.append('/mnt/home/szheng/projects/util_code')
import place_cell_analysis as pa
import place_field_analysis as pf
import pandas as pd
import data_prep_pyn as dpp
import misc

DATABASE_LOC = '/mnt/home/szheng/ceph/ad/database.csv'
db = pd.read_csv(DATABASE_LOC,index_col=[0,1])

sess_to_plot = db.query("owner=='roman'").sort_values('n_pyr_putative',ascending=False)
print(f'{sess_to_plot.shape[0]} sessions!')
# SAVE_FN = 'fr_map_null_trialtype.p'
SAVE_FN = 'fr_map_null_trialtype_vthresh.p' # using v as threshold
SPEED_KEY = 'v'
def main(i,testmode=False):
    # sess_name = sess_to_plot.iloc[i]['sess_name']
    data_dir_full = sess_to_plot.iloc[i]['data_dir_full']
    # data_dir_full = sess_to_plot.query(f'sess_name=="{sess_name}"').loc[:,'data_dir_full'].iloc[0]
    if testmode:
        N_shifts = 1
    else:
        N_shifts = 1000
    # res,unit_names = pa.get_shuffle_fr_map(sess_name,data_dir_full,N_shifts=N_shifts,dosave=True,speedmask=1)
    
    res = pf.get_fr_map_shuffle_wrapper(data_dir_full,nrepeats=N_shifts, dosave=True,force_reload=True,nbins = 100,save_fn =SAVE_FN, speed_key=SPEED_KEY)
    

if __name__ == '__main__':
    args = sys.argv[1:]
    print(args)
    main(int(args[0]))
