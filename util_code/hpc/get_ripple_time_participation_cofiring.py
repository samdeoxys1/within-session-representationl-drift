import sys
sys.path.append('/mnt/home/szheng/projects/seq_detection2/code')
sys.path.append('/mnt/home/szheng/projects/util_code')
sys.path.append('/mnt/home/szheng/projects/nmf_analysis')
sys.path.append('/mnt/home/szheng/projects/nmf_analysis/scripts')
sys.path.append('/mnt/home/szheng/projects/cluster_spikes')
import misc
import preprocess as prep
from importlib import reload
reload(prep)
import data_prep_pyn as dpp
import database 
import pandas as pd
import numpy as np

db = database.db
sub_db = db.query("owner=='roman'").sort_values('n_pyr_putative',ascending=False)

def get_ripple_time_participation_cofiring_wrapper(data_dir_full,
                                                do_save=True,force_reload=False,load_only=False,
                                                save_fn = 'ripple_time_participation_cofiring.p',
                                                save_dir='py_data',
                                                            ):
    
    res_to_save_dir = misc.get_or_create_subdir(data_dir_full,save_dir)
    fn_full,res = misc.get_res(res_to_save_dir,save_fn,force_reload)
    if (res is not None) or load_only:
        return res
    
    mat_to_return=prep.load_stuff(data_dir_full)
    cell_metrics = mat_to_return['cell_metrics']
    ripple_events = mat_to_return['ripples']
    prep_res = dpp.load_spk_beh_df(data_dir_full,force_reload=False,extra_load={})
    cell_cols_d = prep_res['cell_cols_d']
    mergepoints = mat_to_return.mergepoints
    cell_type_mask=mat_to_return['cell_type_mask']
    ripple_time_ints_l_d_epochs, participation_mask_l_d_epochs,spike_times_in_ripple_l_d= prep.get_ripple_time_interval_and_participation_mask(cell_metrics,ripple_events,mergepoints,cell_type_mask)
    sim_df_l_d = prep.get_ripple_cofiring_similarity(participation_mask_l_d_epochs,cell_cols_d['pyr']) # so far only been looking at pyr; corresponding to cell_type_mask above
    
    res_to_save = {
        'sim':sim_df_l_d,
        'time_intervals':ripple_time_ints_l_d_epochs,
        'participation':participation_mask_l_d_epochs,
        'spikes':spike_times_in_ripple_l_d
    }
    misc.save_res(fn_full,res_to_save,dosave=do_save)
    return res_to_save

def main(i):
    data_dir_full = sub_db['data_dir_full'][i]
    get_ripple_time_participation_cofiring_wrapper(data_dir_full,
                                                do_save=True,force_reload=False,load_only=False,
                                                save_fn = 'ripple_time_participation_cofiring.p'
                                                            )

if __name__=="__main__":
    args = sys.argv[1:]
    print(args)
    main(int(args[0]))
