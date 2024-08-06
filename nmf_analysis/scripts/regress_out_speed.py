import numpy as np
import scipy
import pandas as pd
import sys,os,copy,itertools,pdb,pickle,tqdm
sys.path.append('/mnt/home/szheng/projects/util_code')
sys.path.append('/mnt/home/szheng/projects/nmf_analysis')
import place_field_analysis as pf
import change_point_and_behavior as cpb


DATABASE_LOC = '/mnt/home/szheng/ceph/ad/database.csv'
db = pd.read_csv(DATABASE_LOC,index_col=[0,1])

subdb = db.query("owner=='roman'").sort_values('n_pyr_putative',ascending=False)
print(f'{subdb.shape[0]} sessions!')


force_reload = True

def main(i):
    data_dir_full  = subdb.iloc[i]['data_dir_full']
    resid_res=cpb.regress_out_speed_get_residual_wrapper(data_dir_full,dosave=True,force_reload=force_reload,load_only=False,
                                            speed_thresh=1,speed_key='v',
                                            save_fn='place_field_res_speed_residual.p',
                                            nbins=100,fr_key='fr_mean'
                                                )

if __name__ == '__main__':
    args = sys.argv[1:]
    print(args)
    main(int(args[0]))
