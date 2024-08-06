import h5py
import numpy as np
import sys
# sys.path.append('/mnt/home/szheng/projects/seq_detection2/code')
sys.path.append('/mnt/home/szheng/projects/util_code')
sys.path.append('/mnt/home/szheng/projects/nmf_analysis')
sys.path.append('/mnt/home/szheng/projects/nmf_analysis/scripts')
sys.path.append('/mnt/home/szheng/projects/cluster_spikes')
sys.path.append('/mnt/home/szheng/projects/place_variability/code')

import misc
import prep_thomas_data as ptd
import os


FN_FULL = '/mnt/home/szheng/ceph/ad/thomas_data/180301_DG_CA3_CA1.h5'
thomas_dir_base = "/mnt/home/szheng/ceph/ad/thomas_data/180301_DG_CA3_CA1"

DO_MASK_TRANSIENT = False
# SAVE_FN = 'preprocessed.p'
SAVE_FN = f'preprocessed_mask_{DO_MASK_TRANSIENT}.p'

REGION_L = ['CA1','CA3','DG']


def main(i,test_mode=False):
    save_fn = SAVE_FN
    region = REGION_L[i]
    f = h5py.File(FN_FULL, 'r')
    n_exp = f['data'][region].shape[0]
    n_sess_l = []
    n_days_l = []
    failed_l = []

    # if test_mode:
    #     n_exp = 1
    for exp_ind in range(n_exp):
        ref=f['data'][region][exp_ind,0]
        n_sess = f[ref]['metadata']['categories'].shape[0]
        n_sess_l.append(n_sess)
        n_days = int(n_sess // 2)
        n_days_l.append(n_days)
    #     f[ref]
        # if test_mode:
        #     n_days=1
        for day_ind in range(n_days):
            try:
                prep_res = ptd.prep_one_day_within_one_exp(f,exp_ind,day_ind,region=region,do_mask_transient=DO_MASK_TRANSIENT)
                save_dir = misc.get_or_create_subdir(thomas_dir_base, region,f'exp_{exp_ind}',f'day_{day_ind}')
                save_fn_full, _ = misc.get_res(save_dir,save_fn,True)
                misc.save_res(save_fn_full,prep_res)

            except:
                failed_l.append((region,exp_ind,day_ind))
    
    meta = {'n_sess':np.array(n_sess_l),'n_days':np.array(n_days_l),'failed':np.array(failed_l)}
    save_meta_dir = misc.get_or_create_subdir(thomas_dir_base, region)
    save_meta_fn = os.path.join(save_meta_dir,'n_sess.p')
    misc.save_res(save_meta_fn,meta)
    return meta


if __name__ == "__main__":
    region_ind = int(sys.argv[1])
    test_mode = bool(sys.argv[2])
    main(region_ind, test_mode=test_mode)






