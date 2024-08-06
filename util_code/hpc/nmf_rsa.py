import numpy as np
import scipy
import scipy.io
import matplotlib.pyplot as plt
import copy,sys,os,pdb,importlib
from importlib import reload
import pandas as pd
import seaborn as sns
import pickle

sys.path.append('../')
sys.path.append('../..')
import util_code
from util_code import data_prep_new as dpn
from util_code import place_cell_analysis as pa
# import place_cell_analysis as pa

import scipy.ndimage
from scipy.ndimage import gaussian_filter1d
import sklearn
from sklearn.decomposition import NMF

data_dir = '/mnt/home/szheng/ceph/ad/Chronic_H2/'
n_pos_bins = 100

def main(testmode=False):

    animal_sess_l = []
    corr_mat_l = []
    for animal in os.listdir(data_dir):
        data_dir_animal = os.path.join(data_dir,animal)
        if os.path.isdir(data_dir_animal):
            fn_l= os.listdir(data_dir_animal)
            fn_l.sort()
            for fn in fn_l:
                if testmode:
                    if len(corr_mat_l) > 2:
                        break
                if 'sess' in fn:
                    
                    try:
                        corr_mat = pa.get_pop_rep_corr(fn,data_dir,toplot=False,corr=False,n_pos_bins=n_pos_bins)
                        corr_mat_l.append(corr_mat)
                        animal_sess_l.append((animal,fn))
                    except:
                        print(f'{fn} failed')
    animal_sess_l = np.array(animal_sess_l)
    corr_mat_l = np.array(corr_mat_l,dtype=object)

    corr_mat_flattened_l = []
    animal_sess_l_filtered = []
    for ii,corr_mat in enumerate(corr_mat_l):
        if corr_mat.shape[0]==2*n_pos_bins:
            corr_mat_triu_flattened = corr_mat.values[np.triu_indices(corr_mat.shape[0])]
            animal_sess_l_filtered.append(animal_sess_l[ii])
            corr_mat_flattened_l.append(corr_mat_triu_flattened)
    corr_mat_flattened_l = np.stack(corr_mat_flattened_l,axis=0)
    fn = os.path.join(data_dir,'corr_mat_flattened_l.npy')
    np.save(fn,corr_mat_flattened_l)
    print(f'Saved at {fn}!')


    if testmode:
        corr_mat_flattened_l = corr_mat_flattened_l[:2,:100]
    model = NMF()
    W = model.fit_transform(corr_mat_flattened_l)
    H = model.components_
    result = {'model':model,'W':W,'H':H}
    fn = os.path.join(data_dir,'corr_mat_flattened_nmf.p')
    pickle.dump(result,open(fn,'wb'))
    print(f'Saved at {fn}')
    return 

if __name__ == '__main__':
    testmode = bool(int(sys.argv[1]))
    main(testmode)

    
