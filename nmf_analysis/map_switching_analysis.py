import numpy as np
import scipy
import matplotlib.pyplot as plt
import sys,os,copy,pdb,importlib
from importlib import reload
import pandas as pd
sys.path.append('/mnt/home/szheng/projects/util_code')
import data_prep_pyn as dpp     
from sklearn.cluster import KMeans


def cluster_population_vector_all_posbin(spk_beh_df,cell_cols,smth_kws={},n_clust = 2,bin_size=None,speed_key='v',speed_thresh=1.,make_positive=True,do_norm=True):
    spk_beh_df = copy.copy(spk_beh_df)
    eps = 1e-4
    # preprocess
    if ('lin_binned' not in spk_beh_df.columns) or bin_size is not None:
        spk_beh_df = dpp.add_lin_binned(spk_beh_df,bin_size=bin_size,nbins=None)
    if speed_key is not None:
        spk_beh_df = spk_beh_df.loc[spk_beh_df[speed_key]>=speed_thresh]
    
    # smooth
    smth_kws_ = {'sigma':3,'mode':'nearest'}
    smth_kws_.update(smth_kws)
    spk_beh_df[cell_cols]
    spk_beh_df[cell_cols] = scipy.ndimage.gaussian_filter1d(spk_beh_df[cell_cols].astype(float), smth_kws_['sigma'],axis=0,mode=smth_kws_['mode'])
    spk_beh_df = spk_beh_df.loc[spk_beh_df[speed_key]>=speed_thresh]

    gpb = spk_beh_df.groupby('lin_binned')
    clust_label_original = {}
    fitted_mean_d = {}
    for k,val in gpb:
        X = val[cell_cols]
        if do_norm:
            X_normed = X.astype('float') / (np.linalg.norm(X,axis=1,keepdims=True)) # each vector normalized to norm=1
            X_normed = X_normed.dropna(axis=0)
        else:
            X_normed = X
        kmeans = KMeans(n_clusters=n_clust).fit(X_normed)
        labels = kmeans.labels_
        clust_label_original[k] = pd.Series(labels,index=val.index)
        fitted_mean_d[k] = kmeans.cluster_centers_
        if make_positive:
            fitted_mean_d[k][(fitted_mean_d[k] >= -eps)&(fitted_mean_d[k] < 0)] = 0.
            fitted_mean_d[k][np.abs(fitted_mean_d[k]) <= eps] = 0.
    return fitted_mean_d, clust_label_original
        
def align_two_means(mean_1, mean_2):
    '''
    mean_i: n_fac x n_neuron
    '''
    n_fac = mean_1.shape[0]
    corr = np.corrcoef(mean_1,mean_2)
    corr = corr[n_fac:,:n_fac]

    corr_prop = corr / corr.sum(axis=1,keepdims=True) # ignore negative case for now; proportion of similarity of mean_2 (row) to each mean_1 factor (col)
    
    ind_map_for_2 = np.argmax(corr_prop,axis=0) #compre proportion across mean_2 (row), even if it's not the highest proportion within mean_2, if by proportion bigger than the rest, assign that factor to the corresponding col   # ind_map_for_2[i]: "correct" label for the original label i
    # ind_map_for_2 = np.argmax(corr,axis=1) # ind_map_for_2[i]: "correct" label for the original label i

    noflip = np.all(ind_map_for_2 == np.arange(len(ind_map_for_2)))

    return ind_map_for_2,noflip


def align_maps(fitted_mean_d, clust_label_original,neuron_uid=None):
    '''
    fitted_mean_d: {pos: n_fac x n_neuron}
    clust_label_original: {pos: Series: n_time}
    '''
    n_pos = len(fitted_mean_d.keys())
    pos = list(fitted_mean_d.keys())

    fitted_mean_d_new = {}
    clust_label_new = {}

    fitted_mean_d_new[pos[0]] = fitted_mean_d[pos[0]]
    clust_label_new[pos[0]] = clust_label_original[pos[0]]

    n_clust = fitted_mean_d[pos[0]].shape[0]

    

    for ip in range(n_pos-1):
        p_next = pos[(ip+1)] # no loop, because begin and end might not meet in roman's data

        mean_1 = fitted_mean_d_new[pos[ip]] # relative to the already aligned map
        mean_2 = fitted_mean_d[p_next]
        
        ind_map_for_2,noflip = align_two_means(mean_1, mean_2)
        # pdb.set_trace()
        if not noflip:
            fitted_mean_d_new[p_next] = fitted_mean_d[p_next][ind_map_for_2]
            clust_label_new[p_next] = clust_label_original[p_next].map(lambda x:ind_map_for_2[x])
        else:
            fitted_mean_d_new[p_next] = fitted_mean_d[p_next]
            clust_label_new[p_next] = clust_label_original[p_next]
        
    multiple_maps = {}
    fitted_mean_d_stack = np.stack(list(fitted_mean_d.values()),axis=1)
    for c in range(n_clust):
        multiple_maps[c] = pd.DataFrame(fitted_mean_d_stack[c],index=fitted_mean_d.keys()).T
        if neuron_uid is not None:
            multiple_maps[c].index = neuron_uid
    multiple_maps = pd.concat(multiple_maps,axis=0)
    clust_label_new_combined = pd.concat(clust_label_new,axis=0).reset_index(level=0).rename({'level_0':'lin_binned',0:'label'},axis=1)
    clust_label_new_combined = clust_label_new_combined.sort_index()


    return multiple_maps, clust_label_new_combined,fitted_mean_d_new, clust_label_new,noflip







    
