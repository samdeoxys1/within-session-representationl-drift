import sys,os,pickle,copy,pdb
sys.path.append('/mnt/home/szheng/projects/seq_detection2/code')
sys.path.append('/mnt/home/szheng/projects/util_code')
from time import thread_time_ns
import pandas as pd
import numpy as np
import scipy
import data_prep_new as dpn
import place_cell_analysis as pa
import plot_helper as ph
from importlib import reload
import preprocess as prep
reload(prep)


from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scipy.signal as ss
import tqdm
#===================preprare==========================#

def get_mean_fr_by_trial_types(fr,cell_cols_pyr,mask=None):
    fr_filtered = fr.loc[mask]
    fr_mean_trial_type = fr_filtered.groupby(['visitedArm','trial']).mean()[cell_cols_pyr]
    fr_mean_all = fr_filtered.groupby('trial').mean()[cell_cols_pyr]
    fr_to_be_nmfed = {'0':fr_mean_trial_type.loc[0],'1':fr_mean_trial_type.loc[1],'both':fr_mean_all}
    return fr_to_be_nmfed


def normalize(X):
    '''
    X: n_neurons x n_trials

    normalize by max of row
    pyr_mask: n_neurons, True if row_norm_constant[n] 0
    '''
    row_norm_constant = X.max(axis=1,keepdims=True)
    pyr_mask = np.squeeze(row_norm_constant!=0)
    X_normed = X / row_norm_constant
    non_na_original_ind = np.nonzero(pyr_mask)[0]
    
    return X_normed, pyr_mask, non_na_original_ind 

#===================do nmf========================#
def nmf_multiple_fr_df_once(fr_to_be_nmfed,n_compo):
    
    X_to_be_nmfed = {}
    pyr_mask_d = {}
    non_na_original_ind_d={}
    for k,val in fr_to_be_nmfed.items():
        X_to_be_nmfed[k],pyr_mask_d[k],non_na_original_ind_d[k]= normalize(val.values.T)
    pyr_mask_joint = np.all(np.stack(list(pyr_mask_d.values()),axis=0),axis=0) 
    non_na_original_ind_joint = np.nonzero(pyr_mask_joint)[0]
    W_d, W_sorted_d,W_inds_d, factor_assignment_d, H_sorted_d, X_sorted_d, X_recon_sorted_d={},{},{},{},{},{},{}

    for k, val in X_to_be_nmfed.items():
        W_d[k], W_sorted_d[k],W_inds_d[k], factor_assignment_d[k], H_sorted_d[k], X_sorted_d[k], X_recon_sorted_d[k] = nmf_and_sort(val[pyr_mask_joint],n_compo)

    return W_d, W_sorted_d,W_inds_d, factor_assignment_d, H_sorted_d, X_sorted_d, X_recon_sorted_d, pyr_mask_joint,non_na_original_ind_joint

def nmf_get_error(X,n_compo):
    '''
    do nmf get error, helper function
    '''
    model = NMF(n_components=n_compo,max_iter=1600)
    model.fit(X)
    error = model.reconstruction_err_
    error_ratio = error / np.linalg.norm(X,ord='fro')
    return error_ratio


def sweep_nmf_get_error(fr_to_be_nmfed_one_trialtype,sweep_l=None,model='nmf',do_normalize=True,prep_for_concat_bin=False,max_n_compo=20):
    if prep_for_concat_bin:
        X_df = preprocess_for_nmf_with_position(fr_to_be_nmfed_one_trialtype,do_normalize=do_normalize)
    else:
        X_df = fr_to_be_nmfed_one_trialtype
        if do_normalize:
            X_df = X_df / X_df.max(axis=1).values[:,None]
    X = X_df.values
    ntrials=X.shape[1]
    if sweep_l is None:
        sweep_l = np.arange(1,np.minimum(ntrials,max_n_compo)+1)
    error_ratio_l = []
    if model =='nmf':
        for k in tqdm.tqdm(sweep_l):
            model = NMF(n_components=k,max_iter=2000)
            model.fit(X)
            error = model.reconstruction_err_
            error_ratio = error / np.linalg.norm(X,ord='fro')
            error_ratio_l.append(error_ratio)
    elif model=='pca':
        model = PCA(n_components=int(np.max(sweep_l)))
        model.fit(X)
        for k in sweep_l:
            error_ratio = 1 - np.sum(model.explained_variance_ratio_[:k])
            error_ratio_l.append(error_ratio)
    else:
        print('model not implemented')        
        return
    error_ratio_l = pd.Series(error_ratio_l,index=sweep_l.astype(int))
    return error_ratio_l

def nmf_and_sort_h(X_df,k,model=None,correct_neg=True):
    '''
    newer [2023/9/30]: don't sort W; also return df if X_df is df
    '''
    if isinstance(X_df,pd.DataFrame):
        isdf=True
        X = X_df.values
        index=X_df.index
        cols = X_df.columns
    else:
        isdf=False
        X = X_df
    if correct_neg:
        X[X<0]=0.
    if model is None:
        model=NMF(n_components=k,max_iter=800)
    W = model.fit_transform(X)
    H = model.components_
    X_recon = W.dot(H)
    H_inds,H_sorted = sort_by_peak_within_factor(H)
    W = W[:,H_inds]
    
    if isdf:
        W = pd.DataFrame(W,index=index)
        H_sorted = pd.DataFrame(H_sorted,columns=cols)
    
    error = model.reconstruction_err_
    error_ratio = error / np.linalg.norm(X,ord='fro')
    
    return W,H_sorted,X_recon, error_ratio
    
    
    
    

def nmf_and_sort(X, k, model=None):
    '''
    X: n_neurons x n_trials
    '''
    if model is None:
        model=NMF(n_components=k,max_iter=800)
    W = model.fit_transform(X)
    H = model.components_
    X_recon = W.dot(H)
    H_inds,H_sorted = sort_by_peak_within_factor(H)
    W = W[:,H_inds] #reorder factors
    W_inds,W_sorted,factor_assignment=sort_factors(W)
    X_sorted=X[W_inds]
    X_recon_sorted=X_recon[W_inds]

    return W, W_sorted,W_inds, factor_assignment, H_sorted, X_sorted, X_recon_sorted

def masked_nmf(X, k, train_mask,max_iters=10,tol=1e-4):
    '''
    '''
    Z = copy.copy(X)
    scale = np.std(Z)
    test_mask = np.logical_not(train_mask)
    nmasked_out = np.sum(test_mask.astype(int))
    Z[test_mask] = np.random.random(nmasked_out) * scale
    model = NMF(n_components=k,max_iter=4000)

    converged = False
    train_err_l = []
    for i in range(max_iters):
        W = model.fit_transform(Z)
        H = model.components_
        recon = W.dot(H)
        Z[test_mask] = recon[test_mask]
        err = np.linalg.norm(recon[train_mask] - Z[train_mask])
        train_err_l.append(err)
        
        if i>=1:
            improvement = (train_err_l[i-1]-train_err_l[i]) / train_err_l[i-1]
            if improvement < tol:
                converged=True
                break
    
    test_err = np.linalg.norm(recon[test_mask] - X[test_mask])
    return test_err, W, H, Z, recon, i, converged, train_err_l

def preprocess_for_nmf_with_position(fr_to_be_nmfed_one_trialtype,do_normalize=True):
    trial_position_index = fr_to_be_nmfed_one_trialtype.index
    unit_names = fr_to_be_nmfed_one_trialtype.columns
    if do_normalize:
        X_normed,pyr_mask,non_na_original_ind=normalize(fr_to_be_nmfed_one_trialtype.values.T)
    else:
        X_normed = fr_to_be_nmfed_one_trialtype.values.T
        pyr_mask = np.ones(len(unit_names),dtype=bool)
    
    # X_normed_restacked_df = pd.DataFrame(X_normed,columns=trial_position_index).stack() # df: index:(units, position); columns: trials
    unit_names_left = unit_names[pyr_mask]
    X_normed_restacked_df = pd.DataFrame(X_normed[pyr_mask],index=unit_names_left,columns=trial_position_index).stack() # df: index:(units, position); columns: trials

    X_normed_restacked_df = X_normed_restacked_df.dropna(axis=1) # drop the column/trial that contains na, probably due to incomplete trajectory; temporary solution need to discuss
    return X_normed_restacked_df

def nmf_sort_with_position(fr_to_be_nmfed_one_trialtype,n_compo, model=None,do_normalize=True):
    '''
    fr_to_be_nmfed_one_trialtype: df, index: (trial, lin_binned); columns: units

    '''
    # trial_position_index = fr_to_be_nmfed_one_trialtype.index
    # unit_names = fr_to_be_nmfed_one_trialtype.columns
    # if do_normalize:
    #     X_normed,pyr_mask,non_na_original_ind=normalize(fr_to_be_nmfed_one_trialtype.values.T)
    # else:
    #     X_normed = fr_to_be_nmfed_one_trialtype.values
    #     pyr_mask = np.ones(len(unit_names),dtype=bool)
    # # import pdb
    # # pdb.set_trace()
    # # X_normed_restacked_df = pd.DataFrame(X_normed,columns=trial_position_index).stack() # df: index:(units, position); columns: trials
    # unit_names_left = unit_names[pyr_mask]
    # X_normed_restacked_df = pd.DataFrame(X_normed[pyr_mask],index=unit_names_left,columns=trial_position_index).stack() # df: index:(units, position); columns: trials

    # X_normed_restacked_df = X_normed_restacked_df.dropna(axis=1) # drop the column/trial that contains na, probably due to incomplete trajectory; temporary solution need to discuss
    
    X_normed_restacked_df = preprocess_for_nmf_with_position(fr_to_be_nmfed_one_trialtype,do_normalize=do_normalize)

    neuron_position_index=X_normed_restacked_df.index
    trials_left = X_normed_restacked_df.columns
    X_normed_restacked = X_normed_restacked_df.values # (nneuron x npos) x ntrials
    W, W_sorted,W_inds, factor_assignment, H_sorted, X_sorted, X_recon_sorted = nmf_and_sort(X_normed_restacked, n_compo, model=None)
    W_df = pd.DataFrame(W,neuron_position_index)
    H_sorted = pd.DataFrame(H_sorted, columns=trials_left)

    return W_df, W_sorted,W_inds, factor_assignment, H_sorted, X_sorted, X_recon_sorted, X_normed_restacked_df
    

#=========================post process=======================#
def add_metrics_to_W(W, n_compo, pd_kwargs={}):
    if not isinstance(W, pd.DataFrame):
        W_df = pd.DataFrame(W,**pd_kwargs)
    else:
        W_df = W
    W_df['sum'] = W_df.loc[:,0:n_compo-1].sum(axis=1)
    W_df['entropy'] = scipy.stats.entropy(W_df.loc[:,0:n_compo-1],axis=1)
    W_df['skew'] = scipy.stats.skew(W_df.loc[:,0:n_compo-1],axis=1)
    for i in range(0,n_compo):
        W_df[f'skew_{i}'] = W_df.apply(lambda x:skew_one(x.loc[0:n_compo-1],i),axis=1)
    return W_df

def get_peaks_in_W(W_df,n_compo,columns=None):
    '''
    each neuron, get the max among ws at each pos bin, use the peaks as a proxy for fields; 
    
    '''
    if n_compo is not None:
        w_max_across_trial = W_df.loc[:,0:n_compo-1].max(axis=1).unstack()
    else:
        w_max_across_trial = W_df.max(axis=1).unstack()

    peaks_l = []
    peaks_l_all = []
    heights_l = []
    # for n in range(w_max_across_trial.shape[0]):
    for n in w_max_across_trial.index.get_level_values(0):
        # xx=w_max_across_trial.iloc[n]
        xx=w_max_across_trial.loc[n]
        peaks = scipy.signal.find_peaks(xx)[0]
        peaks_l.append(peaks)
        peaks_l_all.extend((n,p) for p in peaks)
    

    W_df_peaks_only = W_df.loc[peaks_l_all]
    return W_df_peaks_only
    
def get_peaks_in_fr(fr_to_be_nmfed_one_trialtype,max_fr_thresh=1.):
    fr_peak_only=get_peaks_in_W(fr_to_be_nmfed_one_trialtype.unstack().T,None)

    # filter out weak peaks
    max_fr_thresh = 1
    fr_peak_only = fr_peak_only.loc[fr_peak_only.max(axis=1) > max_fr_thresh]
    return fr_peak_only


#=====================sorting============================#

def sort_factors(W):
    factor_assignment=np.argmax(W,axis=1)
    nfactors= W.shape[1]
    inds_within_all_l=[]
    for f in range(nfactors):
        mask = factor_assignment==f
        inds = np.nonzero(mask)[0]
        inds_within_all=inds[np.argsort(W[mask][:,f])]
        inds_within_all_l.append(inds_within_all)
    inds_within_all_l = np.concatenate(inds_within_all_l)
    return inds_within_all_l, W[inds_within_all_l],factor_assignment[inds_within_all_l]

def sort_by_peak_within_factor(H):
    '''
    H: nfactors x ntrials
    '''
    # inds = np.argsort(H.argmax(axis=1))
    inds = np.argsort(get_com(H,axis=1))
    
    return inds, H[inds]

def hierarchical_sort(df,to_cut_keys,to_cut_nbins,final_sort_key,**kwargs):
    '''
    for key in to_cut_keys, use pd.cut to divide into nbins in to_cut_nbins. 
    Iteratively sort the bins, bins within bins, finally the final_sort_key within the finest subbin
    '''
    df_c = copy.copy(df)    
    to_cut_binned_keys = []
    for k,nbins in zip(to_cut_keys,to_cut_nbins):
        name = f'{k}_binned'
        to_cut_binned_keys.append(name)
        df_c[name] = pd.cut(df_c[k],nbins,retbins=False,labels=False)
    df_c=df_c.sort_values(by=[*to_cut_binned_keys,final_sort_key],**kwargs)
    return df_c


def get_order_using_cross_correlation(X,truncate=500):
    # sorting by cross correlation peak; does not seem to work that well; but worth investigating why
    '''
    X: n_factors/neurons x n_temporal (could be trials, time etc)
    '''
    if truncate is None:
        truncate = X.shape[1]+1

    n_compo = X.shape[0]
    order_mat = np.zeros((n_compo,n_compo))
    for i in range(n_compo):
        for j in range(i+1,n_compo):
            c = ss.correlate(X[i,:truncate],X[j,:truncate])
            order_mat[i,j] = np.argmax(c) > truncate # positive peak, i later than j
    order_mat = order_mat + np.tril(1-order_mat.T) - np.eye(n_compo)
    # np.triu()
    order = np.argsort(order_mat.sum(axis=1))
    X_cc = X[order]
    return order, X_cc


#===========clustering====================#

def get_clusters_from_factor_assignment(W_inds,factor_assignment):
    '''
    inputs both from nmf_and_sort
    out: list, each is a list containing the indices of neurons assigned to that factor; indices are within W, i.e. some 0 firing neurons might have been filtered out already
    '''
    sample_inds_within_W_sorted_l = []
    n_compo = len(np.unique(factor_assignment))
    for fac in range(n_compo):
        fac_mask = factor_assignment==fac
        sample_inds_within_W_sorted = W_inds[fac_mask][:]
        sample_inds_within_W_sorted_l.append(sample_inds_within_W_sorted)

    return sample_inds_within_W_sorted_l

#=========n_component selection============#
from scipy.spatial.distance import pdist, squareform
def refit_get_similarity(X,n_compo_l,nrefits=10):
    '''
    for each n_compo in n_compo_l
    fit nmf multiple times 
    compare the similarity of h against the h with the lowest err
    '''
    similarity_l = {}
    for n_compo in n_compo_l:
        h_sorted_l = []
        err_l = []
        for _ in tqdm.tqdm(range(nrefits)):
            model = NMF(n_compo, init='random',max_iter=800)
            model.fit(X)
            err = model.reconstruction_err_
            err_l.append(err)
            h = model.components_  
            h_inds,h_sorted=sort_by_peak_within_factor(h)
            h_sorted=h_sorted.reshape(1,-1)
            h_sorted_l.append(h_sorted)
        min_err_index = np.argmin(err_l)
        h_sorted_l = np.concatenate(h_sorted_l,axis=0)
        # corr = np.corrcoef(h_sorted_l)
        dist=squareform(pdist(h_sorted_l)) / np.linalg.norm(h_sorted_l,axis=1)[:,None]
        corr=dist
        mask = np.ones(nrefits,dtype=bool)
        mask[min_err_index]=0
        corr_selected=corr[min_err_index,mask]
        # mean_corr = np.mean(corr[np.triu_indices(nrefits,k=1)])
        mean_corr = np.mean(corr_selected)
        similarity_l[n_compo] = mean_corr
    similarity_l=pd.Series(similarity_l)
    return similarity_l





#============================utils=============================#

def get_com(mat,axis=1):
    nrows,ncols=mat.shape
    
    inds = np.arange(mat.shape[axis])
    if axis==1:
        inds = inds[None,:]
    else:
        inds = inds[:,None]
    # pdb.set_trace()
       
    return np.sum(inds * mat,axis=axis) / np.sum(mat,axis=axis)
    

def skew_one(x,i):
    x=np.array(x)
    return (x[i] - x.mean())**3/x.std()**3