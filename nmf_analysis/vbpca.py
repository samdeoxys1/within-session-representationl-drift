import numpy as np
import scipy
import matplotlib.pyplot as plt
import sys,os,copy,pdb,importlib
from importlib import reload
import pandas as pd
import matplotlib
from matplotlib.animation import FuncAnimation,ArtistAnimation
from scipy.ndimage import gaussian_filter1d
import sklearn
from sklearn.decomposition import PCA
sys.path.append('/mnt/home/szheng/projects/util_code/')
import place_field_analysis as pf
import tqdm
from nnls import nnlsm_activeset
import pca_analysis as pcaa
# import jax.numpy as jnp

'''
X_l: n_pos x n_neuron x n_trial
w_p_l: n_pos x n_neuron x n_fac
ww_l: n_pos x n_neuron x n_neuron
'''

def init(X_l,n_fac=3,lam=0.1,pos_mask=None):
#     pc_df_l,X_reduced_l,exp_var_ratio_l =vector_bundle_pca(fr_map_trial_centered_normed_df,n_comp_default=n_comp_default)
    ww_l = []
    w_l = []
    h_l = []
    for p, Xp in enumerate(X_l):
        pca = PCA(n_components=n_fac).fit(Xp.T) # Xp.T: n_trial x n_neuron
        hp = pca.transform(Xp.T).T # n_fac x n_trial 
        wp = pca.components_.T # n_neuron x n_fac

        w_l.append(wp)
        h_l.append(hp)        
        ww = wp.dot(wp.T) 
        ww_l.append(ww)
    ww_l = np.stack(ww_l,axis=0)
    w_l = np.stack(w_l,axis=0)
    h_l = np.stack(h_l,axis=0)
    
    r2_total, r2_l = pcaa.get_var_explained_ratio(X_l,w_l,h_l,pos_mask=pos_mask)
    lall,lrecon,lr = pcaa.loss(X_l,w_l,h_l,lam,pos_mask=pos_mask)
        
    return ww_l, r2_total, lall, lrecon, lr
        

def fit_subspace_one_pos(p,Xp,ww_l,n_fac=3,lam=0.1,pos_mask=None):
    '''
    get w_p w_p^T
    lam: smoothness penalty constant
    make sure Xp is centered across trials
    '''
    npos = ww_l.shape[0]
    p_prev,p_next = (p-1) % npos, (p+1) % npos
    
    data = Xp.dot(Xp.T)
    smooth =1/2 * (ww_l[p_prev] + ww_l[p_next])
    target = (data + lam * smooth) #/ (np.trace((Xp.T.dot(ww_l[p])).dot(Xp)) + lam) # not in equation, need to think
    
    u,s,vh = np.linalg.svd(target)
    ww = u[:,:n_fac].dot(u[:,:n_fac].T)
    return ww

def fit_subspace_all_pos(X_l,ww_l,n_fac=3,lam=0.1,pos_mask=None):
    npos = X_l.shape[0]
    for p in range(npos):
        wwp = fit_subspace_one_pos(p,X_l[p],ww_l,n_fac=n_fac,lam=lam,pos_mask=pos_mask)
        ww_l[p] = wwp
    return ww_l

def get_projection_distance(ww_l,X_l):
    X_l_proj = np.einsum('pmn,pnt->pmt',ww_l,X_l)
    dist =np.sum(np.linalg.norm(X_l - X_l_proj,ord='fro',axis=(1,2))**2)
    tot_var = np.sum(np.linalg.norm(X_l,ord='fro',axis=(1,2))**2)
    r2 = (tot_var - dist)/tot_var
    return dist,r2
    
def get_smth_loss(ww_l,lam=0.1):
    ww_l_next = np.concatenate([ww_l[1:],ww_l[[0]]],axis=0)
    l_smth = np.sum(np.linalg.norm((ww_l_next - ww_l),axis=(1,2),ord='fro')**2)
    return l_smth
    
def get_loss_all(ww_l,X_l,lam=0.1):
    
    l_recon,r2 = get_projection_distance(ww_l,X_l)
    l_smth = get_smth_loss(ww_l,lam=0.1)
    l_tot = l_recon + l_smth
    return l_tot, r2, l_recon, l_smth

import tqdm
def fit_vbpca(X_l,lam=0.1,n_fac=3,n_iters=10,pos_mask=None):
    
    ww_l_init, r2_init, l_tot_init, lrecon_init, l_smth_init =init(X_l,n_fac=3,lam=lam,pos_mask=None)
    
    ww_l = ww_l_init
    l_tot_l = [l_tot_init]
    l_recon_l = [lrecon_init]
    l_smth_l = [l_smth_init]
    r2_l = [r2_init]
    
    for _ in tqdm.trange(n_iters):
        ww_l = fit_subspace_all_pos(X_l,ww_l,n_fac=n_fac,lam=lam,pos_mask=pos_mask)
        l_tot,r2,l_recon,l_smth = get_loss_all(ww_l,X_l,lam=lam)
        l_tot_l.append(l_tot)
        l_recon_l.append(l_recon)
        l_smth_l.append(l_smth)
        r2_l.append(r2)
    
        # stopping
    
    training_metrics = {'loss':l_tot_l,'loss_reconstruct':l_recon,'loss_smooth':l_smth}
    training_metrics = pd.DataFrame(training_metrics)
    
    
    
    return ww_l, training_metrics

# local alignment, with sign alignment
def get_w_from_ww_local(ww_l,svd_win = 0, n_fac = 1):
    '''
    # local alignment, with sign alignment
    compare prev and next innter product, if negative then flip the sign of next;
    need to generalize in the case of higher d base manifold
    '''
    svd_win = 0
    n_fac = 1
    n_pos = ww_l.shape[0]
    w_l = []
    for p in range(n_pos):
        inds=np.arange((p-svd_win),p + svd_win+1)
        inds = inds%n_pos
        ww_l_sub_concat = np.concatenate(ww_l[inds],axis=1)
        trunsvd = sklearn.decomposition.TruncatedSVD(n_components=n_fac)
        trunsvd.fit(ww_l_sub_concat)
        w = trunsvd.transform(ww_l_sub_concat)
        w,_ = np.linalg.qr(w)
        if p > 0:
            w_prev = w_l[(p-1)%n_pos]
            w = align_two_w(w_prev,w)
        w_l.append(w)

    w_l = np.array(w_l)
    return w_l





# sign alignment
def align_two_w(w_prev,w_next):
    inner_prod=np.einsum('nf,nf->f',w_prev,w_next)
    w_next[:,inner_prod<0] = w_next[:,inner_prod<0] * -1
    return w_next
    

# get w through global realignment, doesn't seem to work well
def get_w_from_ww_global(ww_l,n_fac=3):
    n_pos = ww_l.shape[0]
    ww_l_concat = np.concatenate(list(ww_l),axis=1)
    trun_svd = sklearn.decomposition.TruncatedSVD(n_components=n_fac*n_pos).fit(ww_l_concat)
    w_l = trun_svd.components_[:n_fac].reshape(n_fac,n_pos,-1).swapaxes(0,1).swapaxes(1,2)
    w_ortho_l=[]
    for w in w_l:
        w_,_=np.linalg.qr(w)
        w_ortho_l.append(w_)
    w_ortho_l = np.array(w_ortho_l)
    return w_ortho_l
    

        
import scipy.spatial
from scipy.spatial.distance import squareform, pdist
def grassmannian_dist(ww1_reshaped,ww2_reshaped):
    ww1 = ww1_reshaped.reshape(int(np.sqrt(ww1_reshaped.shape[0])),-1)
    ww2 = ww2_reshaped.reshape(int(np.sqrt(ww2_reshaped.shape[0])),-1)
    angles=scipy.linalg.subspace_angles(ww1,ww2)
    dist = np.sqrt(np.sum(angles**2))
    return dist
    













































# using the idea of fiber alignment
# def preprocess(X,do_sqrt=False,do_smooth=False,center_axis=0,residual_axis=-1,smooth_kws={},do_normalize=False,normalize_kws={},
#     resid_lag = 1,
#     ):
#     '''
#     get: possibly: centered across positions, then both: centered across trials, and residuals with some lags
    
#     X: pos x neuron x trial
#     center_axis: usually position
#     residual_axis: usually trial; the axis to be kept as data samples
#     resid_lag: the window (+ and -) within which the residuals will be used; needed for alignment

#     smooth_kws_ = {'sigma':1,'axis':axis}
#     normalize_kws_ = {'type':'max','percent_max':0.5} # or {'type':'zscore','range':None}
#     '''
#     is_df = False
#     if isinstance(X,pd.DataFrame):
#         is_df = True
#         col = X.columns
#         ind = X.index
#         X = X.values
#     if do_sqrt:
#         X = np.sqrt(X.astype(float))
#     if do_smooth:
#         smooth_kws_ = {'sigma':1,'axis':0}
#         smooth_kws_.update(smooth_kws)
#         X = gaussian_filter1d(X,**smooth_kws_)
#     if center_axis is not None:
#         X = X - np.mean(X,axis=center_axis,keepdims=True)
#     X_mean = np.mean(X,axis=residual_axis,keepdims=True)
#     n_p = X.shape[center_axis]
#     n_resid_one_p = 2 * resid_lag + 1 # symmetric
#     X_resid = np.zeros((n_resid_one_p,*X.shape))
#     # get all unnormalized residuals
#     for p in range(n_p):
#         for i in range(n_resid_one_p):
#             q = (i + (-resid_lag) + p)%n_p # the position relative to whose average the subtraction is done
#             X_resid[i,p,...] = X[p] - X_mean[q] # pth position's population vector at each trial relative to the mean at p+q th position
#     # pdb.set_trace()
#     zero_lag_ind = resid_lag
#     if do_normalize: # normalization constant computed only using residuals relative to the same position, not lagged ones
#         normalize_kws_ = {'type':'max','percent_max':0.5} # or {'type':'zscore','range':None}
#         normalize_kws_.update(normalize_kws)
#         if normalize_kws_['type']=='range':
#             # X_range = X.max(axis=axis) - X.min(axis=axis)
#             X_resid_range = X_resid[[zero_lag_ind]].max(axis=residual_axis) -X_resid[[zero_lag_ind]].min(axis=residual_axis) 
#             max_range = X_resid_range.max()
#             X_resid_min = X_resid[[zero_lag_ind]].min(axis=residual_axis,keepdims=True)
#             X_range_ratio = (X_resid - X_resid_min) / np.maximum(X_resid_range, max_range*normalize_kws['percent_max'])
#             X_resid = X_range_ratio * (normalize_kws_['range'][1] - normalize_kws_['range'][0]) + normalize_kws_['range'][0]
        
#         elif normalize_kws_['type']=='max': 
#             X_max = np.max(np.abs(X_resid[[zero_lag_ind]]),axis=residual_axis,keepdims=True)
#             for_comparison = np.max(X_max) * normalize_kws_['percent_max']
#             X_resid = X_resid / np.maximum(X_max, for_comparison)
#     if is_df:
#         X_resid = pd.DataFrame(X_resid,index=ind,columns=col)
#     return X_resid,X_mean

# def solve_w_one_pos(p,X_resid,h,w_l,lam,alpha,w_l_prev=None,not_masked=True):
#     '''
#     rank 1 update

#     X_resid: n_lag x n_pos x n_neuron x n_trial
#     lam: smoothing penalty
#     alpha: alignment penalty
#     '''
#     n_p = len(w_l)
#     resid_lag = int(np.floor(X_resid.shape[0] / 2))
#     # smooth currently only two steps; can incorporate arbitrary kernel in the future
#     p_next = (p + 1) % n_p
#     p_prev = (p - 1) % n_p
#     smth_target = w_l[p_prev]+w_l[p_next] 

#     signal = X_resid[resid_lag].dot(h.T) * not_masked # resid_lag correspond to the index of the residual at p wrt mean at p
#     if w_l_prev is None:
#         wp = (signal + lam * (w_l[p_prev]+w_l[p_next]) ) / (h.dot(h.T)+2* lam)
#     else:
#         smth_target = w_l[p_prev]+w_l[p_next] 
#         smth_projection = smth_target.T.dot(w_l_prev[p]) * w_l_prev[p]
#         smth_target_ortho = smth_target - smth_projection
#         wp = (signal + lam * (smth_target_ortho) ) / (h.dot(h.T)+2* lam)




