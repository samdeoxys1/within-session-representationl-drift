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

def preprocess_pca(X,do_sqrt=False,do_smooth=False,axis=0,smooth_kws={},do_center=False,do_normalize=False,normalize_kws={}):
    '''
    X: time x n_neurons 
    axis: the time axis, should be consistent with X, 0 by default
    smooth_kws_ = {'sigma':1,'axis':axis}
    normalize_kws_ = {'type':'max','percent_max':0.5} # or {'type':'zscore','range':None}
    '''
    is_df = False
    if isinstance(X,pd.DataFrame):
        is_df = True
        col = X.columns
        ind = X.index
        X = X.values
    if do_sqrt:
        X = np.sqrt(X.astype(float))
    if do_smooth:
        smooth_kws_ = {'sigma':1,'axis':axis}
        smooth_kws_.update(smooth_kws)
        X = gaussian_filter1d(X,**smooth_kws_)
    if do_center:
        X = X - np.mean(X,axis=axis,keepdims=True)
    if do_normalize:
        normalize_kws_ = {'type':'max','percent_max':0.5} # or {'type':'zscore','range':None}
        normalize_kws_.update(normalize_kws)
        if normalize_kws_['type']=='range':
            X_range = X.max(axis=axis) - X.min(axis=axis)
            max_range = X_range.max()
            X_range_ratio = (X - X.min(axis=axis,keepdims=True)) / np.maximum(X_range, max_range*normalize_kws['percent_max'])[:,None]
            X = X_range_ratio * (normalize_kws_['range'][1] - normalize_kws_['range'][0]) + normalize_kws_['range'][0]
        elif normalize_kws_['type']=='max':
            X_max = np.max(X,axis=axis,keepdims=True)
            for_comparison = np.max(X_max) * normalize_kws_['percent_max']
            X = X / np.maximum(X_max, for_comparison)
    if is_df:
        X = pd.DataFrame(X,index=ind,columns=col)
    return X

def do_pca(X,var_thresh=0.9,n_comp_default=None):
    '''
    X: n_neurons x n_time
    '''
    if n_comp_default is None:
        n_comp_default = X.shape[1]
    pca = PCA(n_components = n_comp_default)
    X_reduced = pca.fit_transform(X)
    evr_cumsum = np.cumsum(pca.explained_variance_ratio_)
    thresh_cross = np.nonzero(np.isclose(evr_cumsum, var_thresh))[0] # using isclose instead of >= to make it compatible for when var_thresh=1
    if len(thresh_cross) > 0:
        
        n_comp_sufficient = np.nonzero(np.isclose(evr_cumsum , var_thresh))[0][0] + 1
    else:
        n_comp_sufficient = n_comp_default
    X_reduced = X_reduced[:,:n_comp_sufficient]
    W = pca.components_[:n_comp_sufficient]

    pc_df = pd.DataFrame(W.T)
    pc_df.index = X.columns
    
    if isinstance(X,pd.DataFrame):
        X_reduced = pd.DataFrame(X_reduced,index=X.index)
    return X_reduced,pc_df,pca,n_comp_sufficient

######### vector bundle pca
def preprocess_fr_map_trial(fr_map_trial,cell_cols,max_all_comparison_factor=0.3):
    '''
    # fr_map: n_neurons x n_pos
    fr_map_trial: n_neurons x n_pos x n_trials
    '''
    # type 1: center both across position and across trials, max norm, fr, then make each column norm 1
    fr_map = fr_map_trial.mean(axis=-1)
    
    # double centering: don't understand why subtract the mean axis=(1,2) once won't work! precesion?
    fr_map_trial_center = fr_map_trial.mean(axis=1,keepdims=True)
    fr_map_trial_centered = fr_map_trial - fr_map_trial_center
    fr_map_trial_centered = fr_map_trial_centered - fr_map_trial_centered.mean(axis=2,keepdims=True)

    max_all = fr_map_trial_centered.max()
    max_each_neuron = np.abs(fr_map_trial_centered).max(axis=(1,2),keepdims=True)
    fr_map_trial_centered_normed = fr_map_trial_centered / np.maximum(max_each_neuron,max_all * max_all_comparison_factor)
    fr_map_trial_centered_normed_df = pf.fr_map_trial_to_df(fr_map_trial_centered_normed,cell_cols)

    return fr_map_trial_centered_normed_df,fr_map_trial_centered_normed

def vector_bundle_pca(fr_map_trial_df_prep,n_comp_default=None):
    '''
    around a mean, for each smoothly varying condition (i.e. position here), get the pcs, then link the pcs
    '''

    P = fr_map_trial_df_prep.index.get_level_values(1).nunique()
    if n_comp_default is None:
        L=fr_map_trial_df_prep.shape[1]
        n_comp_default = L
    n_comp_sufficient_l = []
    exp_var_ratio_l = []
    pc_df_l = {}
    X_reduced_l = {}
    for p in range(P):
        X = fr_map_trial_df_prep.loc[(slice(None),p),:].droplevel(1).T # ntrials x nneurons
        X_reduced,pc_df,pca,n_comp_sufficient = do_pca(X,var_thresh=1.,n_comp_default=n_comp_default) # X_reduced: n_trial x n_compo

        exp_var_ratio_l.append(pca.explained_variance_ratio_)
        pc_df_l[p]=pc_df
        n_comp_sufficient_l.append(n_comp_sufficient)
        X_reduced_l[p]=X_reduced
    n_comp_sufficient_l = np.array(n_comp_sufficient_l)
    n_compo_sufficient_min = np.min(n_comp_sufficient_l)
    for p in range(P):
        pc_df_l[p]  = pc_df_l[p].iloc[:,:n_compo_sufficient_min]
        exp_var_ratio_l[p] = exp_var_ratio_l[p][:n_compo_sufficient_min]
        X_reduced_l[p] = X_reduced_l[p].iloc[:,:n_compo_sufficient_min]
    pc_df_l = pd.concat(pc_df_l)
    exp_var_ratio_l = pd.DataFrame(exp_var_ratio_l)
    X_reduced_l = pd.concat(X_reduced_l) # (npos x ntrial) x ncompo
    return pc_df_l,X_reduced_l, exp_var_ratio_l
    
def link_pcs_across_pos(pc_df_l):
    '''
    pc_df_l: (npos x nneurons) x nposbins
    '''
    pass

def solve_w_one_pos(p,X,h,w_l,lam,w_l_prev=None,not_masked=True):
    '''
    X: n_neurons x n_trials
    given h, w^{p-1},w^{p+1}, compute the estimate of w^{p} 
    using:
    w^p=(Xh^T+\lambda w^{(p-1)}+\lambda w^{(p+1)})(hh^T+2\lambda)^{-1}

    smoothing: quadratic variation
    position assumed to be on a ring, 0 and end adjacent
    also assuming all position are present, no missing ones!!!! [crucial, since some ratemap computation might lead to missing positions]
    w_l: n_pos x n_neuron x rank
    '''
    n_p = len(w_l)
    p_next = (p + 1) % n_p
    p_prev = (p - 1) % n_p
    if not_masked:
        signal = X.dot(h.T)
    else:
        signal = 0.
    if w_l_prev is None:
        wp = (signal + lam * (w_l[p_prev]+w_l[p_next]) ) / (h.dot(h.T)+2* lam)
    else:
        smth_target = w_l[p_prev]+w_l[p_next] 
        smth_projection = smth_target.T.dot(w_l_prev[p]) * w_l_prev[p]
        smth_target_ortho = smth_target - smth_projection
        wp = (signal + lam * (smth_target_ortho) ) / (h.dot(h.T)+2* lam)
    return wp

def solve_w_all_pos(X_l,h_l,w_l,lam,w_l_prev=None,pos_mask=None):
    '''
    # fr_map_trial_centered_normed_df: (n_neuron x n_pos) x n_trials
    X_l: n_pos x n_neurons x n_trials
    h_l: n_pos x 1 x n_trials
    w_l: n_pos x n_neurons x 1

    w_l_prev: n_pos x n_neurons x 1; solution from the previous rank 1 decomposition
    '''
    n_p = w_l.shape[0]
    w_l_hat = []
    if pos_mask is None:
        pos_mask = np.ones(n_p,dtype=bool)
    for p in range(n_p):
        # X = fr_map_trial_centered_normed_df.loc[(slice(None),p),:].values
        X = X_l[p]
        h = h_l[p]
        wp = solve_w_one_pos(p,X,h,w_l,lam,w_l_prev=w_l_prev,not_masked=pos_mask[p])
        w_l_hat.append(wp)
    w_l_hat = np.array(w_l_hat)
    w_l_hat = w_l_hat / np.linalg.norm(w_l_hat,axis=1,keepdims=True)
    return w_l_hat

def wrap_w_all_pos_rank1(w_l_hat,pc_df_l):
    '''
    turn w_l_hat: n_pos x n_neurons x 1
    into series: (n_pos x n_neurons), labeled by pc_df_l given by vector_bundle_pca
    '''
    w_l_df=pd.DataFrame(w_l_hat[:,:,0]).stack()
    w_l_df.index = pc_df_l.index
    return w_l_df

def prep_X_l(fr_map_trial_centered_normed):
    '''
    fr_map_trial_centered_normed: n_neurons x n_pos x n_trials

    X_l: n_pos x n_neurons x n_trials
    '''
    X_l = fr_map_trial_centered_normed.swapaxes(0,1)
    return X_l

def solve_h_all_pos(X_l,w_l):
    '''
    X_l: n_pos x n_neurons x n_trials
    w_l: n_pos x n_neurons x rank
    h_l: n_pos x rank x n_trials
    '''
    h_l = np.einsum('pnl,pnr->prl',X_l,w_l)
    return h_l

def get_resid(X_l,w_l,h_l):
    resid = X_l - np.einsum('pnr,prl->pnl',w_l,h_l)
    return resid

def get_var_explained_ratio(X_l,w_l,h_l,pos_mask=None):
    '''
    
    r2_l: n_pos,
    '''
    if pos_mask is None:
        pos_mask = np.ones(X_l.shape[0],dtype=bool)
    X_l = X_l[pos_mask]
    w_l = w_l[pos_mask]
    h_l = h_l[pos_mask]

    resid = get_resid(X_l,w_l,h_l)
    r2_l = 1 - np.linalg.norm(resid,axis=(1,2))**2 / np.linalg.norm(X_l,axis=(1,2))**2
    
    r2_total =  1 - np.linalg.norm(resid)**2 / np.linalg.norm(X_l)**2
    return r2_total, r2_l


def loss_recon(X_l,w_l,h_l,pos_mask=None):
    resid = get_resid(X_l,w_l,h_l)
    if pos_mask is None:
        pos_mask = np.ones(X_l.shape[0],dtype=bool)
    l = np.linalg.norm(resid[pos_mask]) **2
    return l

def loss_regularization(w_l,lam):
    n_p = w_l.shape[0]
    reg_l = []
    for p in range(n_p):
        p_next = (p + 1) % (n_p-1)
        p_prev = (p - 1) % (n_p-1)
        rr=lam * (np.linalg.norm(w_l[p] - w_l[p_prev])**2 +np.linalg.norm(w_l[p] - w_l[p_next])**2 )
        reg_l.append(rr)
    reg_l = np.array(reg_l)
    reg_l_sum = reg_l.sum()
    return reg_l_sum,reg_l

def loss(X_l,w_l,h_l,lam,pos_mask=None):
    l = loss_recon(X_l,w_l,h_l,pos_mask=pos_mask)
    lr,_ = loss_regularization(w_l,lam)
    lall = l+lr
    return lall, l, lr


def get_pos_mask(npos,ratio_consec_bins_to_mask=0.1): 
    '''
    consecutive bins for each trial
    return:
    mask:  npos; 1 if included in train, 0 if included in test; flip it to get the test_mask
    '''
    mask = np.ones(npos,dtype=bool)
    n_consec_bins_to_mask = int(ratio_consec_bins_to_mask * npos)
    upper_bound = npos - 1 - n_consec_bins_to_mask # -1 because maxind = npos-1
    mask_start_ind = np.random.randint(0,upper_bound)
    mask_end_ind = mask_start_ind + n_consec_bins_to_mask
    mask[mask_start_ind:mask_end_ind] = 0

    return mask

def init_w_l_h_l(fr_map_trial_centered_normed_df,X_l,n_comp_default=None,pos_mask=None):
    
    
    pc_df_l,X_reduced_l,exp_var_ratio_l =vector_bundle_pca(fr_map_trial_centered_normed_df,n_comp_default=n_comp_default)
    h_l_allfac = np.concatenate([X_reduced_l.loc[:,eig_ind].unstack(level=1).values[:,None,:] for eig_ind in range(X_reduced_l.shape[1])],axis=1)
    w_l_allfac = np.concatenate([pc_df_l.loc[:,eig_ind].unstack(level=1).values[:,:,None] for eig_ind in range(X_reduced_l.shape[1])],axis=-1)
    if pos_mask is not None:
        n_pos_replacement = np.sum((~pos_mask).astype(int))
        w_replacement_l = []
        for p in range(n_pos_replacement):
            w_replacement = np.random.normal(size=(w_l_allfac.shape[1],w_l_allfac.shape[2]))
            w_replacement,_ = np.linalg.qr(w_replacement)
            w_replacement = w_replacement / np.linalg.norm(w_replacement,axis=0,keepdims=True)
            w_replacement_l.append(w_replacement)
        w_replacement_l = np.array(w_replacement_l)
        h_replacement_l = solve_h_all_pos(X_l[~pos_mask],w_replacement_l)
        w_l_allfac[~pos_mask] = w_replacement_l
        h_l_allfac[~pos_mask] = h_replacement_l
    return w_l_allfac, h_l_allfac

def train_one_factor(X_l,lam=20.,max_iters = 100, stop_thresh = 0.00001,
                    fr_map_trial_centered_normed_df=None,eig_ind=0,w_l_init=None,h_l_init=None,
                    w_l_prev=None,
                    pos_mask = None,return_init=False
                    ):
    '''
    X_l: n_pos x n_neuron x n_trial
    fr_map_trial_centered_normed_df: (n_neuron x n_pos) x n_trial

    stop_thresh: harsher for l_recon: want to encourage to stop at recon critical point, and not using the stop thresh; 
        more lenient for total loss, because we don't want to encourage overfitting on the smoothing penalty
    '''
    if w_l_init is None:
        # pc_df_l,X_reduced_l,exp_var_ratio_l =vector_bundle_pca(fr_map_trial_centered_normed_df)
        # h_l_init = X_reduced_l.loc[:,eig_ind].unstack(level=1).values[:,None,:]
        # w_l_init = pc_df_l.loc[:,eig_ind].unstack(level=1).values[:,:,None]

        w_l_allfac, h_l_allfac =init_w_l_h_l(fr_map_trial_centered_normed_df,X_l,n_component_default=1,pos_mask=pos_mask)
        h_l_init = h_l_allfac[:,[eig_ind],:]
        w_l_init = w_l_allfac[:,:,[eig_ind]]
    
    l_all_l,l_recon_l,l_reg_l=[],[],[]
    w_l = w_l_prev_iter = w_l_init
    h_l = h_l_prev_iter = h_l_init
    
    l_recon_prev_decrease = False
    l_recon_curr_increase = False
    l_all_decrease_below_thresh = False

    for i in range(max_iters):
        w_l_prev_iter = w_l
        h_l_prev_iter = h_l

        w_l_hat = solve_w_all_pos(X_l,h_l,w_l,lam,w_l_prev=w_l_prev,pos_mask=pos_mask)
        h_l_hat = solve_h_all_pos(X_l,w_l_hat)

        l_all,l_recon,l_reg =loss(X_l,w_l_hat,h_l_hat,lam,pos_mask=pos_mask)
        l_all_l.append(l_all)
        l_recon_l.append(l_recon)
        l_reg_l.append(l_reg)
        w_l = w_l_hat
        h_l = h_l_hat

        # stop criterion: (l_recon critical point) or (l_recon decrease below_thresh) & l_all decrease below thresh 
        if i > 2:
            l_recon_prev_decrease = (l_recon_l[i-1] - l_recon_l[i-2]) <= 0
            l_recon_curr_increase = (l_recon_l[i] - l_recon_l[i-1]) > 0
            l_all_decrease_below_thresh = (np.abs(l_all_l[i] -l_all_l[i-1]) / l_all_l[i-1] ) < stop_thresh
            l_recon_decrease_below_thresh = (l_recon_l[i]<=l_recon_l[i-1]) & ((np.abs(l_recon_l[i] -l_recon_l[i-1]) / l_recon_l[i-1] ) < (stop_thresh * 1e-2 ) ) # has to decrease
            
            if ((l_recon_prev_decrease & l_recon_curr_increase) | l_recon_decrease_below_thresh) & l_all_decrease_below_thresh:
                w_l = w_l_prev_iter
                h_l = h_l_prev_iter
                break

    
    # w_l_df = wrap_w_all_pos_rank1(w_l,pc_df_l)
    if return_init:
        return w_l, h_l, l_all_l, l_recon_l, l_reg_l, w_l_init, h_l_init
    else:
        return w_l, h_l, l_all_l, l_recon_l, l_reg_l

def train_all_factor(X_l,n_fac=3,lam=20.,max_iters = 100, stop_thresh = 0.00001,
                    fr_map_trial_centered_normed_df=None,w_l_allfac=None,h_l_allfac=None,
                    pos_mask = None,return_init=False):
    eig_ind = 0
    if w_l_allfac is None:
        w_l_allfac, h_l_allfac = init_w_l_h_l(fr_map_trial_centered_normed_df,X_l,pos_mask=pos_mask,n_comp_default=n_fac)
    w_l_allfac_fit = []
    h_l_allfac_fit = []
    l_all_allfac = []
    l_recon_allfac = []
    l_reg_allfac = []
    X_l_orig = copy.deepcopy(X_l)
    r2_allfac= []
    r2_pos_allfac = []
    for eig_ind in range(n_fac):
        if eig_ind > 0: # for the next problem, get residuals using the previous fit 
            X_l = get_resid(X_l,w_l,h_l)
            w_l_prev = w_l
        else:
            w_l_prev=None
        w_l, h_l, l_all_l, l_recon_l, l_reg_l = train_one_factor(X_l,lam=lam,max_iters = max_iters, stop_thresh = stop_thresh,
                    fr_map_trial_centered_normed_df=fr_map_trial_centered_normed_df,eig_ind=eig_ind,w_l_init=w_l_allfac[:,:,[eig_ind]], h_l_init=h_l_allfac[:,[eig_ind],:],
                    w_l_prev = w_l_prev,
                    pos_mask = pos_mask,return_init=False)
        r2,r2_pos = get_var_explained_ratio(X_l_orig,w_l,h_l,pos_mask=pos_mask)
        r2_allfac.append(r2)
        r2_pos_allfac.append(r2_pos)
        
        # pdb.set_trace()
        w_l_allfac_fit.append(w_l)
        h_l_allfac_fit.append(h_l)
        l_all_allfac.append(l_all_l)
        l_recon_allfac.append(l_recon_l)
        l_reg_allfac.append(l_reg_l)
    w_l_allfac_fit = np.concatenate(w_l_allfac_fit,axis=-1)
    h_l_allfac_fit = np.concatenate(h_l_allfac_fit,axis=1)
    # l_all_allfac = np.array(l_all_allfac)
    # l_recon_allfac = np.array(l_recon_allfac)
    # l_reg_allfac = np.array(l_reg_allfac)

    r2_allfac = np.array(r2_allfac)
    r2_pos_allfac = np.array(r2_pos_allfac) # nfac x npos

    if return_init:
        return w_l_allfac_fit, h_l_allfac_fit,l_all_allfac, l_recon_allfac, l_reg_allfac, w_l_allfac, h_l_allfac, r2_allfac, r2_pos_allfac 
    else:
        return w_l_allfac_fit, h_l_allfac_fit,l_all_allfac, l_recon_allfac, l_reg_allfac,r2_allfac, r2_pos_allfac 

        

def cv_one_factor(X_l,lam_l=[20.],n_cv=10,ratio_consec_bins_to_mask=0.1,
                    max_iters = 100, stop_thresh = 0.00001,
                    fr_map_trial_centered_normed_df=None,eig_ind=0,w_l_init=None,h_l_init=None,
    ):
    '''
    not sure if i want different factors to have different smoothing
    '''
    n_p = X_l.shape[0]
    cv_res = {}
    for m in tqdm.tqdm(range(n_cv)):
        cv_res[m] = {}
        pos_mask = get_pos_mask(n_p,ratio_consec_bins_to_mask=ratio_consec_bins_to_mask)
        for lam in lam_l:
            w_l, h_l, l_all_l, l_recon_l, l_reg_l = \
            train_one_factor(X_l,lam=lam,max_iters = max_iters,
                                fr_map_trial_centered_normed_df=fr_map_trial_centered_normed_df,eig_ind=eig_ind,w_l_init=None,h_l_init=None,
                                pos_mask = pos_mask
                                )
            r2_total_train, r2_l = get_var_explained_ratio(X_l,w_l,h_l,pos_mask=pos_mask)
            r2_total_test, r2_l = get_var_explained_ratio(X_l,w_l,h_l,pos_mask=~pos_mask)
            cv_res[m][lam]={'train':r2_total_train,'test':r2_total_test}
        cv_res[m] = pd.DataFrame(cv_res[m])
    cv_res = pd.concat(cv_res)
    # cv_res = pd.DataFrame(cv_res)
    return cv_res
    
def cv_all_factor(X_l,n_fac=3,lam_l=[20.],n_cv=10,ratio_consec_bins_to_mask=0.1,
                    max_iters = 100, stop_thresh = 0.00001,
                    fr_map_trial_centered_normed_df=None,w_l_allfac=None,h_l_allfac=None,):
    
    n_p = X_l.shape[0]
    cv_res = {}
    for m in tqdm.tqdm(range(n_cv)):
        cv_res[m] = {}
        pos_mask = get_pos_mask(n_p,ratio_consec_bins_to_mask=ratio_consec_bins_to_mask)
        for lam in lam_l:
            w_l_allfac_fit, h_l_allfac_fit,l_all_allfac, l_recon_allfac, l_reg_allfac = train_all_factor(X_l,n_fac=n_fac,lam=lam,max_iters = max_iters, stop_thresh = stop_thresh,
                    fr_map_trial_centered_normed_df=fr_map_trial_centered_normed_df,w_l_allfac=w_l_allfac,h_l_allfac=h_l_allfac,
                    pos_mask = pos_mask,return_init=False)
            r2_total_train, r2_l = get_var_explained_ratio(X_l,w_l_allfac_fit,h_l_allfac_fit,pos_mask=pos_mask)
            r2_total_test, r2_l = get_var_explained_ratio(X_l,w_l_allfac_fit,h_l_allfac_fit,pos_mask=~pos_mask)
            cv_res[m][lam]={'train':r2_total_train,'test':r2_total_test}
        cv_res[m] = pd.DataFrame(cv_res[m])
    cv_res = pd.concat(cv_res)
    # cv_res = pd.DataFrame(cv_res)
    return cv_res


def post_process_loadings(h_l_allfac_fit,trial_ind_l=None):
    '''

    trial_ind_l: n_trials; index within all trials
    '''
    pos_l = np.arange(h_l_allfac_fit.shape[0])
    pos_l=np.tile(pos_l[:,None,None],[1,1,h_l_allfac_fit.shape[-1]])
    h_l_pos = np.concatenate([pos_l,h_l_allfac_fit],axis=1)
    if trial_ind_l is None:
        trial_ind_l = np.arange(h_l_pos.shape[-1])
    h_l_allfac_fit_df = []
    for tt in range(h_l_pos.shape[-1]):
        npcs = h_l_pos.shape[1] - 1
        columns = ['pos'] + [f'pc{nn}' for nn in range(npcs)]
        h_one_trial_df = pd.DataFrame(h_l_pos[:,:,tt],columns=columns)
        
        h_one_trial_df['trial'] = trial_ind_l[tt]
        h_l_allfac_fit_df.append(h_one_trial_df)
    h_l_allfac_fit_df = pd.concat(h_l_allfac_fit_df,axis=0,ignore_index=True)
    return h_l_allfac_fit_df