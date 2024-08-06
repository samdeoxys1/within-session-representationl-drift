import numpy as np
import scipy
import matplotlib.pyplot as plt
import sys,os,copy,pdb,importlib
from importlib import reload
import pandas as pd

from scipy.interpolate import BSpline
from cvxopt import matrix, solvers
solvers.options['show_progress']=False
import plot_helper as ph

def get_basis(n_basis,n_pt=40,degree=3,do_plot=True,ax=None):
    '''
    basis: n_pt x n_basis 
    '''
    k=degree
    n=n_basis
    n_middle = n+k+1 - 2*(k)
    xs = np.linspace(0,1,n_pt)
    t = np.concatenate([np.zeros(k),np.linspace(0,1,n_middle),np.ones(k)])

    
    basis = []
    
    for i in range(n):
        c=np.zeros(n)
        c[i] = 1
        spline=BSpline(t,c,k)
        y=spline(xs)
        basis.append(y)
    basis = np.array(basis).T
    if do_plot:
        if ax is None:
            fig,ax=plt.subplots()        
        ax.plot(xs,basis)
    
    return basis

def get_mode_constraint_mat(n_basis,mode_index,k,reshape=True):
    '''
    negative for qp
    k is N factor, columns of beta, rows of H
    '''
#     mat =np.eye(n_basis-1)
    eps = 0.00
    diag = np.ones(n_basis-1)
    diag[:mode_index] = -1
    left_mat = np.diag(diag)
    left_mat = np.concatenate([left_mat,np.zeros((n_basis-1,1))],axis=1)
    right_mat = np.diag(-diag)
    right_mat = np.concatenate([np.zeros((n_basis-1,1)),right_mat],axis=1)
    mat = left_mat + right_mat
    if reshape:
        mat = np.kron(np.eye(k),mat)
    h = np.zeros(mat.shape[0]) - eps
    mat = -mat    
    return mat,h

def get_mode_constraint_mat_l(n_basis,mode_index_l,k):
    mat_l = []
    h_l = []
    for mode_index in mode_index_l:
        mat,h=get_mode_constraint_mat(n_basis,mode_index,k,reshape=False)

        mat_l.append(mat)
        h_l.append(h)
    # mat_l = np.concatenate(mat_l,axis=0)
    mat_l = scipy.linalg.block_diag(*mat_l)
    h_l = np.concatenate(h_l,axis=0)
    return mat_l,h_l



def opt_beta(X,H,f,G,h,cross_penalty_mixing=None,lam_beta=0.,lam_beta_cross=0.):
    
    P = 2*np.kron(H.dot(H.T), f.T.dot(f)) 
    # cross_penalty_mixing=2 * lam_beta_cross * (np.ones(P.shape) - np.eye(P.shape[0])) # seperately control per column norm and cross terms; care about all cross terms
    # dim=P.shape[0]
    # cross_penalty_mixing=2 * lam_beta_cross * np.diag(np.ones(dim),1)[:dim,:dim] # seperately control per column norm and cross terms; only care about adjacent betas
    P  = P + 2 * lam_beta * np.eye(P.shape[0]) + cross_penalty_mixing
    q = -2 * (f.T.dot(X).dot(H.T)).reshape(-1,order='F')
    P = matrix(P)
    q = matrix(q)

    sol = solvers.qp(P,q,G,h)
    vecbeta = np.array(sol['x'])
    nbasis=f.shape[1]
    beta = vecbeta.reshape(nbasis,-1,order='F')
    return beta

def opt_H(X,f,beta,G_h,h_h,lam_h=0.,avg_X=False):
    W = f.dot(beta)
    ntrials=X.shape[1]
    I = np.eye(ntrials)

    # if avg_X:
        # X = X.mean(axis=1,keepdims=True)
    if avg_X: 
        X[:] = X.mean(axis=1,keepdims=True)


    P = 2*np.kron(I,W.T.dot(W))
    P = P + 2 * lam_h * np.eye(P.shape[0])
    q = -2* (W.T.dot(X)).reshape(-1,order='F')
    P = matrix(P)
    q=matrix(q)

    sol = solvers.qp(P,q,G_h,h_h)
    vecH = np.array(sol['x'])
    nfac=beta.shape[1]
    H = vecH.reshape(nfac,-1,order='F')
    # if avg_X:
        # H = np.tile(H,(1,ntrials))
    
    return H
    

def get_reconstruction_error(X,f,beta,H):
    # pdb.set_trace()
    l = np.linalg.norm(X - f.dot(beta).dot(H)) **2
    return l

def get_reconstruction_error_and_reg(X,f,beta,H,lam_beta=0.,lam_h=0.,lam_beta_cross=0.,cross_penalty_mixing=None):
    l = get_reconstruction_error(X,f,beta,H)
    reg_beta = lam_beta*np.linalg.norm(beta) **2
    # cross=beta.T.dot(beta)
    # cross[np.diag_indices_from(cross)] = 0.
    # reg_beta_cross = lam_beta_cross*np.linalg.norm(cross)**2
    vecbeta=beta.reshape(-1,order='F')
    reg_beta_cross = lam_beta_cross * vecbeta.dot(cross_penalty_mixing).dot(vecbeta)
    reg_H = lam_h * np.linalg.norm(H) **2
    l_tot = l+reg_beta + reg_H + reg_beta_cross
    return l_tot,l,reg_beta,reg_H, reg_beta_cross


def do_unimodal_nmf(X,n_components=5,lam_beta=0.,lam_h=0.,lam_beta_cross=0.,
                    n_basis=5, degree=3, mode_ind_l=None,
                    n_iter_max = 100, rtol= 1e-5,do_plot_basis=False,
                    norm_H = 'max',norm_W=None,
                    verbose=True
                    ):
    assert n_basis==n_components
    vecbeta_l = n_basis * n_components
    vecH_l = n_components * X.shape[1]
    n_pt = X.shape[0]

    if mode_ind_l is None:
        mode_ind_l =np.arange(n_components)

    G_uni,h_uni=get_mode_constraint_mat_l(n_basis,mode_ind_l,n_components)
    # pdb.set_trace()

    G_beta = -np.eye(vecbeta_l)
    h_beta = np.zeros(vecbeta_l)

    G = np.concatenate([G_beta,G_uni],axis=0)
    h = np.concatenate([h_beta,h_uni],axis=0)
    G = matrix(G)
    h=matrix(h)

    G_h = -np.eye(vecH_l)
    h_h = np.zeros(vecH_l)
    G_h = matrix(G_h)
    h_h = matrix(h_h)

    std = np.std(X)
    beta_init = np.sqrt(std) * np.abs(np.random.normal(size=(n_basis,n_components)))
    H_init = np.sqrt(std) * np.abs(np.random.normal(size=(n_components,X.shape[1])))
    f = get_basis(n_basis,n_pt=n_pt,degree=degree,do_plot=do_plot_basis)      

    # penalty_dim = vecbeta_l
    # n_ones=int(n_basis //2)-1
    # circ_generator = np.zeros(n_basis)
    # circ_generator[1:n_ones+1] = 1
    # cross_penalty_mixing=scipy.linalg.circulant(circ_generator).T
    # print(cross_penalty_mixing)
    # print(cross_penalty_mixing.shape)
    # cross_penalty_mixing = np.kron(np.eye(n_components),cross_penalty_mixing)
    # cross_penalty_mixing = 2 * lam_beta_cross * cross_penalty_mixing
    # cross_penalty_mixing=2 * lam_beta_cross * np.diag(np.ones(dim),1)[:dim,:dim] # seperately control per column norm and cross terms; only care about adjacent betas
    # cross_penalty_mixing=2 * lam_beta_cross * (np.ones(vecbeta_l) - np.eye(vecbeta_l)) # seperately control per column norm and cross terms; care about all cross terms # might be incorrect
    # cross term should be block diagonals, otherwise would be crossing betas corresponding to different basis from two factors
    # cross_penalty_mixing = 2 * lam_beta_cross * (np.tile(np.eye(n_basis),[n_components,n_components]) -np.eye(vecbeta_l)) 

    # actually cross term should involve f^Tf, instead of all 1, since W^TW = beta^Tf^Tfbeta
    I_k = np.eye(n_components)
    I_Kf =np.kron(I_k,f)
    M = np.tile(np.eye(n_pt),[n_components,n_components])
    cross_penalty_mixing = I_Kf.T.dot(M).dot(I_Kf)
    cross_penalty_mixing = 2 * lam_beta_cross *(cross_penalty_mixing - np.eye(vecbeta_l))
    



    beta = beta_init
    H = H_init
    l_tot,l,reg_beta,reg_H,reg_beta_cross = get_reconstruction_error_and_reg(X,f,beta,H,lam_beta=lam_beta,lam_h=lam_h,lam_beta_cross=lam_beta_cross,cross_penalty_mixing=cross_penalty_mixing)

    l_tot_l = [l_tot]
    l_l = [l]
    reg_beta_l = [reg_beta]
    reg_H_l = [reg_H]
    reg_beta_cross_l=[reg_beta_cross]
    success = False
    for nn in range(n_iter_max):
        beta = opt_beta(X,H,f,G,h,lam_beta=lam_beta,lam_beta_cross=lam_beta_cross,cross_penalty_mixing=cross_penalty_mixing)
        H = opt_H(X,f,beta,G_h,h_h,lam_h=lam_h)
        l_tot,l,reg_beta,reg_H,reg_beta_cross = get_reconstruction_error_and_reg(X,f,beta,H,lam_beta=lam_beta,lam_h=lam_h,lam_beta_cross=lam_beta_cross,cross_penalty_mixing=cross_penalty_mixing)


        l_tot_l.append(l_tot)
        l_l.append(l)
        reg_beta_l.append(reg_beta)
        reg_H_l.append(reg_H)
        reg_beta_cross_l.append(reg_beta_cross)

        if verbose:
            print(f'iter{nn}--tot: {l_tot:.5f}, recon: {l:.5f}, reg_beta: {reg_beta:.5f}, reg_H: {reg_H:.5f}, reg_beta_cross: {reg_beta_cross:.5f}')

        rimprove=(l_tot_l[nn] - l_tot_l[nn+1]) / l_tot_l[nn]
        if  (rimprove<=rtol) and (rimprove>=-1e-7): # don't know why total loss could increase
            success=True
            break
        
    print(f'success={success}')
    loss_history = pd.concat(
        {
            'total':pd.Series(l_tot_l),
            'reconstruction':pd.Series(l_l),
            'reg_beta':pd.Series(reg_beta_l),
            'reg_H':pd.Series(reg_H_l),
            'reg_beta_cross':pd.Series(reg_beta_cross_l),

        }, axis=1
    )
    W = f.dot(beta)
    eps=1e-10
    if norm_H=='max':
        factor = H.max(axis=1)
        H = H / (factor[:,None]+eps)
        
        W = W * factor[None,:]
    if norm_W=='max':
        factor = W.max(axis=0)
        W = W / (factor[None,:]+eps)
        H = H * factor[:,None]

    return f,beta,H,W,loss_history

from sklearn.decomposition import NMF
def do_nmf_wrapper(X,ma_thresh=0.1,norm_ratio=0.99,clip_ratio=0.99,**kwargs):
    kwargs_ = dict(n_components=2,
                    n_iter_max = 200, rtol= 1e-4,
                    norm_H = None,norm_W='max',normalize_error=True,
                    verbose=False)
    kwargs_.update(kwargs)
    norm_W = kwargs_['norm_W']
    norm_H = kwargs_['norm_H']
    n_components = kwargs_['n_components']
    n_iter_max = kwargs_['n_iter_max']
    normalize_error = kwargs_['normalize_error']

    if clip_ratio is not None: # clip outliers
        clip_factor = np.quantile(X.dropna().values,clip_ratio)
        X[X>clip_factor] = clip_factor
    if norm_ratio is not None: # max norm
        norm_factor = np.quantile(X.dropna().values,norm_ratio)
        X = X /norm_factor

    if isinstance(X,pd.DataFrame):
        X = X.dropna(axis=1,how='all').dropna(axis=0,how='all')
        isdf=True
        Xv = X.fillna(0).values
    else:
        X[np.isnan(X)] = 0.
        Xv = X
    nmf = NMF(n_components=n_components,max_iter=n_iter_max)
    W=nmf.fit_transform(Xv)
    H =nmf.components_
    recon_error  =nmf.reconstruction_err_
    eps=1e-10
    if norm_H=='max':
        factor = H.max(axis=1)
        H = H / (factor[:,None]+eps)
        
        W = W * factor[None,:]
    if norm_W=='max':
        factor = W.max(axis=0)
        W = W / (factor[None,:]+eps)
        H = H * factor[:,None]
    elif norm_W=='mean':
        factor = W.mean(axis=0)
        W = W / (factor[None,:]+eps)
        H = H * factor[:,None]
    elif norm_W=='norm':
        factor = np.sqrt(np.diag(W.T.dot(W)))
        W = W / (factor[None,:]+eps)
        H = H * factor[:,None]

    
    W_hat_ma = W
    H_hat_ma = H
    X_hat = W_hat_ma.dot(H_hat_ma)
    if isdf:
        W_hat_ma = pd.DataFrame(W_hat_ma,index=X.index)
        H_hat_ma = pd.DataFrame(H_hat_ma,columns=X.columns)
        X_hat = pd.DataFrame(X_hat,index=X.index,columns=X.columns)
    if normalize_error:
        recon_error = recon_error**2 / np.linalg.norm(Xv)**2
    return W_hat_ma,H_hat_ma,X_hat,recon_error


def do_unimodal_nmf_wrapper(X,ma_thresh=0.1,norm_ratio=0.99,clip_ratio=0.99,**kwargs):
    '''
    wrapper for do_unimodal_nmf, X can be df, will filter W with low activations
    fill na with 0
    '''
    kwargs_ = dict(n_components=20,lam_beta=20.,lam_h=0.1,lam_beta_cross=20.,
                    n_basis=20, degree=3, mode_ind_l=None,
                    n_iter_max = 100, rtol= 1e-4,do_plot_basis=False,
                    norm_H = None,norm_W='max',
                    verbose=False)
    kwargs_.update(kwargs)
    norm_W = kwargs_['norm_W']

    if clip_ratio is not None: # clip outliers
        clip_factor = np.quantile(X.dropna().values,clip_ratio)
        X[X>clip_factor] = clip_factor
    if norm_ratio is not None: # max norm
        norm_factor = np.quantile(X.dropna().values,norm_ratio)
        X = X /norm_factor
    


    if isinstance(X,pd.DataFrame):
        X = X.fillna(axis=0,method='ffill')
        # X = X.dropna(axis=1,how='all').dropna(axis=0,how='all')
        isdf=True
        # Xv = X.fillna(0).values
        Xv = X.values
    else:
        X[np.isnan(X)] = 0.

    f,beta,H_hat,W_hat,loss_history = do_unimodal_nmf(Xv,**kwargs_)
    # pdb.set_trace()
    # ma = W_hat.sum(axis=0) > W_hat.sum(axis=0).max() * ma_thresh # mask W and H based on peak activation
    
    if norm_W is not None:
        ma = H_hat.max(axis=1) > H_hat.max(axis=1).max() * ma_thresh
    else:
        ma = W_hat.max(axis=0) > W_hat.max(axis=0).max() * ma_thresh # mask W and H based on peak activation
    W_hat_ma = W_hat[:,ma]
    H_hat_ma = H_hat[ma,:]
    X_hat = W_hat_ma.dot(H_hat_ma)
    if norm_factor is not None:
        X_hat = X_hat * norm_factor
    if isdf:
        W_hat_ma = pd.DataFrame(W_hat_ma,index=X.index)
        H_hat_ma = pd.DataFrame(H_hat_ma,columns=X.columns)
        X_hat = pd.DataFrame(X_hat,index=X.index,columns=X.columns)
    

    return W_hat_ma, H_hat_ma, X_hat, loss_history

import matplotlib.gridspec as gridspec
def plot_ratemap_and_wh(X_df,W_hat_ma,H_hat_ma,X_hat=None,region='',exp='',uid='',isnovel=''):
    fig=plt.figure(figsize=(16,8))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1,1,0.05])
    ax00=ax=fig.add_subplot(gs[0, 0])
    # cbar_ax=fig.add_subplot(gs[0, 1])
    fig,ax=ph.heatmap(X_df.T,ax=ax,vmax_quantile=0.99,cbar_ax=None,cbar=False,fig=fig)
    vmin,vmax=ax.collections[0].get_clim()
    ph.plot_day_on_heatmap(X_df,level=0,axis=1,ax=ax)
    ax.set(ylabel='Day-Trial',title=f'{region}, exp{exp}, cell {uid}, isnovel {isnovel}',xlabel='')

    ax=fig.add_subplot(gs[0, 1])
    ax.invert_yaxis()
    ax.plot(H_hat_ma.T,np.arange(H_hat_ma.shape[1]),marker='o')
    ph.plot_day_on_heatmap(X_df,level=0,axis=1,ax=ax)
    ax.set(ylabel='Day-Trial',yticklabels=[],title='H')

    ax=fig.add_subplot(gs[1, 0],sharex=ax00)
    ax.plot(W_hat_ma)
    plt.tight_layout()
    ax.set(xlabel='Position',ylabel='W')
    if X_hat is not None:
        ax=fig.add_subplot(gs[1, 1],sharex=ax00)
        cbar_ax = fig.add_subplot(gs[1,2])
        fig,ax=ph.heatmap(X_hat.T,ax=ax,vmax_quantile=0.99,cbar_ax=cbar_ax,vmin=vmin,vmax=vmax,fig=fig)
        ax.set_title('Reconstruction')
        ph.plot_day_on_heatmap(X_df,level=0,axis=1,ax=ax)
    
    plt.tight_layout()
    # plt.subplots_adjust(hspace=0.3, wspace=0.5) 
    return fig

# old way using axs
# def plot_ratemap_and_wh(X_df,W_hat_ma,H_hat_ma,X_hat=None,fig=None,axs=None,
#                         region='',exp='',uid='',isnovel='',
#                         ):
#     if axs is None:
#         fig,axs = plt.subplots(2,2,figsize=(8,8))
#     ax=axs[0,0]
#     fig,ax=ph.heatmap(X_df.T,ax=ax,vmax_quantile=0.99)
#     ph.plot_day_on_heatmap(X_df,level=0,axis=1,ax=ax)
#     ax.set(ylabel='Day-Trial',title=f'{region}, exp{exp}, cell {uid}, isnovel {isnovel}',xlabel='')

#     ax=axs[0,1]
#     ax.invert_yaxis()
#     ax.plot(H_hat_ma.T,np.arange(H_hat_ma.shape[1]),marker='o')
#     ph.plot_day_on_heatmap(X_df,level=0,axis=1,ax=ax)
#     ax.set(ylabel='Day-Trial',yticklabels=[],title='H')


#     ax=axs[1,0]
#     ax.plot(W_hat_ma)
#     plt.tight_layout()
#     ax.set(xlabel='Position',ylabel='W')
#     if X_hat is not None:
#         ax = axs[1,1]
#         fig,ax=ph.heatmap(X_hat.T,ax=ax,vmax_quantile=0.99)
#         ph.plot_day_on_heatmap(X_df,level=0,axis=1,ax=ax)
#     else:
#         axs[1,1].remove()
    
#     return fig,axs


def pick_nmf_rank_one_neuron(X_df,rank_l=range(1,5),**kwargs):
    recon_l = []
    diff_l = []
    for ii,r in enumerate(rank_l):
        W_hat_ma,H_hat_ma,X_hat,recon = do_nmf_wrapper(X_df,n_components=r,**kwargs)
        recon_l.append(recon)

    recon_l = np.array(recon_l)
    sec_dir=np.diff(np.diff(recon_l))
    neg_sec_dir=sec_dir[sec_dir < 0]
    if len(neg_sec_dir) > 0:
        neg_sec_dir_ind=np.nonzero(sec_dir < 0)[0]
        max_neg_sec_dir_ind=np.max(neg_sec_dir_ind)
        if max_neg_sec_dir_ind < sec_dir.shape[0]-1: # if the negative curvature is not the end
            max_sec_dir=np.argmax(sec_dir[max_neg_sec_dir_ind+1:]) + max_neg_sec_dir_ind + 1  # find the largest positive curvature on the right side of the max negative sec dir ind
            ind_selected=max_sec_dir = max_sec_dir + 1
        else: 
            ind_selected = len(rank_l)-1#rank_l[-1]

    else: # no negative curvature, pick the largest positive
        max_sec_dir=np.argmax(sec_dir) #which difference between line segments is max, n means difference between line segment n and n+1; line segment connects vertex n and n+1. So vertex n+1 is the inflection point 
        ind_selected = max_sec_dir + 1
    return rank_l[ind_selected], recon_l
