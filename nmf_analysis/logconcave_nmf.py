import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import sys,os,copy,pdb,importlib,pickle
from importlib import reload
import pandas as pd
from cvxopt import matrix, solvers
solvers.options['show_progress']=False

def opt_W(X,H,G,h,log_concave_penalty_mat,lam_W=0.,lam_concave=0.):
    I = np.eye(X.shape[0])
    P = 2*np.kron(H.dot(H.T),I)
    log_concave_penalty_mat_full = np.kron(np.eye(H.shape[0]),log_concave_penalty_mat)
#     P = P + 2 * lam_W * np.eye(P.shape[0])
    P = P + 2 * lam_W * np.eye(P.shape[0]) + lam_concave * log_concave_penalty_mat_full
    q = -2 * (X.dot(H.T)).reshape(-1,order='F')
    P = matrix(P)
    q = matrix(q)

    sol = solvers.qp(P,q,G,h)
    vecW = np.array(sol['x'])
    W = vecW.reshape(X.shape[0],-1,order='F')
    return W
    
    


def opt_H(X,W,G_h,h_h,lam_h=0.):

    I = np.eye(X.shape[1])
    P = 2*np.kron(I,W.T.dot(W))
    P = P + 2 * lam_h * np.eye(P.shape[0])
    q = -2* (W.T.dot(X)).reshape(-1,order='F')
    P = matrix(P)
    q=matrix(q)

    sol = solvers.qp(P,q,G_h,h_h)
    vecH = np.array(sol['x'])
    nfac=W.shape[1]
    H = vecH.reshape(nfac,-1,order='F')
    return H
    

def get_reconstruction_error(X,W,H):
    # pdb.set_trace()
    l = np.linalg.norm(X - W.dot(H)) **2
    return l

def get_reconstruction_error_and_reg(X,W,H,log_concave_penalty_mat=None,lam_W=0.,lam_h=0.,lam_concave=0.):
    l = get_reconstruction_error(X,W,H)
    reg_W = lam_W*np.linalg.norm(W) **2
    # cross=beta.T.dot(beta)
    # cross[np.diag_indices_from(cross)] = 0.
    # reg_beta_cross = lam_beta_cross*np.linalg.norm(cross)**2
    reg_concave = lam_concave * np.linalg.norm(W.T.dot(log_concave_penalty_mat).dot(W))**2
    
    reg_H = lam_h * np.linalg.norm(H) **2
    l_tot = l+reg_W + reg_H + reg_concave
    return l_tot,l,reg_W,reg_H, reg_concave

def get_log_concave_penalty_mat(n_pos = 40):
        
    log_concave_penalty_mat = np.eye(n_pos)
    log_concave_penalty_mat[0,0] = 0
    log_concave_penalty_mat[-1,-1] = 0
    log_concave_penalty_mat = log_concave_penalty_mat-np.diag(np.ones(log_concave_penalty_mat.shape[0]-2),k=2)
    log_concave_penalty_mat = -log_concave_penalty_mat # want to minimize the quadratic form
    return log_concave_penalty_mat

def do_logconcave_nmf(X,n_components=2,lam_W=0.,lam_h=0.,lam_concave=0.,
                    n_iter_max = 100, rtol= 1e-5,
                    norm_H = 'max',
                    verbose=True,**kwargs
                    ):
    
    vecW_l = X.shape[0] * n_components
    vecH_l = n_components * X.shape[1]
    n_pt = X.shape[0]

    log_concave_penalty_mat = get_log_concave_penalty_mat(n_pt)  


    G_W = -np.eye(vecW_l)
    h_W = np.zeros(vecW_l)

    G = G_W#np.concatenate([G_beta,G_uni],axis=0)
    h = h_W#np.concatenate([h_beta,h_uni],axis=0)
    G = matrix(G)
    h=matrix(h)

    G_h = -np.eye(vecH_l)
    h_h = np.zeros(vecH_l)
    G_h = matrix(G_h)
    h_h = matrix(h_h)

    std = np.std(X)
    W_init = np.sqrt(std) * np.abs(np.random.normal(size=(n_pt,n_components)))
    H_init = np.sqrt(std) * np.abs(np.random.normal(size=(n_components,X.shape[1])))


    W = W_init
    H = H_init
    
    l_tot,l,reg_W,reg_H, reg_concave = get_reconstruction_error_and_reg(X,W,H,lam_W=lam_W,lam_h=lam_h,log_concave_penalty_mat=log_concave_penalty_mat,lam_concave=lam_concave)

    l_tot_l = [l_tot]
    l_l = [l]
    reg_W_l = [reg_W]
    reg_H_l = [reg_H]
    reg_concave_l=[reg_concave]
    success = False
    for nn in range(n_iter_max):
        W = opt_W(X,H,G,h,lam_W=lam_W,log_concave_penalty_mat=log_concave_penalty_mat)
        H = opt_H(X,W,G_h,h_h,lam_h=lam_h)
        l_tot,l,reg_W,reg_H,reg_concave = get_reconstruction_error_and_reg(X,W,H,lam_W=lam_W,lam_h=lam_h,log_concave_penalty_mat=log_concave_penalty_mat,lam_concave=lam_concave)

        l_tot_l.append(l_tot)
        l_l.append(l)
        reg_W_l.append(reg_W)
        reg_H_l.append(reg_H)
        reg_concave_l.append(reg_concave)

        if verbose:
            print(f'iter{nn}--tot: {l_tot:.5f}, recon: {l:.5f}, reg_W: {reg_W:.5f}, reg_H: {reg_H:.5f}, reg_concave: {reg_concave:.5f}')

        rimprove=(l_tot_l[nn] - l_tot_l[nn+1]) / l_tot_l[nn]
        if  (rimprove<=rtol) and (rimprove>=0):
            success=True
            break
        
    print(f'success={success}')
    loss_history = pd.concat(
        {
            'total':pd.Series(l_tot_l),
            'reconstruction':pd.Series(l_l),
            'reg_W':pd.Series(reg_W_l),
            'reg_H':pd.Series(reg_H_l),
            'reg_concave':pd.Series(reg_concave_l),

        }, axis=1
    )
    
    eps=1e-10
    if norm_H=='max':
        factor = H.max(axis=1)
        H = H / (factor[:,None]+eps)
        
        W = W * factor[None,:]

    return W,H,loss_history

def do_logconcave_nmf_wrapper(X,ma_thresh=0.1,**kwargs):
    '''
    wrapper for do_unimodal_nmf, X can be df, will filter W with low activations
    fill na with 0
    '''
    kwargs_ = dict(n_components=2,lam_beta=0.,lam_h=0.,lam_beta_cross=0.,
                    n_basis=5, n_pt=40, degree=3, mode_ind_l=None,
                    n_iter_max = 100, rtol= 1e-5,do_plot_basis=False,
                    norm_H = 'max',
                    verbose=True)
    kwargs_.update(kwargs)

    if isinstance(X,pd.DataFrame):
        X = X.dropna(axis=1,how='all').dropna(axis=0,how='all')
        isdf=True
        Xv = X.fillna(0).values
    else:
        X[np.isnan(X)] = 0.

    W_hat,H_hat,loss_history = do_logconcave_nmf(Xv,**kwargs_)
    # pdb.set_trace()
    # ma = W_hat.sum(axis=0) > W_hat.sum(axis=0).max() * ma_thresh # mask W and H based on peak activation
    ma = W_hat.max(axis=0) > W_hat.max(axis=0).max() * ma_thresh # mask W and H based on peak activation
    W_hat_ma = W_hat[:,ma]
    H_hat_ma = H_hat[ma,:]
    X_hat = W_hat_ma.dot(H_hat_ma)
    if isdf:
        W_hat_ma = pd.DataFrame(W_hat_ma,index=X.index)
        H_hat_ma = pd.DataFrame(H_hat_ma,columns=X.columns)
        X_hat = pd.DataFrame(X_hat,index=X.index,columns=X.columns)
    

    return W_hat_ma, H_hat_ma, X_hat, loss_history