import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import sys,os,copy,pdb,importlib,pickle
from importlib import reload
import pandas as pd

sys.path.append('/mnt/home/szheng/projects/place_variability/code')
sys.path.append('/mnt/home/szheng/projects/util_code')
import traceback
import misc
import database

import unimodal_nmf as unmf

def get_support(x,max_thresh=0.5,extend=2):
    '''
    x: series, a column of W_hat_ma, place field template/factor; 
    find the position bin where x is at max_thresh * max, then extend the range by the factor extend
    
    '''
    maxlen = x.shape[0]
    max_thresh_val = x.max()*max_thresh
    ma = (x - max_thresh_val) > 0
#     pdb.set_trace()
    support= np.array(x.index[ma.diff().fillna(False)])
    extend_side = None
    if ma.iloc[0]:
        extend_side='right'
        support = np.insert(support,0,0)
    elif ma.iloc[-1]:
        extend_side = 'left'
        support = np.append(support,maxlen-1)
        
    
    if extend is not None:
        length = support[-1]-support[0]
        if extend_side is None:
            len_to_extend_per_side=int(length * (extend-1)//2)
        else:
            len_to_extend_per_side=int(length * (extend-1)) # if only one side then extend the whole length in one direction
        left=max(support[0] - len_to_extend_per_side,0)
        right=min(support[-1] + len_to_extend_per_side,maxlen-1)
        support = np.array([*support,left,right])
        
    return support

def get_field(W,out_of_field_size_thresh=5,**kwargs):
    field_range=W.apply(get_support,axis=0,**kwargs)
    
    if field_range.shape[0]==2:
        field_range.index=['start','end']
    else:
        field_range.index=['start','end','window_start','window_end']
    field_range = field_range.T
    nfield=field_range.shape[0]
    for i in range(nfield):
        if i>0:
            window_start_curr = field_range.iloc[i]['window_start']
            start_curr = field_range.iloc[i]['start']
            prev_end = field_range.iloc[i-1]['end']
            if window_start_curr < prev_end:
                field_range.iloc[i]['window_start'] = prev_end
            if start_curr < prev_end:
                field_range.iloc[i]['start'] = prev_end

        if i<(nfield-1):
            window_end_curr = field_range.iloc[i]['window_end']
            end_curr = field_range.iloc[i]['end']
            next_start = field_range.iloc[i+1]['start']
            if window_end_curr > next_start:
                field_range.iloc[i]['window_end'] = next_start
            if end_curr > next_start:
                field_range.iloc[i]['end'] = next_start

    


    out_of_field_size=(field_range['window_end'] - field_range['end']) + (field_range['start'] - field_range['window_start'])
    ma = out_of_field_size > out_of_field_size_thresh
    field_range_ma = field_range.loc[ma]

    return field_range, field_range_ma
        
def circular_shuffle_get_null_activation_all_field(X_df_compare,field_range,n_roll_min = 10,nrepeats = 1000):
    n_pos = X_df_compare.shape[0]
    diff_null_l=[]
    for n in range(nrepeats):
        nn=np.random.randint(n_roll_min,n_pos-n_roll_min)
        X_df_compare_null=pd.DataFrame(np.roll(X_df_compare,nn,axis=0),index=X_df_compare.index,columns=X_df_compare.columns)
        diff_null = get_activation_all_field(X_df_compare_null,field_range).loc['diff']
        diff_null_l.append(diff_null)
    diff_null_l = pd.concat(diff_null_l,axis=0)
    return diff_null_l

def get_activation_all_field(X_df_compare,field_range):
    pre_day_activation = {}
    for field_id,row in field_range.iterrows():
        pre_day_activation_per_field = {}
        ma_full = (X_df_compare.index>=row['window_start'])&(X_df_compare.index<=row['window_end'])
        ma_field = (X_df_compare.index>=row['start'])&(X_df_compare.index<=row['end'])
        outside = ma_full & np.logical_not(ma_field)
        pre_day_activation_per_field['within'] = X_df_compare.loc[ma_field].mean(axis=0)
        pre_day_activation_per_field['outside'] = X_df_compare.loc[outside].mean(axis=0)
        pre_day_activation_per_field['diff'] = pre_day_activation_per_field['within'] - pre_day_activation_per_field['outside']
        pre_day_activation_per_field = pd.concat(pre_day_activation_per_field,axis=1)
        pre_day_activation[field_id] = pre_day_activation_per_field.T
    pre_day_activation = pd.concat(pre_day_activation,axis=0).swaplevel(0,1)
    return pre_day_activation

### fit unmf and get activation

def fit_unmf_and_get_activation(X_df,get_field_day = 1,extend=2,max_thresh=0.2,**kwargs):
    day_l = X_df.columns.get_level_values(0).unique()
    compare_day_l = day_l[:get_field_day]
    X_df_compare = X_df.loc[:,compare_day_l]
    X_df_field = X_df.loc[:,[get_field_day]] #X_df_# 
    norm_W = kwargs.get('norm_W','mean')

    out_of_field_size_thresh = kwargs.get('out_of_field_size_thresh',5)
    W_hat_ma, H_hat_ma,X_hat, loss_history = unmf.do_unimodal_nmf_wrapper(X_df_field,**kwargs)
    field_range, field_range_ma=get_field(W_hat_ma,extend=extend,max_thresh=max_thresh,out_of_field_size_thresh=out_of_field_size_thresh)
    print(W_hat_ma.shape[1])
    print(field_range)
    print(field_range_ma)
    pre_day_activation=get_activation_all_field(X_df_compare,field_range_ma)

    
    return pre_day_activation,W_hat_ma, H_hat_ma,field_range, field_range_ma

def fit_unmf_and_get_activation_all_comparisons_onecell(X_df,**kwargs):
    '''
    see fit_unmf_and_get_activation and unmf.do_unimodal_nmf_wrapper for kwargs
    pre_day_activation_d: (uid x get_field_day x [within,outside,diff] x field_id) x (day x trial)

    '''
    day_l = X_df.columns.get_level_values(0).unique()
    pre_day_activation_d = {}
    W_hat_d={}
    H_hat_d={}
    field_range_d={}
    field_range_ma_d = {}
    for d in day_l[1:]: # day for getting field
        get_field_day = d
        pre_day_activation,W_hat_ma, H_hat_ma,field_range, field_range_ma = fit_unmf_and_get_activation(X_df,get_field_day = get_field_day,**kwargs)
        pre_day_activation_d[d] = pre_day_activation
        W_hat_d[d] = W_hat_ma
        H_hat_d[d] = H_hat_ma
        field_range_d[d] = field_range
        field_range_ma_d[d] = field_range_ma

    pre_day_activation_d = pd.concat(pre_day_activation_d,axis=0)
    W_hat_d = pd.concat(W_hat_d,axis=0)
    H_hat_d = pd.concat(H_hat_d,axis=0)
    field_range_d = pd.concat(field_range_d,axis=0)
    field_range_ma_d = pd.concat(field_range_ma_d,axis=0)
    return pre_day_activation_d,W_hat_d,H_hat_d,field_range_d,field_range_ma_d



def shuffle_all_day(X_df,min_roll=10):
    gpb=X_df.groupby(axis=1,level=0)
    npos = X_df.shape[0]
    ndays=len(gpb)
    n_l = []
    st = min_roll
    # ed = npos-min_roll
    # for ii in range(ndays):
    #     # pdb.set_trace()
    #     npt_left=ndays-ii
    #     ed = npos-min_roll - npt_left * min_roll
    #     n_next=np.random.randint(st,ed)
    #     n_l.append(n_next)
    #     st=n_next + min_roll
    # n_l = np.array(n_l)
    # n_l = np.random.permutation(n_l)
    
    # print(n_l)    
    index,col=X_df.index,X_df.columns
    X_df_null = []
    for d,val in gpb:
        n = np.random.randint(min_roll,npos-min_roll)
        # n = n_l[d]
        X_df_null.append(np.roll(val,n,axis=0))
    X_df_null=np.concatenate(X_df_null,axis=1)
    X_df_null = pd.DataFrame(X_df_null,index=index,columns=col)
    return X_df_null




def block_reusing_index(H,ntrial_win=3,method='mov_avg_max'):
    '''
    '''
    if method=='mov_avg_max':
        # moving average of activations
        H_mov_avg = H.groupby(level=0,axis=1).apply(lambda x:x.rolling(ntrial_win,axis=1).mean())

        # max within day
        val_per_day = H_mov_avg.groupby(level=0,axis=1).max()
    elif method=='mean':
        val_per_day = H.groupby(level=0,axis=1).mean()

    # two largest day
    two_largest_day=val_per_day.apply(lambda x:x.nlargest(2).reset_index(drop=True),axis=1)

    # ratio of second / largest day
    ratio = two_largest_day[1] / two_largest_day[0]

    return ratio,two_largest_day
    

def test_one_neuron(X):
    n_basis = 20
    lam_beta=lam_beta_cross = 20.#1.
    lam_h = .1 # 0.1
    rtol=1e-3
    # lam_beta_cross = 10.#0.1
    W_hat_ma, H_hat_ma,X_hat, loss_history = unmf.do_unimodal_nmf_wrapper(X_df,ma_thresh=0.05,
                                                                    n_components=n_basis,lam_beta=lam_beta,lam_h=lam_h,lam_beta_cross=lam_beta_cross,
                                                                    n_basis=n_basis,
                                                                    n_iter_max=100,rtol=1e-3,
                                                                    verbose=True,norm_H=None,norm_W='max'
                                                                )


    field_range=get_field(W_hat_ma,extend=2,max_thresh=0.1)

    import tqdm
    # shuffle do nmf
    nrepeats = 20
    n_roll_min=10
    H_hat_ma_null_l = []
    recon_null_l=[]
    loss_history_l = []
    for n in tqdm.tqdm(range(nrepeats)):
        X_df_null=shuffle_all_day(X_df,min_roll=n_roll_min)
        W_hat_ma_null, H_hat_ma_null,X_hat_null, loss_history = unmf.do_unimodal_nmf_wrapper(X_df_null,ma_thresh=0.05,
                                                                    n_components=n_basis,lam_beta=lam_beta,lam_h=lam_h,lam_beta_cross=lam_beta_cross,
                                                                    n_basis=n_basis,
                                                                    n_iter_max=100,rtol=rtol,
                                                                    verbose=False,norm_H=None,norm_W='max',
                                                                )
        H_hat_ma_null_l.append(H_hat_ma_null)
        recon_null = loss_history['reconstruction'].iloc[-1]
        recon_null_l.append(recon_null)
        loss_history_l.append(loss_history)
    H_hat_ma_null_l = pd.concat(H_hat_ma_null_l,axis=0)
    loss_history_l = pd.concat(loss_history_l,axis=0)
    
import tqdm
def nmf_pick_rank_shuffle_one_neuron(X_df,rank_l=range(1,5),nrepeats=1000,alpha=0.05,n_roll_min=10,**kwargs):
    '''
    pick rank using curvature / elbow
    circularly shuffle each day, get nmf reconstruction error null distribution
    '''
    n_components,recon_l = unmf.pick_nmf_rank_one_neuron(X_df,rank_l=rank_l,**kwargs)
    W_hat_ma,H_hat_ma,X_hat,recon = unmf.do_nmf_wrapper(X_df,n_components=n_components,**kwargs)
    H_hat_ma_null_l = []
    recon_null_l=[]
    H = H_hat_ma

    # for n in tqdm.tqdm(range(nrepeats)):
    for n in range(nrepeats):
        X_df_null=shuffle_all_day(X_df,min_roll=n_roll_min)
        W_hat_ma_null, H_hat_ma_null,X_hat_null, recon_null = unmf.do_nmf_wrapper(X_df_null,n_components=n_components,**kwargs)
        # H_hat_ma_null_l.append(H_hat_ma_null)
        recon_null_l.append(recon_null)
    # H_hat_ma_null_l = pd.concat(H_hat_ma_null_l,axis=0)
    recon_null_l = np.array(recon_null_l)
    p = np.mean(recon > recon_null_l)
    recon_null_mean= np.mean(recon_null_l)
    recon_null_ci_up = np.quantile(recon_null_l,1-alpha/2)
    recon_null_ci_low = np.quantile(recon_null_l,alpha/2)

    res_one = pd.Series({'recon':recon,'p':p,'null_mean':recon_null_mean,'null_ci_up':recon_null_ci_up,'null_ci_low':recon_null_ci_low})

    return res_one, recon_null_l


     