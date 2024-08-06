# %%
import numpy 
import scipy
from scipy.signal import find_peaks
import data_prep_new as dpn
import place_cell_analysis as pa
import plot_helper as ph
from importlib import reload
import itertools, sys, os, copy, pickle
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import jax
import jax.numpy as np
import jax.scipy as scipy
from jax import value_and_grad, grad, jit, vmap, jacfwd, jacrev
from jax.example_libraries import optimizers as jax_opt
import submitit

from math_functions import *

# import scipy
# from scipy.stats import norm

# %%
# sess_name="e15_13f1_220117"
# py_data_dir = "/mnt/home/szheng/ceph/ad/roman_data/e15/e15_13f1/e15_13f1_220117/py_data"
# fr_map_ = pickle.load(open(os.path.join(py_data_dir,'fr_map.p'),'rb'))

# # %%
# fr_map=fr_map_['fr_map']
# fr_map_trial=fr_map_['fr_map_trial']

# %%
# firing rate function given parameters 

def gm_func_by_trial(regressors,pars):
    '''
    pars:
        logws/: ntrial x Kfields;    amplitude
        mus/logsigmas_l:1 x Kfields; center/width
        b_l: ntrial x 1;      baseline
        S: npos x ntrial;    sparse element

    regressors:
        xs: np.arange(npos);    probe positions
    ======
    fs: npos x ntrial
    '''
    logws_l = pars['logws']
    mus_l = pars['mus']
    logsigmas_l = pars['logsigmas']
    # b_l = pars['b']
    logb_l = pars['logb']
    
    ntrials = logws_l.shape[0]

    xs = regressors['xs']
    if 'S' in pars.keys():
        S = pars['S']
    else:
        S = np.zeros((len(xs),ntrials))
    assert len(xs)==S.shape[0]

    nt,K = logws_l.shape
    xs_l = np.tile(xs[:,None],[1,nt])
    fs_all_trial_one_field_l = []
    for k in range(K):
        # fs_all_trial_one_field=scipy.stats.norm.pdf(xs_l,loc=mus_l[:,k],scale=softplus(logsigmas_l[:,k]))
        fs_all_trial_one_field = unnormalized_normal_pdf(xs_l,loc=mus_l[:,k],scale=softplus(logsigmas_l[:,k]))
        assert fs_all_trial_one_field.shape == (xs.shape[0],nt)
        fs_all_trial_one_field_l.append(fs_all_trial_one_field)
    fs_all_trial_one_field_l_stacked = np.array(fs_all_trial_one_field_l)
    ws_l = softplus(logws_l) # make sure ws nonnegative
    fs_l = np.einsum('kpn,nk->pn',fs_all_trial_one_field_l_stacked,ws_l)

    # fs_l_final = fs_l + b_l[None,:] + S
    # fs_l_final = fs_l + b_l[None,:] 
    # fs_l_final = fs_l + softplus(logb_l[None,:])
    fs_l_final = fs_l + softplus(logb_l)
    return fs_l_final

# @jit
# def mse(a,b):
#     return np.sum(1/2*(a-b)**2)

# @jit
# def mse_no_reduce(a,b):
#     return (1/2 * (a-b)**2)

# @jit
# def softplus(x):
#     # return np.log(1+np.exp(x))

#     return np.logaddexp(x,0.)
# @jit
# def inv_softplus(y):
#     # threshold = np.log(eps) + 2.
#     # is_too_small = x < np.exp(threshold)
#     # is_too_large = x > -threshold
#     # too_small_value = tf.math.log(x)
#     # too_large_value = x
#     # This `where` will ultimately be a NOP because we won't select this
#     # codepath whenever we used the surrogate `ones_like`.
#     # x = tf.where(is_too_small | is_too_large, tf.ones([], x.dtype), x)
#     # y = x + tf.math.log(-tf.math.expm1(-x))  # == log(expm1(x))
#     eps = 1e-8
#     x = np.log(1-np.exp(-y)+eps) + y # a stable version of inv softplus
#     return x




    # return np.log(np.exp(y)-1)

@jit
def quadratic_variation(x):
    return np.mean(np.diff(x,axis=0)**2)
@jit
def quadratic_variation_no_reduce(x):
    return np.diff(x,axis=0,prepend=x[[0],:])**2

def loss_no_reg(regressors,pars,ys_l,loss_type="mse",mask=None):
    if mask is None:
        mask = np.ones_like(ys_l)
    fs_l_final = gm_func_by_trial(regressors,pars)
    if loss_type=="mse":
        loss_element = mse_no_reduce(ys_l, fs_l_final)
        # loss = np.mean(loss_element * mask) # incorrect!
        loss = np.sum(loss_element * mask) / np.sum(mask)
        # loss = np.sum(loss_element * mask)
        # loss = mse(ys_l,fs_l_final)
    return loss

    

def regularization(pars,reg_pars,reg_type=None,mask=None):
    # if mask is None: # right now no use for mask
    #     mask = np.ones_like(pars['S'])
    # R_S = reg_pars['S_l1']*np.mean(mask * np.abs(pars['S']))
    R_S = 0.
    R_sigma_thresh=reg_pars['g_sigma_thresh']*np.mean(jax.nn.relu(reg_pars['sigma_thresh'] - softplus(pars['logsigmas']))) # if sigma too small, penalized; otherwise no penalty
    if reg_type is None:
        return R_S + R_sigma_thresh
    elif reg_type =="diag":
        R_mu_var = 1/reg_pars['g_mu'] * np.mean(np.var(pars['mus'],axis=0))
        R_sigma_var = 1/reg_pars['g_sigma'] * np.mean(np.var(softplus(pars['logsigmas']),axis=0))
        R_w_var = 1/reg_pars['g_w'] * np.mean(np.var(softplus(pars['logws']),axis=0))
        R_b_var = 1/reg_pars['g_b'] * np.mean(np.var(softplus(pars['logb']),axis=0))
        return R_mu_var + R_sigma_var +R_w_var+R_b_var+ R_S + R_sigma_thresh
    elif reg_type=="quad_variation":
        R_mu_var = 1/reg_pars['g_mu'] * quadratic_variation(pars['mus'])
        R_sigma_var = 1/reg_pars['g_sigma'] * quadratic_variation(softplus(pars['logsigmas']))
        R_w_var = 1/reg_pars['g_w'] * quadratic_variation(softplus(pars['logws']))
        # R_b_var = 1/reg_pars['g_b'] * quadratic_variation(pars['b'])
        R_b_var = 1/reg_pars['g_b'] * quadratic_variation(softplus(pars['logb']))
        return R_mu_var + R_sigma_var + R_w_var+ R_b_var + R_S + R_sigma_thresh

def loss_total(regressors,pars,ys_l,reg_pars,loss_type="mse",reg_type=None,mask=None):
    l = loss_no_reg(regressors,pars,ys_l,loss_type=loss_type,mask=mask)
    reg = regularization(pars,reg_pars,reg_type=reg_type,mask=mask)
    return l + reg
    # return l 
    
#%% initializations
def init_one_trial(ys,K=2):
    peaks = numpy.array(find_peaks(ys,distance=20)[0])
    ind_in_peaks=numpy.flip(numpy.argsort(ys[peaks]))
    init_locs_ = peaks[ind_in_peaks][:K]
    if len(init_locs_) < K:
        for _ in range(K - len(init_locs_)): # to make sure all trials have the same length of init values
            init_locs_=numpy.append(init_locs_,numpy.nan)
        init_locs = init_locs_
    else:
        init_locs = init_locs_
    return init_locs

def realign_trial_with_nan(tr1,tr2):
    '''
    [x,x,...,nan,nan]
    [x,x,...,x,x]
    find the place for 
    '''
    tr1_nonna=tr1[~numpy.isnan(tr1)]
    tr1_fixed = numpy.zeros_like(tr1)
    available_ind = list(range(len(tr1)))
    tr_diff = numpy.abs(tr2.T - tr1_nonna[:,None])
    for x in tr1_nonna:
        ind = numpy.argmin(numpy.abs(x - tr2)[numpy.array(available_ind)])
        tr1_fixed[available_ind[ind]] = x
        available_ind.remove(available_ind[ind]) 
    tr1_fixed[numpy.array(available_ind)]=tr2[numpy.array(available_ind)]
    return tr1_fixed

def init_all_trials(ys_l=None,ntrials=10,K=2,sigma_default=5,random_init=False):
    '''
    when the number of peaks a trial contains is fewer than K, still need to align the existing fields using the adjacent trials, fill the nan with the adjacent trial and initialize w to 0
    if adjacent trial also nan, keep searching; if doens't work just pick one from the trial that does not have nan; if no trials have enough fields, return
    '''
    pars={}
    nfields=K
    if ys_l is not None:
        ntrials =nt= ys_l.shape[1]
    if random_init:
        init_max_w = 100.
        init_max_sigma = 50.
        init_max_b = 5.
        init_max_mu = 100.
        pars['logws']=inv_softplus(numpy.random.rand(ntrials,nfields) * init_max_w) 
        pars['logb'] = inv_softplus(numpy.random.rand(ntrials,) * init_max_b)  
        pars['logsigmas']=inv_softplus(numpy.random.rand(ntrials,nfields) * init_max_sigma)  
        pars['mus']=numpy.random.rand(ntrials,nfields) * init_max_mu
        return pars
    
    init_locs_l_=numpy.array([init_one_trial(ys,K=K) for ys in ys_l.T]) # ntrials x K
    init_locs_l = numpy.sort(init_locs_l_,axis=1)
    trial_index_containing_nan = np.nonzero(numpy.isnan(init_locs_l).sum(axis=1)>0)[0]
    trial_index_no_nan = np.nonzero(numpy.isnan(init_locs_l).sum(axis=1)==0)[0]
    if len(trial_index_no_nan)==0:
        print('Wrong K! All trials have not enough fields')
        return
    for ti in trial_index_containing_nan:
        ti_cmp= ti-1
        while ti_cmp in trial_index_containing_nan:
            ti_cmp = ti_cmp - 1
        if ti_cmp < 0: # if the trial is not the first, use the previous trial as the reference
            ti_cmp = trial_index_no_nan[0]
        init_locs_l[ti] = realign_trial_with_nan(init_locs_l[ti],init_locs_l[ti_cmp])
    
    mus_l = np.array(init_locs_l,dtype=float)
    logsigmas_l=logsigma0 = inv_softplus(np.ones((ntrials,nfields)) * sigma_default)
    sigma0 =softplus(logsigma0)

    peak_y_l = ys_l[init_locs_l[:,0].astype(int),np.arange(nt)] # using init_locs_l_ here, such that the peak will not be 0; otherwise the index here could be interpolated which used to be nan
    logws_l=w0 =   inv_softplus(np.dot((peak_y_l[:,None] / scipy.stats.norm.pdf(0,loc=0,scale=sigma0[0,0])), np.ones(nfields)[None,:]))
    # import pdb
    # pdb.set_trace()
    
    b_l = ys_l.mean(axis=0)
    logb_l = inv_softplus(b_l)
    S = np.ones_like(ys_l) * 0
    
    pars['logws']=logws_l 
    pars['mus']=mus_l
    pars['logsigmas']=logsigmas_l 
    # pars['b']=b_l 
    pars['logb'] = logb_l
    pars['S']=S 

    
    
    return pars

def init_both_trial_types(fr_map_one_unit_both_trials_dict,trial_ind_dict,bin_to_lin=None,random_init=False):
    '''
    initialize both left and right trials based on the firing maps for one unit on these trials. 
    
    fr_map_one_unit_both_trials_dict: {trial_type: nbins x ntrials}
    trial_ind_dict: {trial_type:[inds for trials within that type]}
    bin_to_lin: (nbins,) if given, map the mus from indices among bins to physical distance in cm.
    
    output:
    # pars_dict: {trial_type: {name: ntrials x nfields} (except for b, ntrials x 1)}
    pars_l: ntrials(both together) x nfields 
    
    '''
    pars_dict = {}
    ntrials_total = numpy.sum([len(val) for k,val in trial_ind_dict.items()])
    
    for trial_type, fm in fr_map_one_unit_both_trials_dict.items():
        pars_dict[trial_type]=init_all_trials(fm,random_init=random_init)
    
    if bin_to_lin is not None:
        for trial_type in pars_dict.keys():
            pars_dict[trial_type]['mus'] = bin_to_lin[pars_dict[trial_type]['mus'].astype(int)]

    pars = {}
    
    params_keys = list(pars_dict[0].keys())
    if 'S' in params_keys: # remove 'S' which is used in the firing map formulation
        params_keys.remove('S')
    for key in params_keys:
        if len(pars_dict[0][key].shape) == 2:
            pars[key] = numpy.zeros((ntrials_total,pars_dict[0][key].shape[1]))
        else: # for b, only 1d
            pars[key] = numpy.zeros(ntrials_total)
        for trial_type in pars_dict.keys():
            inds = trial_ind_dict[trial_type].astype(int)
            
            pars[key][inds] = pars_dict[trial_type][key]

    return pars

# initializers for regressors and regularization hyperparameters
def get_regressor(regressors_={}):
    regressors={}
    xs=np.arange(100,dtype=float)
    regressors['xs']=xs
    regressors.update(regressors_)
    return regressors

def get_reg_pars(reg_pars_={}):
    reg_pars={'S_l1':100000000.}
    reg_pars['g_mu'] = 200.
    reg_pars['g_sigma'] = 1000.#10000000.
    reg_pars['g_w'] = 10000000.#1.#
    reg_pars['g_b'] = 10000000.
    reg_pars['sigma_thresh'] = 1
    reg_pars['g_sigma_thresh'] = 1000. # want a high penalty for too small sigma such that all sigma will be above 1 bin
    reg_pars['nfields']=2
    reg_pars['g_order'] = 1000. # high penalty for order violation of the fields; 
    reg_pars['g_sigma_shrinkage'] = 100. # weak penalty for growing sigma
    reg_pars.update(reg_pars_) # update default reg_pars
    return reg_pars


        

    

def train_adam(func,regressors,pars,ys_l,reg_pars,*args,loss_type='mse',reg_type=None,lr=0.1,niters=100,mask=None):
# def train_adam(func,*args,argnums=1,lr=0.1,niters=100,**kwargs):
    opt_init,opt_update,get_params=jax_opt.adam(lr)
    loss_l=[]
    @jit
    def train_step(step_i,opt_state):
        params=get_params(opt_state)
        loss,grads = value_and_grad(func,argnums=1)(regressors,params,ys_l,*args,reg_pars=reg_pars,loss_type=loss_type,reg_type=reg_type,mask=mask) # notice the params here
        # loss,grads = value_and_grad(func,argnums=argnums)(*args,**kwargs) 
        return loss, opt_update(step_i, grads, opt_state)
    # opt_state=opt_init(args[argnums])
    opt_state=opt_init(pars)
    for ii in range(niters):
        loss,opt_state = train_step(ii,opt_state)
        loss_l.append(loss)
    return get_params(opt_state), np.array(loss_l)

#
# %%
# test gm_func_by_trial
def main():
    log_folder = os.path.join('.', 'slurm_jobs', 'logs/%j')
    ex = submitit.AutoExecutor(folder=log_folder)
    if ex.cluster == 'slurm':
        print('submitit executor will schedule jobs on slurm!')
    else:
        print(f"!!! Slurm executable `srun` not found. Will execute jobs on '{ex.cluster}'")
    ex.update_parameters(
        slurm_job_name='gm_fit',
        nodes=1,
        slurm_partition="ccn", 
        cpus_per_task=1,
        mem_gb=4,  # 32gb for train mode, 8gb for eval mode
        timeout_min=60
    )

    fr_map_ = pickle.load(open(os.path.join(py_data_dir,'fr_map.p'),'rb'))
    fr_map=fr_map_['fr_map']
    uid_l = fr_map[0].index
    
    # actual job
    jobs = []
    ch_l= [0,1]
    ind_l = range(len(fr_map[0]))
    with ex.batch():
        for ch in ch_l:
            for ind in ind_l:
                job = ex.submit(fit_main,ch,ind)
                jobs.append(job)
    print("all jobs submitted")

    idx = 0
    for f in ch_l:
        for b in ind_l:
            print(f'Job {jobs[idx].job_id} === ch: {f}, ind: {b}')
            idx += 1
    

# plt.plot(loss_l)
# plt.show()

def fit_main(ch,ind,nfields=1,dosave=True,reg_type = 'quad_variation',reg_pars_={}):
    # prep saving
    tosavedir_root = os.path.join(py_data_dir,'gm_fit')
    if not os.path.exists(tosavedir_root):
        os.mkdir(tosavedir_root)
        print(f'{tosavedir_root} made!')

    # prep hyperparams
    regressors={}
    xs=np.arange(100,dtype=float)
    regressors['xs']=xs
    reg_pars={'S_l1':100000000.}
    # gr_p=grad(loss_total,1)(regressors,pars,ys_l,reg_pars,loss_type="mse",reg_type=None)

    # reg_type = 'quad_variation'#'diag'
    reg_pars['g_mu'] = 0.1#1000000000.
    reg_pars['g_sigma'] = 10.#10000000.
    reg_pars['g_w'] = 10000000.#1.#
    reg_pars['g_b'] = 10000000.
    reg_pars.update(reg_pars_) # update default reg_pars

    # prep saving, with hyperparam info
    tosavedir_sub = os.path.join(tosavedir_root,f'nfields{nfields}_reg_{reg_type}_mu_sigma_noS')
    if not os.path.exists(tosavedir_sub):
        os.mkdir(tosavedir_sub)
        print(f'{tosavedir_sub} made!')

    fr_map_ = pickle.load(open(os.path.join(py_data_dir,'fr_map.p'),'rb'))
    fr_map_trial=fr_map_['fr_map_trial']
    ys_l=fr_cell = fr_map_trial[ch][ind]
    # nfields=1
    pars = init_all_trials(ys_l,K=nfields)
    pars_learned,loss_l=train_adam(loss_total,regressors,pars,ys_l,reg_pars,loss_type='mse',reg_type=reg_type,lr=0.05,niters=4000)
    pars_learned['ws'] = softplus(pars_learned['logws'])
    pars_learned['sigmas'] = softplus(pars_learned['logsigmas'])
    # pars_learned_all[ch][uid_l[ind]] = pars_learned
    tosave = {'reg_type':reg_type,'reg_pars':reg_pars,'sess_name':sess_name,'py_data_dir':py_data_dir,'pars':pars_learned,'loss':loss_l}
    tosave_fn = os.path.join(tosavedir_sub,f'ch_{ch}_ind_{ind}.p')
    if dosave:
        pickle.dump(tosave,open(tosave_fn,'wb'))

#%%
def fit(ys_l,regressors_={},reg_type = 'quad_variation',reg_pars_={},nfields=1,mask=None,lr=0.05,loss_type='mse',niters=4000):
    '''
    more standard function for fitting: 
    ys_l: npos x ntrial, the target
    '''
    if mask is None:
        mask = np.ones_like(ys_l)

    # prep hyperparams
    regressors = get_regressor(regressors_=regressors_)
    reg_pars = get_reg_pars(reg_pars_=reg_pars_)
    
    pars = init_all_trials(ys_l,K=nfields)
    if pars is None: # in case when nfields is not enough for deterministic init
        pars = init_all_trials(ys_l,K=2,sigma_default=5,random_init=True) 

    pars_learned,loss_l=train_adam(loss_total,regressors,pars,ys_l,reg_pars,loss_type=loss_type,reg_type=reg_type,lr=lr,niters=niters,mask=mask)
    pars_learned['ws'] = softplus(pars_learned['logws'])
    pars_learned['sigmas'] = softplus(pars_learned['logsigmas'])
    pars_learned['peaks'] = pars_learned['ws'] * scipy.stats.norm.pdf(0,loc=0,scale=pars_learned['sigmas'])
    pars_learned['b'] = softplus(pars_learned['logb'])

    return pars_learned,loss_l

# ys_l = fr_map_trial[0][0]
# pars_learned = fit(ys_l,regressors_={},reg_type = 'quad_variation',reg_pars_={},nfields=1,mask=None)


#%%
# fit_main(0,2,dosave=False)

# vmap(fit_main)(in_axes=(None,0,None,None,None,None))
# %%
# res=pickle.load(open("/mnt/home/szheng/ceph/ad/roman_data/e15/e15_13f1/e15_13f1_220117/py_data/gm_fit/nfields1_reg_quad_variation_mu_sigma_noS\
# /ch_0_ind_2.p",'rb'))
# %%

if __name__=='__main__':
    main()
    # fit_main(0,2,nfields=1)
# %%
