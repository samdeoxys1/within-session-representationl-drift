import scipy
from scipy.stats import rankdata
import numpy
import jax.numpy as np
import matplotlib.pyplot
from jax import vmap
from jax.tree_util import tree_map
import pandas as pd

from math_functions import *
import gm as gm
import gm_cv as gc
import plot_helper as ph
import gm_glm_bayesian as glm
from importlib import reload
reload(glm)
import pickle,os,sys
from functools import partial
import submitit


def get_train_test_mask_from_space(regressors,ratio_consec_bins_to_mask=0.2):
    pos_discrete=regressors['position'].astype(int)
    npos = len(numpy.unique(pos_discrete))
    ntrials = len(numpy.unique(regressors['trial_inds_int']))
    reindex= rankdata(regressors['trial_inds_int'],method='dense')-1
    
    ma = gc.get_train_test_mask(npos,ntrials,ratio_consec_bins_to_mask=ratio_consec_bins_to_mask)
    mask= ma[pos_discrete, reindex]
    return mask

# def get_nfields_mask_dict(nfields_max):
#     nfields_mask_dict={}
#     for i in range(nfields_max):
#         nfields_mask_dict[i]=np.zeros(nfields_max).at[:i].set(1)
#     return nfields_mask_dict


# @partial(jax.jit, static_argnames=('nfields_max',))
def train_and_test(key,target_smthed,regressors,nfields_mask,reg_pars_={},reg_type = 0,mask=None,loss_type=0,niters_l=[4000],lr_l=[0.05],nfields_max=5,smthwin_l=[30]):

    '''
    given a set of hyperparam, and mask: train on the train set get par and loss, and get test loss
    '''
    
    if mask is None:
        mask = np.ones_like(target_smthed[0])
    # key = jax.random.PRNGKey(key_int)
    pars_trans_init=glm.random_init_jax(key,regressors,nfields_max=nfields_max,uncentered=True)
    # prepare regressors and hyperparams
    # train
    
    # pars_learned_trans = glm.train_adam(regressors,pars_trans_init,target,reg_pars_,nfields_mask,reg_type = reg_type,mask=mask,loss_type=loss_type,niters=niters,lr=lr)
    pars_learned_trans = glm.train_adam_schedule(regressors,pars_trans_init,target_smthed,reg_pars_,nfields_mask,reg_type = reg_type,mask=mask,niters_l=niters_l,lr_l=lr_l,smthwin_l=smthwin_l)

    # test; by convention the last smoothed target should be the true target
    test_loss = glm.negative_logpdf_no_reg(regressors,pars_learned_trans,target_smthed[-1],nfields_mask,loss_type=loss_type,mask=np.logical_not(mask))
    # return pars_learned_trans, loss_l, test_loss
    return pars_learned_trans, test_loss


# def cv_one_neuron_one_mask(key,mask,target,regressors,reg_pars_l,fit_kwargs=dict(reg_type = 'gaussian_logprior_laplacian',lr=0.05,loss_type='poisson',niters=4000),nfields_max=5):
#     # mask = get_train_test_mask_from_space(regressors,ratio_consec_bins_to_mask=cv_kwargs['mask_ratio'])
#     # nfields_l = np.array([rp_['nfields'] for rp_ in reg_pars_l] )
#     nfields_mask_l=reg_pars_l['nfields_mask']
#     key_l = jax.random.split(key,num=nfields_mask_l.shape[0])
#     # pars_trans_init_l = vmap(glm.random_init_jax,in_axes=(0,None,None,0))(key_l,regressors,ntrials,nfields_l)
#     train_and_test_wrapper=lambda key,rp_,nfields_mask,mask: train_and_test(key,target,regressors,nfields_mask,reg_pars_=rp_,mask=mask,**fit_kwargs)
#     pars_learned_l,_,test_loss_l= vmap(train_and_test_wrapper,in_axes=(0,0,0,None))(key_l,reg_pars_l,nfields_mask_l,mask)
#     loss_noreg_wrapper=lambda par,nfields_mask:glm.negative_logpdf_no_reg(regressors,par,target,nfields_mask,mask=mask,loss_type=fit_kwargs['loss_type'])
#     train_loss_l = vmap(loss_noreg_wrapper)(pars_learned_l,nfields_mask_l)
#     return pars_learned_l, train_loss_l, test_loss_l

def expand_cv_param_grid_for_vmap(reg_pars,reg_pars_tosweep_kwargs={'g_w':[1.,10.]},nfields_max = 5):
    '''
    use gm.expand_cv_param_grid; that one return a list, not good for vmap; so this one put 
    the to be sweeped dimension into each array in the dict. Also create nfields_mask for all the different nfields 
    so reg_pars or reg_pars_to_sweep_kwargs must have nfields
    '''
    reg_pars_ = gm.get_reg_pars(reg_pars) # add in the default
    reg_pars_l = gc.expand_cv_param_grid(reg_pars_,**reg_pars_tosweep_kwargs)
    reg_pars_name = list(reg_pars_l[0].keys())
    reg_pars_d = {}
    for k in reg_pars_name:
        reg_pars_d[k] = np.array([rp_[k] for rp_ in reg_pars_l])
    reg_pars_d['nfields_mask']=np.array([np.zeros(nfields_max).at[:i].set(1).astype(bool) for i in reg_pars_d['nfields']])
    reg_pars_df =pd.DataFrame(reg_pars_l)
    reg_pars_df['reg_pars_ind'] = reg_pars_df.index
    return reg_pars_d,reg_pars_df

def cv_prep(regressors,reg_pars_tosweep_kwargs_={},reg_pars_={},fit_kwargs_={},cv_kwargs_={},save_kwargs_={},nfields_max=5):
    save_kwargs = {'dir':'','fn':'cv_one_neuron.p'}
    save_kwargs.update(save_kwargs_)
    if save_kwargs_ is not None and save_kwargs_ !={}:
        if not os.path.exists(save_kwargs['dir']):
            os.makedirs(save_kwargs['dir'])
            print(f'{save_kwargs["dir"]} made')
    fn_full = os.path.join(save_kwargs['dir'],save_kwargs['fn'])

    cv_kwargs = {'mask_ratio':0.2,'cv_fold':10}
    cv_kwargs.update(cv_kwargs_)

    # fit_kwargs = dict(reg_type = 'gaussian_logprior_laplacian',lr=0.05,loss_type='poisson',niters=4000)
    # fit_kwargs = dict(reg_type = 0,lr=0.05,loss_type=0,niters=4000,nfields_max=nfields_max)
    fit_kwargs = dict(reg_type = 0,loss_type=0,nfields_max=nfields_max)
    fit_kwargs.update(fit_kwargs_)

    #reg_pars_tosweep_kwargs = dict(g_w=[1000000],nfields=[2,3,4],g_mu=[0.1,1,100,10000],g_sigma=[1,100,10000])
    reg_pars_tosweep_kwargs = {}
    reg_pars_tosweep_kwargs.update(reg_pars_tosweep_kwargs_)

    reg_pars_ = gm.get_reg_pars(reg_pars_) # add in the default
    # reg_pars_l = gc.expand_cv_param_grid(reg_pars_,**reg_pars_tosweep_kwargs)
    reg_pars_l,reg_pars_df = expand_cv_param_grid_for_vmap(reg_pars_,reg_pars_tosweep_kwargs=reg_pars_tosweep_kwargs,nfields_max=fit_kwargs['nfields_max'])  
    # n_reg_pars_to_sweep =reg_pars_l['nfields_mask'].shape[0]
    reg_pars_name = list(reg_pars_l.keys())

    mask_l = np.array([get_train_test_mask_from_space(regressors,ratio_consec_bins_to_mask=cv_kwargs['mask_ratio']) for _ in range(cv_kwargs['cv_fold'])])
    
    return fn_full, cv_kwargs, fit_kwargs, reg_pars_l, reg_pars_name, mask_l, reg_pars_df

def vmap_grid_to_map_grid(reg_pars_l,mask_l,cv_kwargs):
    '''
    use the results from cv_prep; repeat the pytrees in a way like itertools.product
    such that each zip of the arguments can be fed into one cv func, and no need to wrap the cv func twice
    '''
    reg_pars_l_repeated=tree_map(lambda x:np.tile(x,cv_kwargs['cv_fold']) if len(x.shape)==1 else np.tile(x,(cv_kwargs['cv_fold'],1)),reg_pars_l)
    n_regpars = len(reg_pars_l['g_w'])
    mask_l_repeated = np.repeat(mask_l,n_regpars,axis=0)
    return reg_pars_l_repeated, mask_l_repeated


def cv_all_neurons(target_all_neurons,regressors,reg_pars_tosweep_kwargs_={},reg_pars_={},fit_kwargs_={},cv_kwargs_={},dosave=False,save_kwargs_={},forcereload=False,key_int=0,col_names=None):
    key = jax.random.PRNGKey(key_int)
    n_neurons = target_all_neurons.shape[1]
    
    
    fn_full, cv_kwargs, fit_kwargs, reg_pars_l, reg_pars_name, mask_l, reg_pars_df=cv_prep(regressors,reg_pars_tosweep_kwargs_=reg_pars_tosweep_kwargs_,reg_pars_=reg_pars_,fit_kwargs_=fit_kwargs_,cv_kwargs_=cv_kwargs_,save_kwargs_=save_kwargs_)
    reg_pars_l_repeated, mask_l_repeated =vmap_grid_to_map_grid(reg_pars_l,mask_l,cv_kwargs)

    smthwin_l = fit_kwargs['smthwin_l']
    target_all_neurons_smthed = glm.smooth_target(target_all_neurons,smthwin_l)

    if os.path.exists(fn_full) and (not forcereload):
        print(f'{fn_full} already exists')
        res=pickle.load(open(fn_full,'rb'))
        return res

    # prepping the for result dataframe
    if col_names is None:
        cols = numpy.arange(n_neurons)
    else:
        cols = col_names

    cv_fold = cv_kwargs_['cv_fold']
    lr_l = fit_kwargs['lr_l']
    niters_l = fit_kwargs['niters_l']
    nfields_max = fit_kwargs['nfields_max']
    train_and_test_all_neurons_wrapper = lambda args:train_and_test_all_neurons(key,target_all_neurons_smthed,regressors,args[0],reg_pars_=args[1],reg_type = 0,mask=args[2],loss_type=0,niters_l=niters_l,lr_l=lr_l,smthwin_l=smthwin_l,nfields_max=nfields_max)

    pars_learned_l,test_loss_l = jax.lax.map(train_and_test_all_neurons_wrapper,(reg_pars_l_repeated['nfields_mask'],reg_pars_l_repeated,mask_l_repeated))
    reg_pars_df_full = pd.concat([reg_pars_df]*cv_fold).reset_index(drop=True)
    
    test_loss_df = pd.DataFrame(test_loss_l,columns=cols)
    test_loss_all_df = pd.concat([reg_pars_df_full,test_loss_df],axis=1)
    # pars_learned = res[0]

    # res = {'loss':test_loss_all_df,'pars_learned':pars_learned,'mask':mask_l}
    res = {'loss':test_loss_all_df,'pars_learned':pars_learned_l,'mask':mask_l}
    if dosave:
        
        pickle.dump(res,open(fn_full,'wb'))
        print(f'saved at {fn_full}')
    return res





# def train_and_test_all_neurons(key,target_all_neurons,regressors,nfields_mask,reg_pars_={},reg_type = 0,mask=None,loss_type=0,niters=4000,lr=0.05,nfields_max=5):
def train_and_test_all_neurons(key,target_all_neurons_smthed,regressors,nfields_mask,reg_pars_={},reg_type = 0,mask=None,loss_type=0,niters_l=[4000],lr_l=[0.05],smthwin_l=[30],nfields_max=5):

    '''
    given a set of hyperparam, and mask: train on the train set get par and loss, and get test loss
    target_all_neurons_smthed: nsmthwin x ntimes x nneurons
    '''
    
    if mask is None:
        mask = np.ones_like(target_all_neurons_smthed[0].shape[0]) # notice the target_all_neurons here is all neurons!!!
    # key = jax.random.PRNGKey(key_int)
    n_neurons = target_all_neurons_smthed[0].shape[1]
    key_l=jax.random.split(key,n_neurons)
    pars_trans_init_allneurons=glm.random_init_jax_allneurons(key_l,regressors,nfields_max)
#     pars_trans_init=vmap(glm.random_init_jax(key_l,regressors,nfields_max=nfields_max,uncentered=True)
    # prepare regressors and hyperparams
    # train
    # pars_learned_trans,loss_l = gm.train_adam(glm.negative_logpdf,regressors,pars_trans_init,target,reg_pars_,nfields_mask,reg_type = reg_type,mask=mask,loss_type=loss_type,niters=niters,lr=lr)
    # pars_learned_trans = glm.train_adam_allneurons_same_regpars(regressors,pars_trans_init_allneurons,target_all_neurons,reg_pars_,nfields_mask,reg_type,mask,loss_type,niters,lr)
    pars_learned_trans = glm.train_adam_allneurons_same_regpars_schedule(regressors,pars_trans_init_allneurons,target_all_neurons_smthed,reg_pars_,nfields_mask,reg_type,lr_l,niters_l,smthwin_l,mask)

    # test
    test_loss = vmap(glm.negative_logpdf_no_reg,in_axes=(None,0,1,None,None,None))(regressors,pars_learned_trans,target_all_neurons_smthed[-1],nfields_mask,np.logical_not(mask),loss_type)
    # return pars_learned_trans, loss_l, test_loss
    return pars_learned_trans, test_loss

def main(py_data_dir,test_mode=1):

    log_folder = os.path.join('.', 'slurm_jobs','logs/glm_cv/%j')
    ex = submitit.AutoExecutor(folder=log_folder)
    if ex.cluster == 'slurm':
        print('submitit executor will schedule jobs on slurm!')
    else:
        print(f"!!! Slurm executable `srun` not found. Will execute jobs on '{ex.cluster}'")
    if test_mode:
            ex.update_parameters(
            slurm_job_name='glm_cv',
            nodes=1,
            slurm_partition="genx", 
            cpus_per_task=4,
            mem_gb=64,  # 32gb for train mode, 8gb for eval mode
            timeout_min=60
        )
    else:

        ex.update_parameters(
            slurm_job_name='glm_cv',
            nodes=1,
            slurm_partition="gpu", 
            cpus_per_task=4,
            gpus_per_node=1,
            mem_gb=64,  # 32gb for train mode, 8gb for eval mode
            timeout_min=2880
        )

    # load data
    # sess_name="e15_13f1_220117"
    # py_data_dir = "/mnt/home/szheng/ceph/ad/roman_data/e15/e15_13f1/e15_13f1_220117/py_data"
    fr_ = pickle.load(open(os.path.join(py_data_dir,'fr.p'),'rb'))
    pos_bins = fr_['pos_bins']['lin']
    bin_to_lin=numpy.concatenate([pos_bins[:-1,None],pos_bins[1:,None]],axis=1).mean(axis=1)

    fr = fr_['df']['pyr']
    fr_map_ = pickle.load(open(os.path.join(py_data_dir,'fr_map.p'),'rb'))['fr_map_trial']
    cell_cols = fr_['cell_cols']
    regressors=glm.get_regressors(fr)
    target_spk_allneurons = np.array(fr[cell_cols['pyr']].values)

    # both trial types
    ch_l= [0,1]
    regressors_one_trial_d, target_spk_one_trial_d = {}, {}
    

    # different hyperparam range for test_mode or not
    if test_mode:
        cv_kwargs_={'cv_fold':1}
        reg_pars_tosweep_kwargs_={'g_w':[100.],'g_b':[100.],'nfields':[1,2]}      
        nfields_max = numpy.max(reg_pars_tosweep_kwargs_['nfields'])
        # fit_kwargs_=dict(lr=0.05,niters=2,nfields_max=nfields_max)
        fit_kwargs_=dict(niters_l = np.array([1,1]),smthwin_l = np.array([30,0.1]),lr_l = np.array([0.05,0.02]),nfields_max=nfields_max)
        
    else:
        cv_kwargs_= {'cv_fold':10}
        reg_pars_tosweep_kwargs_={'g_w':[0.0001,0.001,0.01,0.1,100.,10000],'g_b':[0.001,0.1,100.,10000],'nfields':[1,2,3,4]}
        nfields_max = numpy.max(reg_pars_tosweep_kwargs_['nfields'])
        # fit_kwargs_=dict(lr=0.05,niters=4000,nfields_max=nfields_max)
        fit_kwargs_=dict(niters_l = np.array([2000,2000]),smthwin_l = np.array([30,0.1]),lr_l = np.array([0.05,0.02]),nfields_max=nfields_max)

    
    reg_pars_ = gm.get_reg_pars({'g_w':1000.,'g_b':10000.,'g_sigma_thresh':100000.,'sigma_thresh':3,'g_sigma_shrinkage':0.})

    jobs = []
    for ch in ch_l:
        regressors_one_trial_d[ch], target_spk_one_trial_d[ch]=glm.subselect_regressors(regressors, target_spk_allneurons, trial_type=ch)
        target_all_neurons = target_spk_one_trial_d[ch]
        regressors_ = regressors_one_trial_d[ch]
        regressors_['ntrials'] = len(numpy.unique(regressors_['trial_inds_int']))
        save_kwargs_ = {'dir':os.path.join(py_data_dir,'glm_deterministic'),'fn':f'cv_trial{ch}_allneuron_new.p'}
        fn_full, cv_kwargs, fit_kwargs, reg_pars_l, reg_pars_name, mask_l, reg_pars_df=cv_prep(regressors_,reg_pars_tosweep_kwargs_=reg_pars_tosweep_kwargs_,reg_pars_=reg_pars_,fit_kwargs_=fit_kwargs_,cv_kwargs_=cv_kwargs_,save_kwargs_={},nfields_max=nfields_max)
        nfields_mask_l = reg_pars_l['nfields_mask']
        dosave = True
        forcereload = True
        with ex.batch():
            job = ex.submit(cv_all_neurons,target_all_neurons,regressors_,reg_pars_tosweep_kwargs_=reg_pars_tosweep_kwargs_,reg_pars_=reg_pars_,fit_kwargs_=fit_kwargs_,cv_kwargs_=cv_kwargs_,dosave=dosave,save_kwargs_=save_kwargs_,forcereload=forcereload,key_int=0,col_names=cell_cols['pyr'])
            jobs.append(job)
        print("all jobs submitted")
            # %time res=cv_all_neurons(target_all_neurons,regressors_,reg_pars_tosweep_kwargs_=reg_pars_tosweep_kwargs_,reg_pars_=reg_pars_,fit_kwargs_=fit_kwargs_,cv_kwargs_=cv_kwargs_,dosave=dosave,save_kwargs_=save_kwargs_,forcereload=forcereload,key_int=0,col_names=cell_cols['pyr'])

if __name__=='__main__':
    py_data_dir = "/mnt/home/szheng/ceph/ad/roman_data/e15/e15_13f1/e15_13f1_220117/py_data"
    main(py_data_dir,int(sys.argv[1]))