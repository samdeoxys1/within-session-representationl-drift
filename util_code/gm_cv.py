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
import gm
reload(gm)
import pickle,os

sess_name="e15_13f1_220117"
py_data_dir = "/mnt/home/szheng/ceph/ad/roman_data/e15/e15_13f1/e15_13f1_220117/py_data"
REG_PARS_TOSWEEP_KWARGS_={'g_w':[100,1000,10000,100000,1000000],'nfields':[1,2,3,4],'g_mu':[0.1,1,100,1000,10000,100000,1000000],'g_sigma':[0.1,1,100,1000,10000,100000,1000000]}
FIT_KWARGS_={'lr':0.05,'niters':5000}
CV_KWARGS_={'cv_fold':30,'mask_ratio':0.2}
TOSAVEDIR_SUB = 'gm_cv_1'


#%% cross validation
def get_train_test_mask(npos,ntrials,ratio_consec_bins_to_mask=0.1): 
    '''
    consecutive bins for each trial
    return:
    mask:  npos x ntrials; 1 if included in train, 0 if included in test; flip it to get the test_mask
    '''
    mask = numpy.ones((npos,ntrials))
    n_consec_bins_to_mask = int(ratio_consec_bins_to_mask * npos)
    upper_bound = npos - 1 - n_consec_bins_to_mask # -1 because maxind = npos-1
    mask_start_ind_all_trials = numpy.random.randint(0,upper_bound,size=ntrials)
    mask_end_ind_all_trials = mask_start_ind_all_trials + n_consec_bins_to_mask
    for nt in range(ntrials):
        mask[mask_start_ind_all_trials[nt]:mask_end_ind_all_trials[nt],nt]=0

    return mask

# get_train_test_mask(10,100,0.1)
def train_and_test(ys_l,regressors_={},reg_type = 'quad_variation',reg_pars_={},nfields=1,mask=None,loss_type='mse',niters=4000,lr=0.05):
    '''
    given a set of hyperparam, and mask: train on the train set get par and loss, and get test loss
    '''
    if mask is None:
        mask = np.ones_like(ys_l)
    # prepare regressors and hyperparams
    regressors = gm.get_regressor(regressors_=regressors_)
    # train
    pars_learned,loss_l = gm.fit(ys_l,regressors_=regressors,reg_type = reg_type,reg_pars_=reg_pars_,nfields=nfields,mask=mask,loss_type=loss_type,niters=niters,lr=lr)
    # test
    test_loss = gm.loss_no_reg(regressors,pars_learned,ys_l,loss_type=loss_type,mask=np.logical_not(mask))
    return pars_learned, loss_l, test_loss

## for grid search
import itertools
def expand_cv_param_grid(reg_pars_,**kwargs):
    '''
    reg_pars_ is the dict for updating common reg pars
    kwargs: name=[...possible values...] 
    '''
    reg_pars_l = []
    combos=itertools.product(*kwargs.values())
    toupdate =[{k:c for k,c in zip(kwargs.keys(),combo)} for combo in combos]
    for t in toupdate:
        base=copy.copy(reg_pars_)
        base.update(t)
        reg_pars_l.append(base)
    return reg_pars_l


def cv_one_neuron(ys_l,regressors_={},reg_pars_tosweep_kwargs_={},reg_pars_={},fit_kwargs_={},cv_kwargs_={},dosave=False,save_kwargs_={},forcereload=False):
    # init the defaults and update!

    save_kwargs = {'dir':'','fn':'cv_one_neuron.p'}
    save_kwargs.update(save_kwargs_)
    fn_full = os.path.join(save_kwargs['dir'],save_kwargs['fn'])
    if os.path.exists(fn_full) and (not forcereload):
        print(f'{fn_full} already exists')
        res=pickle.load(open(fn_full,'rb'))
        return res

    cv_kwargs = {'mask_ratio':0.2,'cv_fold':10}
    cv_kwargs.update(cv_kwargs_)

    fit_kwargs = dict(reg_type = 'quad_variation',lr=0.05,loss_type='mse',niters=4000)
    fit_kwargs.update(fit_kwargs_)

    regressors = gm.get_regressor(regressors_=regressors_)
    # reg_pars = get_reg_pars(reg_pars_=reg_pars_)
    #reg_pars_tosweep_kwargs = dict(g_w=[1000000],nfields=[2,3,4],g_mu=[0.1,1,100,10000],g_sigma=[1,100,10000])
    reg_pars_tosweep_kwargs = {}
    reg_pars_tosweep_kwargs.update(reg_pars_tosweep_kwargs_)

    reg_pars_ = gm.get_reg_pars(reg_pars_) # add in the default
    reg_pars_l = expand_cv_param_grid(reg_pars_,**reg_pars_tosweep_kwargs)
    reg_pars_name = list(reg_pars_l[0].keys())

    test_loss_l=[]
    train_loss_l=[]
    pars_learned_l=[]
    mask_l =[]

    for ii,cc in enumerate(range(cv_kwargs['cv_fold'])):
        print(cc)
        mask = get_train_test_mask(ys_l.shape[0],ys_l.shape[1],cv_kwargs['mask_ratio'])

        test_loss_l.append([])
        train_loss_l.append([])
        pars_learned_l.append([])
        mask_l.append(mask)
        for rp_ in reg_pars_l:
            print(rp_)
            nfields = rp_['nfields']
            pars_learned,loss_l,test_loss=train_and_test(ys_l,regressors_={},reg_pars_=rp_,nfields=nfields,mask=mask,**fit_kwargs)

            train_loss_no_reg = gm.loss_no_reg(regressors,pars_learned,ys_l,loss_type='mse',mask=mask)

            test_loss_l[ii].append(test_loss)
            train_loss_l[ii].append(train_loss_no_reg)
            pars_learned_l[ii].append(pars_learned)

    reg_pars_and_loss_df=[]
    for ii,(tll,rll) in enumerate(zip(test_loss_l,train_loss_l)):
        reg_pars_df=pd.DataFrame(reg_pars_l)
        reg_pars_df['test_loss'] = numpy.array(tll)
        reg_pars_df['train_loss']=numpy.array(rll)
        reg_pars_df['cv_index'] = ii
        reg_pars_df=reg_pars_df.reset_index().rename(columns={'index':'reg_par_index'})
        reg_pars_and_loss_df.append(reg_pars_df)
    reg_pars_and_loss_df=pd.concat(reg_pars_and_loss_df,ignore_index=True)
    mask_l = numpy.stack(mask_l,axis=0)

    best_reg_par_ind = reg_pars_and_loss_df.groupby('reg_par_index')['test_loss'].mean().idxmin()
    best_reg_par = reg_pars_and_loss_df.query(f'reg_par_index=={best_reg_par_ind}').iloc[0][reg_pars_name]

    best_pars_learned = [pp[best_reg_par_ind] for pp in pars_learned_l]

    res = {'loss':reg_pars_and_loss_df,'pars_learned':pars_learned_l,'mask':mask_l,'best_reg_par':best_reg_par,'best_pars_learned':best_pars_learned}
    if dosave:
        
        pickle.dump(res,open(fn_full,'wb'))
        print(f'saved at {fn_full}')

    return res

def cv_one_neuron_wrapper(ch,ind,testmode=0,forcereload=True):
    # prep saving
    tosavedir_root = os.path.join(py_data_dir,'gm_fit')
    if not os.path.exists(tosavedir_root):
        os.mkdir(tosavedir_root)
        print(f'{tosavedir_root} made!')
    
    tosavedir_sub = TOSAVEDIR_SUB
    # prep saving, with hyperparam info
    tosavedir_sub = os.path.join(tosavedir_root,tosavedir_sub)
    if not os.path.exists(tosavedir_sub):
        os.mkdir(tosavedir_sub)
        print(f'{tosavedir_sub} made!')
    save_kwargs_ = {'dir':tosavedir_sub,'fn':f'ch_{ch}_ind_{ind}.p'}

    fr_map_ = pickle.load(open(os.path.join(py_data_dir,'fr_map.p'),'rb'))
    fr_map_trial=fr_map_['fr_map_trial']
    ys_l = fr_map_trial[ch][ind]

    if testmode:
        res = cv_one_neuron(ys_l,regressors_={},reg_pars_tosweep_kwargs_={'g_sigma':[1,2]},reg_pars_={},fit_kwargs_=FIT_KWARGS_,cv_kwargs_={'cv_fold':3},dosave=True,save_kwargs_=save_kwargs_,forcereload=forcereload)
    else:
        res = cv_one_neuron(ys_l,regressors_={},reg_pars_tosweep_kwargs_=REG_PARS_TOSWEEP_KWARGS_,reg_pars_={},fit_kwargs_=FIT_KWARGS_,cv_kwargs_=CV_KWARGS_,dosave=True,save_kwargs_=save_kwargs_,forcereload=forcereload)



def main():
    log_folder = os.path.join('.', 'slurm_jobs', 'logs/%j')
    ex = submitit.AutoExecutor(folder=log_folder)
    if ex.cluster == 'slurm':
        print('submitit executor will schedule jobs on slurm!')
    else:
        print(f"!!! Slurm executable `srun` not found. Will execute jobs on '{ex.cluster}'")
    ex.update_parameters(
        slurm_job_name='gm_cv',
        nodes=1,
        slurm_partition="genx", 
        cpus_per_task=1,
        mem_gb=4,  # 32gb for train mode, 8gb for eval mode
        timeout_min=2880
    )

    fr_map_ = pickle.load(open(os.path.join(py_data_dir,'fr_map.p'),'rb'))
    fr_map=fr_map_['fr_map']
    uid_l = fr_map[0].index
    
    # actual job
    jobs = []
    ch_l= [0,1]
    ind_l = range(len(fr_map[0])) #python index within unit that has frmap
    with ex.batch():
        for ch in ch_l:
            for ind in ind_l:
                job = ex.submit(cv_one_neuron_wrapper,ch,ind)
                jobs.append(job)
    print("all jobs submitted")

    idx = 0
    for f in ch_l:
        for b in ind_l:
            print(f'Job {jobs[idx].job_id} === ch: {f}, ind: {b}')
            idx += 1

if __name__=='__main__':
    main()