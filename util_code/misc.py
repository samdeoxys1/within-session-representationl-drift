import os,sys,pickle,glob
import pandas as pd
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import scipy


# def get_or_create_subdir(root,*args,doclear=False):
def get_or_create_subdir(*args,doclear=False):
    # if len(args)==0:
    #     dir = root
    # else:
    #     if root !='':
    #         dir = os.path.join(root,*args)
    #     else:
    #         dir = os.path.join(*args)
    # print(args)
    dir = os.path.join(*args)
    print(dir)
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f'{dir} made!',flush=True)
    if doclear:
        files = glob.glob(os.path.join(dir,'*'))
        for f in files:
            os.remove(f)
    return dir

def get_res(dir,fn,force_reload):
    fn_full = os.path.join(dir,fn)
    if os.path.exists(fn_full) and not force_reload:
        res = pickle.load(open(fn_full,'rb'))
        print(f'{fn_full} exists; loading---')
    else:
        res = None
    return fn_full, res

def save_res(fn_full,res,dosave=True):
    if dosave:
        pickle.dump(res,open(fn_full,'wb'))
        print(f'saved at {fn_full}')

def save_fig_plotly(fn_full,fig,do_html=True,do_png=True,do_svg=False):
    if do_html:
        fig.write_html(fn_full+'.html')
    if do_png:
        fig.write_image(fn_full+'.png')
    if do_svg:
        fig.write_image(fn_full+'.svg')
    print(f'saved at {fn_full}')


# set operations
def get_intersect_difference(ser1,ser2):
    union = pd.Series(np.union1d(ser1, ser2))
    # intersection of the series
    intersect = pd.Series(np.intersect1d(ser1, ser2))
    
    # uncommon elements in both the series 
    notcommonseries = union[~union.isin(intersect)]
    return intersect, notcommonseries

# parallel process lambda function
_func = None
def worker_init(func):
  global _func
  _func = func

def worker(x):
  return _func(x)

def xmap(func, iterable, processes=None):
  with Pool(processes, initializer=worker_init, initargs=(func,)) as p:
    return p.map(worker, iterable)

# dictionary query
def get_nth_key(dictionary, n=0):
    if n < 0:
        n += len(dictionary)
    for i, key in enumerate(dictionary.keys()):
        if i == n:
            return key
    raise IndexError("dictionary index out of range") 

def get_nth_val(dictionary, n=0):
    key = get_nth_key(dictionary,n=n)
    return dictionary[key]


# using statsmodels; consider creating a seperate file
import statsmodels.formula.api as smf

def summarize_statsmodels(model,drop_intercept=True):
    results_df = pd.DataFrame({
    'Coefficients': model.params,
    'Standard Errors': model.bse,
    't-values':model.tvalues,
    'P-values': model.pvalues,
    'CI_low': model.conf_int()[0],
    'CI_high': model.conf_int()[1]
        
    })
    if drop_intercept:
        results_df = results_df.drop('Intercept')
    return results_df

def print_stats_onevar(xx):
    '''
    for one var, report N, effect size=mean / std, pval from wilcoxon
    '''
    N = xx.shape[0]
    efsz = xx.mean()/xx.std()
    _,pval=scipy.stats.wilcoxon(xx)
    print(f'N = {N}; Effect size = {efsz}; wilcoxon p={pval}')
    return efsz, pval, N

def print_stats_twovar(xx,key_col,val_col,order=None):
    '''
    for one var, report N, effect size=mean / std, pval from wilcoxon
    '''
    N = xx.shape[0]

    if order is None:
        key_l =xx[key_col].unique()
    else:
        key_l = order
    val_l = []
    mean_l = []
    for k in key_l:
        val_one = xx.loc[xx[key_col]==k,val_col]
        val_l.append(val_one)
        mean_l.append(val_one.mean())
    _,pval=scipy.stats.ranksums(*val_l)

    
    mean_diff = mean_l[key_l[0]] - mean_l[key_l[1]]
    efsz = mean_diff / xx[val_col].std()
    
    print(f'{key_l}; N = {N}; Effect size = {efsz}; wilcoxon rank sum p={pval}\n')
    return efsz, pval, N


###### delete files under wildcard subfolders
import os
import shutil
import glob
import tqdm

def delete_contents(directory):
    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"The path {directory} does not exist.")
        return

    # Remove all contents of the directory
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
            print(f"Deleted file: {item_path}")
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
            print(f"Deleted directory: {item_path}")


def delete_contents_all(
    base_path = "/mnt/home/szheng/ceph/ad/ipshita_data_full/auditory/Final",
    pattern = "*/Final/*/py_data/theta_decoding_lickLoc_y",
        ):    
    '''
    give base_path,
    and pattern for subfolders where contents should be deleted
    '''

    # Use glob to find all directories that match the pattern
    target_dirs = glob.glob(os.path.join(base_path, pattern))


    # Delete contents of each directory that matches the pattern
    for directory in tqdm.tqdm(target_dirs):
        delete_contents(directory)

