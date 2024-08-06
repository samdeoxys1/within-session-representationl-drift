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

def get_expected_fr_from_pos(lin_binned,fr_map_run,v=None,fr_map_lowsp=None,v_thresh=1):
    '''
    lin_binned: position bin in time
    fr_map_run,fr_map_lowsp: ratemaps, nneurons x nposbins
    v: speed in time
    v_thresh: thresh for run vs lowsp
    '''

    fr_exp = fr_map_run.loc[:,lin_binned]
    if v is not None:
        low_sp_ma = v < v_thresh
#         import pdb
#         pdb.set_trace()
        fr_exp.loc[:,low_sp_ma.values] = fr_map_lowsp.loc[:,lin_binned.loc[low_sp_ma]]
    return fr_exp
    
    
    