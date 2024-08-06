import numpy as np
import scipy
import matplotlib.pyplot as plt
import sys,os,copy,pdb,importlib
from importlib import reload
import pandas as pd
sys.path.append('/mnt/home/szheng/projects/util_code')
import data_prep_pyn as dpp     
from sklearn.cluster import KMeans
import scipy.spatial 
from scipy.spatial.distance import pdist, squareform
import cluster_analysis as ca
import seaborn as sns


def plot_gap_test(n_clusters_l,gap_k, s_k, test_stats,fig=None,ax=None):
    fig,ax=plt.subplots()
    ax.errorbar(n_clusters_l, gap_k, yerr=s_k,label='gap stats')
    ax.set(xticks=n_clusters_l,ylabel='gap stats',xlabel='num. clusters')
    ax2 = ax.twinx()
    ax2.plot(n_clusters_l[:-1], test_stats, marker='o',color='C1',label='difference')
    ax2.axhline(0,color='C1',linestyle=':',linewidth=3)
    ax2.set(xticks=n_clusters_l,ylabel='difference')

    return fig, ax, ax2


