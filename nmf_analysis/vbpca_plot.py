import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['image.cmap'] = 'Greys'

import numpy as np
import pandas as pd

import sys,os,pickle,pdb,copy,itertools
sys.path.append('/mnt/home/szheng/projects/util_code')
import plot_helper as ph
import animate_helper as ah
from scipy.ndimage import gaussian_filter1d


def plot_loss(l_all_allfac,l_recon_allfac,r2_per_fac=None):
    nplots = len(l_all_allfac)
    fig,axs = ph.subplots_wrapper(nplots)
    for ee in range(nplots):
        if isinstance(axs,np.ndarray)>0:
            ax = axs.ravel()[ee]
        else:
            ax=axs
        l_all_l = l_all_allfac[ee]
        l_recon_l = l_recon_allfac[ee]
        # fig,ax=plt.subplots()
        ax.ticklabel_format(useOffset=False)
        title = f'factor {ee} loss'
        if r2_per_fac is not None:
            title += f' r2 = {r2_per_fac[ee]:.02f}'
        ax.set_title(title)
        ax.plot(l_all_l,label='all')
        ax2=ax.twinx()
        ax2.plot(l_recon_l,color='C1',label='recon')
        ax2.ticklabel_format(useOffset=False)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
    plt.tight_layout()
    return fig,ax
