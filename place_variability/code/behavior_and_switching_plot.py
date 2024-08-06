import numpy as np
import scipy
import matplotlib.pyplot as plt
import sys,os,copy,pdb,importlib
from importlib import reload
import pandas as pd
import statsmodels.api
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib 
import re
matplotlib.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.spines.top']=False
plt.rcParams['axes.spines.right']=False

def refine_tick_labels(inds,lin_cat_map=None):
    if lin_cat_map is None:
        lin_cat_map = ['Home','Central','T','Return side','Return pre home']
    refine_label_map = {'n_field':'Num. fields','trial_z':'Trial','speed_z':'Speed','speed_cv_z':'Speed CV','correct':'Correct','correct_prev_seperate':'Last Trial Correct'}
    label_l = []
    for k in inds:
        if 'lin' in k:
            pattern = r"lin.*?\[T\.(.*?)\]"
            matches = re.findall(pattern, k)
            label = lin_cat_map[int(matches[0])]
            label_l.append(label)
        else:
            label=refine_label_map[k]
            label_l.append(label)
    return label_l
            
            
            

def plot_scatter_with_slope(data,x,y,hue,fig=None,ax=None,figsize=(3,2),**kwargs):
    '''
    plot scatter according to hue, a cateogrical variable, and a regression line, assuming 0 intercept
    '''
    if ax is None:
        fig,ax=plt.subplots(figsize=figsize)
    data[hue] = data[hue].astype('category')
    ax=sns.scatterplot(data=data,x=x,y=y,hue=hue,linewidth=0,s=5.)
    
    
    
    
    model_null=smf.ols(formula=f"{y}~{x}-1",data=data)
    res_null = model_null.fit()
    slope_null = res_null.params['n_field']
    xmin,xmax = 0,data[x].max()
    ymin,ymax = 0, slope_null * xmax

    ax.plot([xmin,xmax],[ymin,ymax],color='k')

    ax.legend(bbox_to_anchor=[1.05,1])
    legend_text = kwargs.get('legend_text',None)
    if legend_text is not None:
        leg=ax.get_legend()
        new_labels = legend_text
        for t, l in zip(leg.texts, new_labels):
            t.set_text(l)


    
    
    return fig,ax
