import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import sys,os,copy,pdb,importlib,pickle
from importlib import reload
import pandas as pd
import plotly.express as px
import matplotlib
import plotly.graph_objects as go
import plotly.io as pio
import misc

from plotly.subplots import make_subplots

def plot_save_multi_heatmap(tuning_l, 
                            key_l=None, # subplot titles
                            fig_fn = 'plot.html',
                            fig_dir=['./'],
                            title=None,
                            zmin=None,
                            zauto=True
                            ):
    fig_dir_ = misc.get_or_create_subdir(*fig_dir)
    fig_fn_full = os.path.join(fig_dir_,fig_fn)

    nplots = tuning_l.shape[0] # first dimension as the nplots
    ncols =4
    nrows = nplots // ncols + 1
    # key_l_str = [str(k) for k in key_l]
    key_l_str = key_l
    fig = make_subplots(rows=nrows, cols=ncols,subplot_titles=key_l_str)
    cbar_loc_r = np.linspace(0.25,1,nrows)
    cbar_loc_c = np.linspace(0.25,1,ncols)
    if zmin is not None:
        zauto=False
    for ii,tuning in enumerate(tuning_l):
        r,c = ii//ncols, ii%ncols
        val = tuning
        
        heatmap = go.Heatmap(
            z=val,#x=val.columns,y=val.index,
            colorscale='viridis',
            showscale = False,
            colorbar=dict(x=cbar_loc_c[c],y=cbar_loc_r[r],len=0.9/nrows),
            zmin=zmin,zauto=zauto
        )
        fig.add_trace(heatmap,row=r+1,col=c+1)
    
    if title is not None:
        fig.update_layout(title_text=title)
    fig.update_layout(height=300*nrows, width=800)

    
    fig.write_html(fig_fn_full, auto_open=False)



def plot_multi_lines(df,line_cols,
                    xlabel='index',
                    do_zscore=True,
                    ):
    fig = go.Figure()
    if xlabel=='index':
        x = df.index
    else:
        x = df[xlabel]
    for col in line_cols:
        if do_zscore:
            y=scipy.stats.zscore(df[col])
        else:
            y=df[col]
        fig.add_trace(go.Scatter(x=x,y=y,mode='lines',name=col))

    # # Set the extent for the x-axis (if necessary)
    # fig.update_xaxes(range=[df.time.min(), df.time.max()])

    # fig.show("notebook")

    return fig

from plotly.subplots import make_subplots
def plot_multi_heatmap(tuning_df_l,title=None):
    key_l = list(tuning_df_l.keys())
    fig = make_subplots(rows=len(tuning_df_l), cols=1, subplot_titles=key_l)
    for ii,k in enumerate(key_l):
        val = tuning_df_l[k]
        heatmap = go.Heatmap(
            z=val.values,x=val.columns,y=val.index,
            colorscale='viridis',
            coloraxis=f"coloraxis"
        )
        fig.add_trace(heatmap,row=ii+1,col=1)
    
    if title is not None:
        fig.update_layout(title_text=title)

    # fig.show("notebook")
    return fig


def save_to_html(
        fig_l,
        fig_fn = 'plot.html',
        fig_dir=['./'],
        title=''
        ):

    fig_dir_ = misc.get_or_create_subdir(*fig_dir)
    fig_fn_full = os.path.join(fig_dir_,fig_fn)
    
    fig_html_l = []
    combined_html = f"""
        <html>
        <head>
        <title>{title}</title>
        </head>
        <body>
    """
    for ii,fig in enumerate(fig_l):
        fig_html = pio.to_html(fig,full_html=False,include_plotlyjs='cdn')
        fig_html_l.append(fig_html)

        combined_html+=f"""
        <h2>Figure {ii}</h2>
        {fig_html}
        """
    
    combined_html+="""
        </body>
        </html>"""
    
    with open(fig_fn_full, "w") as file:
        file.write(combined_html)
    print(f'fig saved at {fig_fn_full}')