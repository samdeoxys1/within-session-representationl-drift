import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import sys,os,copy,pdb,importlib,pickle
from importlib import reload
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib

# matplotlib.rcParams.update(matplotlib.rcParamsDefault)
# rcdict={'axes.labelsize':20,'axes.titlesize':20}
fs=10
rcdict={'font.size':fs,'axes.labelsize':fs,'axes.titlesize':fs,'xtick.labelsize':fs,'ytick.labelsize':fs,'legend.fontsize':fs}
matplotlib.rcParams.update(rcdict)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['figure.figsize']=(3,2)



def plot_lines_with_state(df,line_cols,state_col='state',xlabel = 'time'):
    '''
    plot line_cols in lines, state_col in heatmap
    '''
    fig = go.Figure()
    for col in line_cols:
        fig.add_trace(go.Scatter(x=df[xlabel],y=scipy.stats.zscore(df[col]),mode='lines',name=col))

    heatmap = go.Heatmap(
        z=[df['state'].values],
        x=df[xlabel],
        yaxis='y3',  # Assign to the secondary y-axis
    #     colorscale=['blue', 'red'],  # Example colorscale, adjust as needed
        showscale=False,  # Hide the color scale
        opacity=0.5  # Match the alpha value
    )

    fig.add_trace(heatmap)

    # Update layout to create a secondary y-axis
    fig.update_layout(
    #     xaxis2=dict(domain=[0, 1], overlaying='x'), 
    #     yaxis2=dict(domain=[0.2,1],side='right'),
        yaxis=dict(domain=[0.2, 1]),  # Adjust domain to make space for heatmap
        yaxis3=dict(domain=[0., 0.2],  # Position for the heatmap
                    showticklabels=False),  # Hide tick labels for the heatmap's y-axis
    )

    # Set the extent for the x-axis (if necessary)
    fig.update_xaxes(range=[df.time.min(), df.time.max()])

    fig.show("notebook")

    return fig

