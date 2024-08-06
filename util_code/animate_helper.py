import numpy as np
import scipy
import matplotlib.pyplot as plt
import sys,os,copy,pdb,importlib
from importlib import reload
import pandas as pd

from matplotlib import animation
from matplotlib.animation import FuncAnimation,ArtistAnimation
from mpl_toolkits import mplot3d
import seaborn as sns
import misc

def plot_background_maze(x_all,y_all,hist2d_kws={},fig=None,ax=None):
    if ax is None:
        fig,ax = plt.subplots()
    hist2d_kws_ =dict(bins=(20,20),vmax=10,cmap='Greys')
    hist2d_kws_.update(hist2d_kws)
    # H,xedges,yedges=np.histogram2d(y_all,x_all,**hist2d_kws_)
    # H,xedges,yedges=np.histogram2d(y_all,x_all,bins=hist2d_kws_['bins'])
    H,xedges,yedges=np.histogram2d(x_all,y_all,bins=hist2d_kws_['bins'])
    # xedges_val = np.concatenate([xedges[:-1,None],xedges[1:,None]],axis=1).mean(axis=1).astype(int)
    # yedges_val = np.concatenate([yedges[:-1,None],yedges[1:,None]],axis=1).mean(axis=1).astype(int)
    # H = pd.DataFrame(H,index=yedges_val,columns=xedges_val)
    # _=sns.heatmap(H,cmap='Greys',ax=ax,vmax=10)
    # ax.invert_yaxis()
    ratio = np.mean(np.diff(yedges)) / np.mean(np.diff(xedges))
    # need to transpose H, because x values are along the first dimension, rows, but want it to be on columns
    ax.imshow(H.T,origin='lower',extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]],cmap=hist2d_kws_['cmap'],vmax=hist2d_kws_['vmax'],aspect=ratio)
    ax.set_adjustable('box')
    return fig,ax

def animate_movement_in_maze(data,x_all,y_all,hist2d_kws={},ax_kws={},fig=None,ax=None,
                                    animate_kws={},dosave=False,
                                    do_plot_background_maze=True,
                                    n_tail=100,trial_l=None,speed_l=None,
                                    save_dir='',save_fn='behavior'
                                ):
    '''
    data: 2xT
    '''
    if ax is None:
        fig,ax=plt.subplots()
    ax_kws_ = {}
    ax_kws_['xlim'] = (data[0].min() - np.abs(data[0].min()) * 0.2, data[0].max() + np.abs(data[0].max())*0.2)
    ax_kws_['ylim'] = (data[1].min() - np.abs(data[1].min()) * 0.2, data[1].max() + np.abs(data[1].max())*0.2)
    ax_kws_.update(ax_kws)
    
    if do_plot_background_maze:
        fig,ax=plot_background_maze(x_all,y_all,hist2d_kws=hist2d_kws,fig=fig,ax=ax)
    
    scat = ax.scatter(data[0,0],data[1,0],alpha=1,c='red',s=10)
    if trial_l is not None:
        ntrials = int(trial_l.max())
        text=ax.text(0.95,0.95,"",transform=ax.transAxes,ha='right',va='top')
    if speed_l is not None:
        text_sp = ax.text(1.,0.85,"",transform=ax.transAxes,ha='right',va='top',color='C1')
    def init():
        ax.set(**ax_kws_)
        
        
            
        return scat,
    
    def animate(frame):
        start_frame = np.maximum(frame+1-n_tail,0)
        scat.set_offsets(data[:,start_frame:(frame+1)].T)
        to_return = [scat]
        if trial_l is not None:
            text.set_text(f"trial: {int(trial_l[frame])} / {ntrials}\n frame:{frame}")
            to_return.append(text)
            # return scat, text
        if speed_l is not None:
            text_sp.set_text(f'v = {speed_l[frame]:.02f}')
            to_return.append(text_sp)
        to_return = tuple(to_return)
        return to_return
        # return scat,
    
    kws = dict(blit=True,frames=data.shape[1],interval=10)
    kws.update(animate_kws)
    ani = FuncAnimation(fig,animate,init_func=init,**kws)
    plt.show()
    if dosave:
        save_dir=misc.get_or_create_subdir(save_dir)
        save_full_fn = os.path.join(save_dir,save_fn)    
        ani.save(f'{save_full_fn}',dpi=100)
        # ani.to_jshtml()
    return fig,ax,ani



def animate_3d_scatter(data,legend_handles=None,color_l=None,marker_l=None,fig=None,ax=None,ax_kws={},animate_kws={},dosave=False):
    '''
    data: 3 x T
    color_l: T,

    '''

    if color_l is None:
        color_l = ['C0'] * data.shape[1]
    if marker_l is None:
        marker_l = ['o'] * data.shape[1]
    if ax is None:
        fig=plt.figure()
        ax = fig.add_subplot(projection='3d')
    
    ax_kws_ = {}
    ax_kws_['xlim'] = (data[0].min() - np.abs(data[0].min()) * 0.2, data[0].max() + np.abs(data[0].max())*0.2)
    ax_kws_['ylim'] = (data[1].min() - np.abs(data[1].min()) * 0.2, data[1].max() + np.abs(data[1].max())*0.2)
    ax_kws_['zlim'] = (data[2].min() - np.abs(data[2].min()) * 0.2, data[2].max() + np.abs(data[2].max())*0.2)
    ax_kws_.update(ax_kws)

    def init():
        ax.set(**ax_kws_)
        if legend_handles is not None:
            ax.legend(handles=legend_handles)
        return scat,
    
    def animate(frame,data,scat,color_l):
        scat._offsets3d = data[:,:frame]
        scat.set_color(color_l[:frame])
        # alpha_l = np.append(np.ones(frame-2) * 0.5,1)
        # alpha_l=list(np.ones(frame-2) * 0.5) + [1] # not sure how to make changing alpha yet!!!
        
        # scat.set_alpha(alpha_l)
        # scat.set_marker(marker_l[:frame])
        return scat,
    
    scat = ax.scatter(data[0,0],data[1,0],data[2,0],alpha=0.4)
    
    kws = dict(blit=False,frames=data.shape[1],interval=10)
    kws.update(animate_kws)
    ani = FuncAnimation(fig,animate,init_func=init,fargs=(data,scat,color_l),**kws)
    plt.show()
    if dosave:
        pass
    return fig,ax,ani

def animate_line(data,legend_handles=None,color_l=None,fig=None,ax=None,ax_kws={},animate_kws={},dosave=False):
    
    if ax is None:
        fig,ax = plt.subplots()
    if color_l is None:
        color_l = ['C0'] * data.shape[1]

    line, = ax.plot([],[])

    ax_kws_ = {}
    ax_kws['xlim'] = (data[0].min() , data[0].max())
    ax_kws['ylim'] = (data[1].min() - np.abs(data[1].min()) * 0.2, data[1].max() + np.abs(data[1].max())*0.2)

    def init():
        ax.set(**ax_kws)
        if legend_handles is not None:
            ax.legend(handles=legend_handles)
        return line,

    def animate(frame,data,line,color_l):
        # line.set_data(data[:,:frame])
        line.set_data(data[0,:frame],data[1,:frame])
        # line.set_color(color_l[:frame])
        # scat.set_marker(marker_l[:frame])
        return line,

    kws = dict(blit=False,frames=data.shape[1],interval=10)
    kws.update(animate_kws)
    ani = FuncAnimation(fig,animate,init_func=init,fargs=(data,line,color_l),**kws)    
    plt.show()
    return fig,ax, ani



def line_morph(frame,data,line,color_l):
    '''
    data: first column: position; later ones: to plot
    '''
    # line.set_data(data[:,:frame])
    # xs = np.arange(data.shape[0])
    
    xs = data[:,0]
    line.set_data(xs,data[:,frame+1])
    line.set_color(color_l[frame])
    

    # scat.set_marker(marker_l[:frame])
    return line,

def animate_line_functional(animate_func,data,legend_handles=None,color_l=None,fig=None,ax=None,ax_kws={},animate_kws={},dosave=False):
    
    if ax is None:
        fig,ax = plt.subplots()
    if color_l is None:
        color_l = ['C0'] * data.shape[1]

    line, = ax.plot([],[],marker='o')

    ax_kws_ = {}
    ax_kws_['xlim'] = (data[0].min() , data[0].max())
    ax_kws_['ylim'] = (data[1].min() - np.abs(data[1].min()) * 0.2, data[1].max() + np.abs(data[1].max())*0.2)
    ax_kws_.update(ax_kws)

    def init():
        ax.set(**ax_kws_)
        if legend_handles is not None:
            ax.legend(handles=legend_handles)
        return line,

    # def animate(frame,data,line,color_l):
    #     # line.set_data(data[:,:frame])
    #     line.set_data(data[0,:frame],data[1,:frame])
    #     # line.set_color(color_l[:frame])
    #     # scat.set_marker(marker_l[:frame])
    #     return line,

    kws = dict(blit=False,frames=data.shape[1],interval=10)
    kws.update(animate_kws)
    ani = FuncAnimation(fig,animate_func,init_func=init,fargs=(data,line,color_l),**kws)    
    
    plt.show()
    return fig,ax, ani


def animate_line_morph(data,legend_handles=None,color_l=None,linestyle_l=None,fig=None,ax=None,ax_kws={},animate_kws={},max_length=5,dosave=False,save_kws={}):
    if ax is None:
        fig,ax = plt.subplots()
    if color_l is None:
        color_l = ['C0'] * data.shape[1]
    if linestyle_l is None:
        linestyle_l = ['-'] * data.shape[1]

    line_list = []

    ax_kws_ = {}
    ax_kws_['xlim'] = (data[:,0].min() , data[:,0].max())
    ax_kws_['ylim'] = (data[:,1:].min().min() - np.abs(data[:,1:].min().min()) * 0.2, data[:,1:].max().max() + np.abs(data[:,1:].max().max())*0.2)
    ax_kws_.update(ax_kws)

    def init():
        ax.clear()
        ax.set(**ax_kws_)
        if legend_handles is not None:
            ax.legend(handles=legend_handles)
    
    def update(i):
        if not i:
            init()
            line_list[:] = []
        # else: # change alpha for older lines
            # diff2max = max(0, max_length-len(line_list))
            # [x.set_alpha((j+diff2max)/max_length) for j, x in enumerate(line_list)]
        #delete line segments that we don't see anymore to declutter the space
        if len(line_list)>max_length:
            del_line = line_list.pop(0)
            del_line.remove()    
        
        #plot new segment and append it to the list
        newsegm, = ax.plot(data[:,0],data[:,i+1],color=color_l[i],linestyle=linestyle_l[i]) 
        line_list.append(newsegm)
    
    kws = dict(blit=False,frames=data.shape[1]-1,interval=10,repeat=False)
    kws.update(animate_kws)
    ani = FuncAnimation(fig,update,init_func=init,**kws)

    if dosave:
        dir = misc.get_or_create_subdir(save_kws['root'], 'py_figures',save_kws['sub_dir'])
        writervideo = animation.ImageMagickWriter(fps=save_kws['fps'])
        ani_name = save_kws['name']
        save_name=f'{ani_name}.gif'
        fn_full = os.path.join(dir,save_name)
        ani.save(fn_full, writer=writervideo)
        print(f'{fn_full} saved!')


    return fig,ax, ani

    

import matplotlib
import itertools
def map_colors(trial_l,npos=None,cmap_name='vlag'):
    '''
    trial_l: some list of values to be mapped onto colors
    '''
    minima = min(trial_l)
    maxima = max(trial_l)
    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    cmap = matplotlib.cm.get_cmap(cmap_name)
    # fig,ax=plt.subplots(figsize=(1,5))
    # cb = matplotlib.colorbar.ColorbarBase(ax,cmap=cmap,norm=norm)
    rgba = [cmap(norm(x)) for x in trial_l]
    if npos is not None: # if need one color for each position point
        rgba = [[i]*npos for i in rgba]
        rgba=list(itertools.chain.from_iterable(rgba))
    else:
        return rgba # one color for one trial

