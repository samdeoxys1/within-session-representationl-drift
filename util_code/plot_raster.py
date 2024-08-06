import numpy as np
import scipy
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from brokenaxes import brokenaxes
from matplotlib.gridspec import GridSpec
import copy,sys,os,pdb,importlib
from importlib import reload
import pandas as pd
import seaborn as sns
import glob

import place_cell_analysis as pa
reload(pa)

import data_prep_new as dpn
reload(dpn)
import plot_helper as ph
from plot_helper import *
# import place_cell_analysis as pa

import scipy.ndimage
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

from scipy.interpolate import interp1d 

import matplotlib
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
rcdict={'axes.labelsize':20,'axes.titlesize':20}
matplotlib.rcParams.update(rcdict)

DATABASE_LOC = '/mnt/home/szheng/ceph/ad/database.csv'
# db = pd.read_pickle(DATABASE_LOC)
db = pd.read_csv(DATABASE_LOC,index_col=[0,1])

# db.sort_values('n_pyr_putative',ascending=False).groupby(level=0).head(1)

azsess='AZ11_210504_sess10'
nazsess='Naz1_210518_sess17'#'Naz2_210422_sess8'
romsess='e15_13f1_220117'


# 2d plot raster
def plot_spikes_2d(unit,pos_2d,spk_triggered_positions_trial_all,cell_metric,**kwargs):
    '''
    unit: 0 based index; translate to uid by +1
    '''
    if 'ax' in kwargs.keys():
        ax=kwargs['ax']
        fig=kwargs['fig']
    else:
        fig,ax=plt.subplots()
    ax.plot(pos_2d[:,0],pos_2d[:,1],color='0.8',linewidth=0.1,alpha=0.5)
#     unit=83
    # toplotdata=spk_triggered_positions_trial_all[unit][0]
    toplotdata=spk_triggered_positions_trial_all[unit][0]
    
    ax.scatter(toplotdata[:,0],toplotdata[:,1],s=0.1)
    ax.set_title(f"{cell_metric['putativeCellType'][unit]}, uid={unit+1}\nFR={cell_metric['firingRate'][unit]:.02f}")
    return fig,ax

def plot_arm_color(behav_df,section_colordict,**kwargs):
    if 'ax' in kwargs.keys():
        ax=kwargs['ax']
        fig=kwargs['fig']
    else:
        fig,ax=plt.subplots()
    behav_df_downsampled=behav_df.iloc[:-1:40]
    for key,val in behav_df_downsampled.groupby('section_together'):
        ax.plot(val['x'],val['y'],c=section_colordict[key])
    return fig,ax
        

from matplotlib.ticker import MaxNLocator
def plot_spikes_on_lin_tmaze(spk_triggered_positions_trial,choice, section_markers=None,section_colordict=None,**kwargs):
    kw={'figsize':(10,6)}
    scatter_kwargs={'marker':'|','linewidth':0.7,'s':80}
    kw.update(kwargs)
    figsize=kw['figsize']
    if 'scatter_kwargs' in kwargs.keys():
        scatter_kwargs.update(kwargs['scatter_kwargs'])
    figisze=kw['figsize']
    
    ntrials_l = [np.sum(choice==ch) for ch in [0,1]]
    if 'axs' not in kwargs.keys():
        fig,axs = plt.subplots(2,1,figsize=figsize,sharex=True,gridspec_kw={'height_ratios': ntrials_l})
#         fig,axs = plt.subplots(2,1,figsize=figsize,sharex=True)
    else:
        fig=kwargs['fig']
        axs=kwargs['axs']

    if section_markers is None:
        section_markers = np.array([0,74,111,185,222])
    if section_colordict is None:
        section_colordict = make_section_colordict()
    
    for ch in [0,1]:
        trial_spk_pair_l=[]
        trial_mask = choice==ch
    #     spk_triggered_positions_trial = spk_triggered_positions_trial_all_speedmasked[unit][trial_mask]
        spk_triggered_positions_trial_ = spk_triggered_positions_trial[trial_mask]
        for (trial,spk_trial) in enumerate(spk_triggered_positions_trial_):
            trial_col=np.ones(len(spk_trial),dtype=np.int32)*trial
            trial_spk_pair=np.concatenate([spk_trial[:,None],trial_col[:,None]],axis=1)
            trial_spk_pair_l.append(trial_spk_pair)
        trial_spk_pair_l=np.concatenate(trial_spk_pair_l,axis=0)
        # then get color
        section_assignments=pd.cut(trial_spk_pair_l[:,0],section_markers,labels=False,include_lowest=True,right=True)
        c_l = [section_colordict[(ch,sec)] for sec in section_assignments]

        axs[ch].scatter(trial_spk_pair_l[:,0],trial_spk_pair_l[:,1],c=c_l,**scatter_kwargs)
        axs[ch].yaxis.set_major_locator(MaxNLocator(integer=True))
        axs[ch].axvline(15,linestyle=':')
        axs[ch].axvline(205,linestyle=':')
        axs[ch]=turn_off_spines(axs[ch])
    plt.tight_layout()
    return fig,axs
#     plt.xlim([15,205])

def make_section_colordict():
    pal=sns.color_palette()
    section_colordict={(0,):pal[-3],(0,1):pal[0],(1,1):pal[-1],(0,2):pal[1],(1,2):pal[3],(0,3):pal[2],(1,3):pal[-2]}
    section_colordict[(0,0)]=section_colordict[(0,)] # to make the dict easier to access, include the keys of both trial,0 combo
    section_colordict[(1,0)]=section_colordict[(0,)]
    return section_colordict

def plot_firing_map_1d(pos_bins_dict,fr_map_one_unit,**kwargs):
    bin_to_lin=copy.copy(pos_bins_dict['lin'])
    bin_to_lin=np.concatenate([bin_to_lin[:-1,None],bin_to_lin[1:,None]],axis=1).mean(axis=1)
    if 'ax' not in kwargs.keys():
        fig,ax=plt.subplots()
    else:
        ax=kwargs['ax']
        fig=kwargs['fig']
    ax.plot(bin_to_lin, fr_map_one_unit,c='k',linewidth=1.5)
    return fig,ax

def plot_spikes_fr_on_lin_tmaze(unit_1_ind,spk_triggered_positions_trial_all,pos_bins_dict,fr_map_dict,choice,**kwargs):
    '''
    unit is a number (0-indexed) index for spk_triggered_positions_trial_all; but a string (1-indexed) index in fr_map_dict; careful!!
    '''
    fig,axs=plot_spikes_on_lin_tmaze(spk_triggered_positions_trial_all[unit_1_ind-1],choice,**kwargs)

    for ch in [0,1]:
        ax_=axs[ch].twinx()
        # ax_.spines['right'].set_position(('axes',1.0))
        ax_.yaxis.set_ticks_position('right')
        ax_.set_ylabel('Firing Rate')
        fr_map=fr_map_dict[ch].loc[f'unit_{unit_1_ind}']
        fig,ax_=plot_firing_map_1d(pos_bins_dict,fr_map,fig=fig,ax=ax_)
        ax_=turn_off_spines(ax_)
        ax_.spines['right'].set_visible(True)
    return fig,axs

def preprocess_for_plot(sess_name,data_dir_full=None,sigma=30,speedmask=5,n_pos_bins=100,gauss_width=2.5):
    # load data
    if data_dir_full is None:
        data_dir_full = db.query(f'sess_name=="{sess_name}"').loc[:,'data_dir_full'].iloc[0]

    cell_metric,behavior,spike_times,uid,fr,cell_type,mergepoints,behav_timestamps,position,\
                rReward,lReward,endDelay,startPoint,visitedArm \
    = dpn.load_sess(sess_name=sess_name, data_dir=None, data_dir_full=data_dir_full)

    # prep save directory
    fig_path_root=os.path.join(data_dir_full,'py_figures')
    if not os.path.exists(fig_path_root):
        os.mkdir(fig_path_root)
        print(f'{fig_path_root} made!')
    fig_sub_path=os.path.join(fig_path_root,'raster')
    if not os.path.exists(fig_sub_path):
        os.mkdir(fig_sub_path)
        print(f'{fig_sub_path} made!')

    # prepro speed
    pos_2d = np.stack([behavior['position']['x'],behavior['position']['y']],axis=1)
    dt = behav_timestamps[2]-behav_timestamps[1]
    speed,_ = dpn.smooth_get_speed(pos_2d,dt,sigma=sigma)

    # 2d prepro
    # 2d preprocess
    trial_markers = np.array([[behav_timestamps[0],behav_timestamps[-1]]])
    pos = pos_2d
    time_stamps=behav_timestamps
    spk_times_all=cell_metric['spikes']['times']
    time_stamps_trial, spk_times_trial_all, [pos_trial,speed_trial]=\
    pa.get_stuff_by_trial(trial_markers,behav_timestamps,spk_times_all,pos,speed)

    # 2d get spike triggered position
    spk_triggered_positions_trial_all_speedmasked_2d = pa.get_spk_triggered_positions(pos_trial,time_stamps_trial,spk_times_trial_all,speedmask=speedmask,speed_trial=speed_trial)

    # [[[[[[trial definition; might change!]]]]]]
    trial_ints = behavior['trials']['startPoint']
    mask=np.isnan(trial_ints)
    mask = ~np.logical_or(mask[:,0],mask[:,1])
    trial_ints = trial_ints[mask]
    choice=behavior['trials']['visitedArm'][mask]

    # linear scatter plot preparation: spike triggered position by trial
    pos=behavior['position']['lin']
    trial_markers =trial_ints
    spk_times_all=cell_metric['spikes']['times']
    time_stamps_trial, spk_times_trial_all, [pos_trial,speed_trial]=\
    pa.get_stuff_by_trial(trial_markers,behav_timestamps,spk_times_all,pos,speed)
    spk_triggered_positions_trial_all_speedmasked = pa.get_spk_triggered_positions(pos_trial,time_stamps_trial,spk_times_trial_all,speedmask=speedmask,speed_trial=speed_trial)

    # assign section/arm to each time
    behav_df,_ = dpn.get_beh_df(behav_timestamps,position,visitedArm,startPoint,n_pos_bins=n_pos_bins)
    # section_markers = np.array([0,75,112.5,187.5,222])
    section_markers = np.array([0,74,111,185,222])
    behav_df['section']=pd.cut(behav_df['lin'],section_markers,labels=False,include_lowest=True)
    behav_df.groupby(['section','visitedArm'])
    func = lambda x:x.assign(section_together=[(x.name[1],)]*len(x)) if x.name[1]==0 else x.assign(section_together=[x.name]*len(x))
    behav_df=behav_df.groupby(['visitedArm','section']).apply(func)

    # make a colordict for different sections of the maze
    import seaborn as sns
    pal=sns.color_palette()
    section_colordict={(0,):pal[-3],(0,1):pal[0],(1,1):pal[-1],(0,2):pal[1],(1,2):pal[3],(0,3):pal[2],(1,3):pal[-2]}
    section_colordict[(0,0)]=section_colordict[(0,)] # to make the dict easier to access, include the keys of both trial,0 combo
    section_colordict[(1,0)]=section_colordict[(0,)]

    # process firing map
    df_dict, pos_bins_dict,cell_cols_dict = dpn.get_fr_beh_df(spike_times,uid,behav_timestamps,cell_type,position,visitedArm,startPoint,n_pos_bins=n_pos_bins,israte=False)
    df1 = df_dict['pyr']
    df2 = df_dict['int']
    cols = df2.columns.difference(df1.columns)
    df_all = df1.join(df2[cols])
    cell_cols_all=cell_cols_dict['pyr']+cell_cols_dict['int']

    fr_map_all_trials_dict = pa.get_fr_map_trial(df_all,cell_cols_all,gauss_width=gauss_width,order=['smooth','divide','average'],speed_thresh=speedmask,n_lin_bins=n_pos_bins)
    fr_map_dict = {0:fr_map_all_trials_dict[0][0],1:fr_map_all_trials_dict[1][0]}

    return cell_metric,pos_2d,spk_triggered_positions_trial_all_speedmasked_2d,behav_df,section_colordict,spk_triggered_positions_trial_all_speedmasked,pos_bins_dict,choice,fr_map_dict,fig_sub_path


def plot_one_cell_(uid,cell_metric,pos_2d,spk_triggered_positions_trial_all_speedmasked_2d,behav_df,section_colordict,spk_triggered_positions_trial_all_speedmasked,pos_bins_dict,choice,fr_map_dict,fig_sub_path,savefig=False,sess_name=None):
    '''
    uid is 1 based
    sess_name only needed if savefig is True
    use all result from preprocess_for_plot
    '''
    nunits=len(spk_triggered_positions_trial_all_speedmasked_2d)
    unit = uid
    fig,axs=plt.subplot_mosaic([['0','1'],['2','2'],['3','3']],figsize=(10,10),gridspec_kw={'height_ratios':[1,0.5,0.5]})
    axs=list(axs.values())
    # 2d raster use unit 0-based index;
    fig,ax=plot_spikes_2d(unit-1,pos_2d,spk_triggered_positions_trial_all_speedmasked_2d,cell_metric,fig=fig,ax=axs[0])
    ax.axis('off')
    fig,ax=plot_arm_color(behav_df,section_colordict,fig=fig,ax=axs[1])
    ax.axis('off')
    # plot_spikes_fr_on_lin_tmaze use unit 1-based index
    fig,axs=plot_spikes_fr_on_lin_tmaze(unit,spk_triggered_positions_trial_all_speedmasked,pos_bins_dict,fr_map_dict,choice,fig=fig,axs=axs[2:])

    plt.tight_layout()
    if savefig:
        fn=os.path.join(fig_sub_path,f'{sess_name}_unit{unit}.pdf')
        fig.savefig(fn,bbox_inches='tight')
        print(f'{fn} saved!')
        plt.close(fig)
        return fig,ax
    else:
        return fig,ax

def plot_one_cell(uid,sess_name,sigma=30,speedmask=5,n_pos_bins=100,savefig=False):
    args = preprocess_for_plot(sess_name,sigma=sigma,speedmask=speedmask,n_pos_bins=n_pos_bins)
    fig,ax=plot_one_cell_(uid,*args,savefig=savefig,sess_name=sess_name)
    return fig,ax

def plot_one_session(sess_name,data_dir_full=None,sigma=30,speedmask=5,n_pos_bins=100):
    sess_name = sess_name
    args = preprocess_for_plot(sess_name,data_dir_full=data_dir_full,sigma=sigma,speedmask=speedmask,n_pos_bins=n_pos_bins)
    # plot all cells one session
    cell_metric=args[0]
    nunits=len(cell_metric['UID'])
    for uid in range(1,nunits+1):
        plot_one_cell_(uid,*args,savefig=True,sess_name=sess_name)
    



#==================spike on trajectory===========#
def plot_spike_on_lin_trajectory_all_units(data_dir_full=None,cell_metrics=None,behavior=None,u_ind_l=None,
                                           dosave=False,save_dir=None,save_fn='raster1d_allunit.pdf',fig=None,axs=None
                                          ):
    '''
    plot spike on top of 1d trajectory for neurons with indices (0 based) in u_ind_l; 
    either specify data_dir_full: full path that contains the .mat
    or cell_metrics and behavior + save_dir
    '''
    if cell_metrics is None:
        cell_metrics=glob.glob(os.path.join(data_dir_full,'*cell_metrics.cellinfo.mat'))[0]
        cell_metrics = dpn.loadmat_full(cell_metrics,'cell_metrics')
    if behavior is None:
        behavior=glob.glob(os.path.join(data_dir_full,'*Behavior*.mat'))
        behavior=[b for b in behavior if 'Tracking' not in b][0] # avoid some struct that also contains Tracking
        behavior = dpn.loadmat_full(behavior,'behavior')
    spike_times = cell_metrics.spikes.times
    lin=behavior.position.lin
    beh_timestamps = behavior.timestamps
    pos_func = scipy.interpolate.interp1d(beh_timestamps,lin)

    beh_interval = behavior.timestamps[[0,-1]]
    duration = beh_interval[-1]-beh_interval[0]
    spike_times_beh,beh_epoch_mask = dpn.select_time_in_intervals_all(spike_times,beh_interval)


    trial_ints = behavior['trials']['startPoint']
    mask=np.isnan(trial_ints)
    mask = ~np.logical_or(mask[:,0],mask[:,1])
    trial_ints = trial_ints[mask]
    choice=behavior['trials']['visitedArm'][mask]
    correct= behavior.trials.choice

    beh_timestamps_trial,mask_trial = dpn.select_time_in_intervals(beh_timestamps,trial_ints)
    if u_ind_l is None:
        u_ind_l = np.arange(len(cell_metrics.UID))

    nplots = len(u_ind_l)

    if axs is None:
        fig,axs=ph.subplots_wrapper(nplots,return_axs=True,basewidth=duration/100,baseheight=5)
    if nplots==1:
        axs = np.array([axs])
    # fig,ax=plt.subplots(figsize=(duration/100,5))
    for jj,u_ind in enumerate(u_ind_l):
        ax = axs.ravel()[jj]

    #     u_ind=2
        spk_1 = spike_times_beh[u_ind]
        spk_trig_pos_1 = pos_func(spk_1)
        # ax.plot(beh_timestamps,behavior.position.lin,alpha=0.1)
        ax.scatter(spk_1,spk_trig_pos_1,c='k',marker='|')
        correct_to_alpha = {True:0.5,False:0.2}
        correct_to_linewidth={True:2,False:0.5}
        for ii,(bt,mt) in enumerate(zip(beh_timestamps_trial,mask_trial)):
            choice_type = choice[ii]
            rewarded = correct[ii]
            if np.isnan(rewarded):
                rewarded=True
            ax.plot(bt,behavior.position.lin[mt],linewidth=correct_to_linewidth[rewarded],alpha=correct_to_alpha[rewarded],c=f'C{choice_type}')
        ax=ph.turn_off_spines(ax,to_turn=['top','right'])
        ax.set_ylabel('Linearized Position (cm)')
        ax.set_xlabel('time (s)')

        #title
        uid = cell_metrics['UID'][u_ind]
        fr=cell_metrics['firingRate'][u_ind]
        celltype =cell_metrics['putativeCellType'][u_ind]
        ax.set_title(f'{celltype}, UID={uid}\nFR={fr:.02f}')
    plt.tight_layout()
    if dosave:
        if save_dir is None:
            save_dir = os.path.join(data_dir_full,'py_figures','raster1d')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                print(f'{save_dir} created!')
        fig.savefig(os.path.join(save_dir,save_fn))
        print(f'{save_fn} saved at {save_dir}')
    
    return fig,axs
    

#================population raster=========#
import pynapple as nap
def population_raster(spike_trains,uid_l=None,ep=None,fig=None,ax=None,pos_func=None,subplot_spec=None):
    if isinstance(ep,list) or isinstance(ep,tuple):
        ep = nap.IntervalSet(start=ep[0],end=ep[1])
    elif isinstance(ep,np.ndarray):
        ep = nap.IntervalSet(start=ep[:,0],end=ep[:,1])
    # if uid_l is not None:
        # spike_trains_sub = spike_trains[uid_l]
    if uid_l is None:
        uid_l = spike_trains.keys()
        # spike_trains_sub = spike_trains
        

    if ax is None:
        if ep is None:
            fig,ax=plt.subplots()
            brokenaxis=False
        else:
            if ep.shape[0]>1:
                
                fig=plt.figure(figsize=(10,6))
                
                ax = brokenaxes(xlims=tuple(ep.values),subplot_spec=subplot_spec)
                brokenaxis=True
            else:
                fig,ax=plt.subplots()
                brokenaxis=False
    
    for ii,n in enumerate(uid_l):
        if ep is not None:
            data = spike_trains[n].restrict(ep).fillna(ii)
        else:
            data = spike_trains[n].fillna(ii)
        ax.plot(data,'|')
    
    if pos_func is not None: # turn x axis into position
        if brokenaxis:
            axs = ax.axs
        else:
            axs=[ax]
        for aa in axs:
            aa.xaxis.set_major_locator(ticker.MaxNLocator(6))
            aa.set_xticklabels(np.round(pos_func(aa.get_xticks()),1),rotation=45)
    
    return fig,ax,ep
import scipy.interpolate
from scipy.interpolate import interp1d
def population_raster_with_behvar(spike_trains,beh_df,uid_l=None,ep=None,behvar_l=[],fig=None,axs=None,convert_to_pos='lin'):
    if convert_to_pos is not None:
        pos_func = interp1d(beh_df.index,beh_df[convert_to_pos])
    else:
        pos_func=None
    if isinstance(ep,list) or isinstance(ep,tuple):
        ep = nap.IntervalSet(start=ep[0],end=ep[1])
    elif isinstance(ep,np.ndarray):
        ep = nap.IntervalSet(start=ep[:,0],end=ep[:,1])
    brokenaxis = False
    if ep is not None:
        if ep.shape[0]>1:
            brokenaxis=True

    nplots = len(behvar_l) + 1
    height_ratios = [2] + [1] * len(behvar_l) # make the rasterplot taller
    if axs is None:
        if not brokenaxis:
            fig,axs=plt.subplots(nplots,1,figsize=(10,(nplots-1)*2+4),sharex=True,gridspec_kw={'height_ratios':height_ratios})
        else:
            # fig = plt.subplots(figsize=(10,(nplots-1)*2+4),constrained_layout=False)
            sps = GridSpec(nplots,1)
    if axs is not None:
        fig,ax,ep=population_raster(spike_trains,uid_l=uid_l,ep=ep,fig=fig,ax=axs[0],pos_func=pos_func)
    else:
        axs_l = []
        fig,ax,ep=population_raster(spike_trains,uid_l=uid_l,ep=ep,fig=fig,subplot_spec=sps[0],pos_func=pos_func)
        axs_l.append(ax)
    if ep is not None:
        beh_df_restrict = nap.TsdFrame(beh_df).restrict(ep)
    else:
        beh_df_restrict = beh_df
    for ii,behvar in enumerate(behvar_l):
        if axs is not None:
            axs[ii+1].plot(beh_df_restrict[behvar].index,beh_df_restrict[behvar],label=behvar)
            axs[ii+1].legend()
        else:
            bax = brokenaxes(tuple(ep.values),subplot_spec=sps[ii+1])
            bax.plot(beh_df_restrict[behvar].index,beh_df_restrict[behvar],label=behvar)
            bax.legend()
            axs_l.append(bax)
    fig.tight_layout()
    # to figx issue in bax caused by tight layout
    if brokenaxis:
        for bax in axs_l:
            for handle in bax.diag_handles:
                handle.remove()
            bax.draw_diags()
    
    return fig,axs

# plot place field boundaries by trial
def plot_field_boundaries(field_bounds_one_neuron,cl_key='field_index_cl'):
    '''
    field_bounds_one_neuron: df, nfields x [start, end, peak, com, field_index_cl, trial]
    '''

    fig,ax=plt.subplots()
    for tt,row in field_bounds_one_neuron.iterrows():
    #     cl=row['cl_from_start']
    #     cl=row['cl_from_ed']
        cl = row[cl_key]            
        # ax.axhline(tt,xmin=row['start'] / 100,xmax=row['end'] / 100,color=f'C{cl}')
        ax.plot((row['start'],row['end']),(tt,tt),color=f'C{cl}')
    ax.set_xlim(0,100)
    return fig,ax