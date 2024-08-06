import numpy as np
import scipy,os,copy,pickle,sys,itertools,pdb
from importlib import reload
import pandas as pd
from scipy.interpolate import interp1d

from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
sys.path.append('/mnt/home/szheng/projects/util_code')
import data_prep_pyn as dpp

'''
make sure beh_df already numerically indexed; currently assuming no holes in the index
'''


def combine_tmaze_turninfo(xy_sampled_d, segment_d,line_pd_d):
    '''
    combine results from corners_d,xy_sampled_d,segment_d=dpp.find_tmaze_turns(beh_df,**find_turns_kws_)
    assuming the dictionary only contains two turns
    hard coded: 4 segments per turn, turn 0: 0,1,2,3, the other turn 5,6,7,8
    '''

    # index = list(range(4)) + [0] + list(range(4,7))
    # corners = pd.concat(corners_d.values(),axis=0)
    # corners.index = index

    xy_sampled = np.concatenate(list(xy_sampled_d.values()),axis=0)

    tt_l = list(segment_d.keys())
    segment_to_be_modified = segment_d[tt_l[1]]+4
    # segment_to_be_modified[segment_to_be_modified==0] =segment_to_be_modified[segment_to_be_modified==0] + 3
    segment = np.concatenate([segment_d[tt_l[0]],segment_to_be_modified],axis=0)

    line_pd = np.concatenate(list(line_pd_d.values()),axis=0)

    return  xy_sampled, segment,line_pd


def linefit_with_direction(xy,thresh=1e-1,corner_median_number=5):
    '''
    fit a line to xy: n_samples x 2
    get anchor point p0 and direction D
    '''
#     p0 = np.median(xy[0:corner_median_number],axis=0)
    p0 = np.mean(xy,axis=0) # use the mean
    xy_ = xy - p0[None,:]
    x = xy_[:,0]
    y = xy_[:,1]
    stdx = np.std(x)
    stdy = np.std(y)
    if stdx < thresh*stdy:
        dir_y = 1 if np.sign(y).sum() > 0 else -1
        D = np.array([0,dir_y])
    elif stdy < thresh*stdx:
        dir_x = 1 if np.sign(x).sum() > 0 else -1
        D = np.array([dir_x,0])
    else:
        data = xy_
        mean_data = np.mean(data, axis=0)
        centered_data = data - mean_data
        _, _, Vt = np.linalg.svd(centered_data)
        v = Vt[-1]
        # Calculate the direction vector D and normalize it
        D = np.array([v[1], -v[0]])
        D = D / np.linalg.norm(D)
        diff = np.diff(centered_data,axis=0)
        projection = np.dot(diff,D[:,None])
        if np.sign(projection).sum() < 0:
            D = -D
    return p0,D

def get_segment_line(xy_sampled_d,segment_d,thresh=1e-1,corner_median_number=5):
    '''
    linefit each maze segment
    '''
    line_pd_d = {}
    
    for (k,xys),(k,seg) in zip(xy_sampled_d.items(),segment_d.items()):
        vec_l = []
        for s in np.unique(seg):
            data = xys[seg==s]
            p0,D=linefit_with_direction(data,thresh=thresh,corner_median_number=corner_median_number)
            vec_l.append([*p0,*D])
        vec_l = np.array(vec_l)
        line_pd_d[k] = vec_l
    return line_pd_d
    
def get_dist_to_line(xy_l,pd_l):
    '''
    pd_l: n x 4, or 4,; xy of the reference point, then xy direction of the line
    '''
    if len(pd_l.shape)==1:
        pd_l = np.tile(pd_l,(xy_l.shape[0],1)) # if pd_l is 1d, then broadcast
    diff = xy_l[:,:2] - pd_l[:,:2]
    proj = np.einsum('ni,ni->n',diff,pd_l[:,2:])
    dist = np.sqrt(np.linalg.norm(diff,axis=1)**2 - proj**2)
    return dist
from scipy.spatial.distance import cdist
# def get_dist_to_maze(xy_l,xy_sampled_all,segment_all,line_pd_all):
#     '''
#     idea, first find closest sample points, then find the segment and line direction of them, 
#     then use the closest sample points as reference to compute projection distance

#     '''
#     dist_to_sample = np.min(cdist(xy_l,xy_sampled_all),axis=1) # dist computed as shorted dist to sample
#     closest_pts_inds = np.argmin(cdist(xy_l,xy_sampled_all),axis=1)
#     closest_pts_seg = segment_all[closest_pts_inds].astype(int)
#     closest_pts_xy = xy_sampled_all[closest_pts_inds]
#     closest_pts_pd = line_pd_all[closest_pts_seg]
#     closest_pts_pd[:,:2] = closest_pts_xy
#     dist = get_dist_to_line(xy_l,closest_pts_pd)
#     return dist

def get_dist_to_maze(xy_l,xy_sampled_all):
    '''
    idea, first find closest sample points, then find the segment and line direction of them, 
    then use the closest sample points as reference to compute projection distance

    '''
    dist = np.min(cdist(xy_l,xy_sampled_all),axis=1) # dist computed as shorted dist to sample
    
    return dist


def get_mask_edges(ma):
    ma = np.concatenate([[0],ma,[0]])
    diff = np.diff(ma.astype(int))
    st = np.nonzero(diff == 1)[0] 
    ed = np.nonzero(diff == -1)[0]
    edges = np.array([st,ed]).T
    return edges

def merge_edges(edges,thresh=14):
    edges_merged = []
    i=0
    while i < (edges.shape[0]-1):
        j=i+1
        prev_st = edges[i,0]
        prev_ed = edges[i,1]
        next_st = edges[j,0]
        any_merge = False
        while (next_st - prev_ed) <= thresh:
            if j<=(edges.shape[0]-2):
                j=j+1
                next_st = edges[j,0]
                next_ed = edges[j,1]
                
            else:
                j=j+1
                break
            any_merge=True
                
        j = j-1
        edges_merged.append([prev_st,edges[j,1]])
        i = j+1    
    edges_merged = np.array(edges_merged)
    return edges_merged

def extend_off_track_to_on(off_track_edges,on_track_edges):
    '''
    extend the beginning/end of off_track_edges to the closest end/beginning of on_track_edges smaller/bigger
    if none, append 0/end
    '''
    token = 11111111
    mat = np.subtract.outer(off_track_edges[:,0], on_track_edges[:,1])
    mat[mat<0] =token # if nothing before, only negative values, replace with a token
    beg_minus_ed = np.min(mat,axis=1)
    beg_minus_ed_ind = np.argmin(mat,axis=1) 
    extended_st = np.zeros_like(beg_minus_ed_ind)
    extended_st[beg_minus_ed==token] = 0
    extended_st[beg_minus_ed!=token] = on_track_edges[beg_minus_ed_ind[beg_minus_ed!=token],1]


    mat = np.subtract.outer(off_track_edges[:,1], on_track_edges[:,0])
    mat[mat>0] =-token # if nothing before, only negative values, replace with a token
    ed_minus_beg = np.max(mat,axis=1)
    ed_minus_beg_ind = np.argmax(mat,axis=1)
    extended_ed = np.zeros_like(ed_minus_beg_ind)
    extended_ed[ed_minus_beg==-token] = 0
    extended_ed[ed_minus_beg!=-token] = on_track_edges[ed_minus_beg_ind[ed_minus_beg!=-token],0]

    extended_st_ed = np.array([extended_st, extended_ed]).T
    extended_st_ed = np.unique(extended_st_ed,axis=0)
    return extended_st_ed

def detect_offtrack_event(beh_df,find_turns_kws={},off_track_thresh = 3,
            on_track_thresh = 1,edges_merge_time=0.4,st_ed_dist_thresh = 20.):
    
    find_turns_kws_ = dict(n_lin=200,speed_key='speed_gauss',speed_thresh=10.)
    find_turns_kws_.update(find_turns_kws)
    # beh_df = beh_df.reset_index(drop=True)
    for k,bdf in beh_df.groupby('task_index'): # task can be duplicated!!!! use task_index!!!
        if bdf['task'].iloc[0]=='alternation':
            dt = np.median(np.diff(bdf.index))
            corners_d,xy_sampled_d,segment_d=dpp.find_tmaze_turns(bdf,**find_turns_kws_)
            xy_sampled_all = np.concatenate(list(xy_sampled_d.values()),axis=0)
        elif bdf['task'].iloc[0]=='linearMaze':
            xy_sampled_all,_ = dpp.get_xy_samples_from_lin_one(bdf)
        xy_l = bdf[['x','y']].values
        dist = get_dist_to_maze(xy_l,xy_sampled_all)
        
        beh_df.loc[bdf.index,'dist_to_maze'] = dist
        # prelim on off track
        beh_df.loc[bdf.index,'off_track'] = dist>off_track_thresh
        beh_df.loc[bdf.index,'on_track'] = dist<=on_track_thresh
        # merge
        off_track_edges = get_mask_edges(beh_df.loc[bdf.index,'off_track'])
        edge_merge_nbins = int(edges_merge_time / dt)
        off_track_edges_merged = merge_edges(off_track_edges,thresh=edge_merge_nbins)
        on_track_edges = get_mask_edges(beh_df.loc[bdf.index,'on_track'])
        # extend off track to on
        
        if off_track_edges_merged.shape[0]==0:
            beh_df.loc[bdf.index,'off_track_event'] = False
        else:
            off_track_extended_st_ed = extend_off_track_to_on(off_track_edges_merged,on_track_edges)

            # filter too short ones
            ma = (off_track_extended_st_ed[:,1]-off_track_extended_st_ed[:,0]) >= edge_merge_nbins
            off_track_extended_st_ed_no_short = off_track_extended_st_ed[ma]

            # filter st ed too far away
            st_ed_dist = np.linalg.norm(xy_l[off_track_extended_st_ed_no_short[:,0]] - xy_l[off_track_extended_st_ed_no_short[:,1]],axis=1)
            off_track_extended_st_ed_no_far = off_track_extended_st_ed_no_short[st_ed_dist <= st_ed_dist_thresh]

            # fill in the event
            off_track_event = np.zeros(bdf.shape[0],dtype=bool)
            for st,ed in off_track_extended_st_ed_no_far:
                off_track_event[st:ed] = 1
            beh_df.loc[bdf.index,'off_track_event'] = off_track_event



    return beh_df

def detect_speed_related_event(beh_df,exclude_key_l=['off_track_event'],
                        speed_key='speed_gauss',speed_thresh=1.,
                        compare_type='<=',
                        edges_merge_time = 0.2,
                        event_key = 'pause_event',
                        ):
    '''
    detect speed related event like pause, locomotion, etc
    speed can be bigger/smaller/in between
    can merge based on edges_merge_time
    use exclude_key_l to exclude rows where those columns are True
    '''
    gpb = beh_df.groupby('task_index')
    

    for k, bdf in gpb:
        dt = np.median(np.diff(bdf.index))
        # prelim on pause
        ma_l = [np.logical_not(bdf[key]) | bdf[key].isna() for key in exclude_key_l]
        ma = np.all(ma_l,axis=0)
        
        if compare_type=='<=':
            pause_prelim = (bdf[speed_key]<=speed_thresh) & ma
        elif compare_type=='<':
            pause_prelim = (bdf[speed_key]<speed_thresh) & ma
        elif compare_type=='>=':
            pause_prelim = (bdf[speed_key]>=speed_thresh) & ma
        elif compare_type=='>':
            pause_prelim = (bdf[speed_key]>speed_thresh) & ma
        elif compare_type=='between':
            pause_prelim = (bdf[speed_key].between(speed_thresh[0],speed_thresh[1])) & ma
        
        
        pause_edges = get_mask_edges(pause_prelim)

        # merge
        if edges_merge_time is not None:
            edge_merge_nbins = int(edges_merge_time / dt)
            pause_edges_merged = merge_edges(pause_edges,thresh=edge_merge_nbins)
        else:
            pause_edges_merged = pause_edges

        if pause_edges_merged.shape[0]==0:
            beh_df.loc[bdf.index,event_key] = False
        else:
            # fill in the event
            pause_event = np.zeros(bdf.shape[0],dtype=bool)
            for st,ed in pause_edges_merged:
                pause_event[st:ed] = 1
            beh_df.loc[bdf.index,event_key] = pause_event
    
    return beh_df

from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

def plot_one_headscan(sub_beh_df,one_chunk,colors=None,cmin=None,cmax=None,fig=None,ax=None,
                        color_discrete=False,cbar_tick_labels=None,figsize=(3,2),
                        **kwargs):
    kwargs_={'s':0.05,'cbar_label_norm':False}
    kwargs_.update(kwargs)
    if ax is None:
        fig,ax=plt.subplots(figsize=figsize)
    ax.plot(sub_beh_df['x'], sub_beh_df['y'],color='grey',alpha=0.3)
    if colors is None:
        colors = np.arange(one_chunk.shape[0])
    if isinstance(colors,str): # use provided single color
        c = colors
        ax.scatter(one_chunk['x'],one_chunk['y'],c=c,s=kwargs_['s'])
    else:
        cmap = plt.get_cmap('jet')
        if cmin is None:
            cmin = min(colors)
        if cmax is None:
            cmax = max(colors)
        if color_discrete:
            levels = MaxNLocator(nbins=int(cmax-cmin+1)).tick_values(cmin,cmax+1)
            norm = BoundaryNorm(levels,ncolors=cmap.N,clip=True)
        else:
            norm = plt.Normalize(cmin, cmax)
        c = cmap(norm(colors))
        ax.scatter(one_chunk['x'],one_chunk['y'],c=c,s=kwargs_['s'])
        
        cbar_label_norm = kwargs_['cbar_label_norm']
        if cbar_label_norm: #whether color bar labels bewteen 0 and 1 or full value range
            cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap),ax=ax)
        else:
            cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap,norm=norm),ax=ax)
        if cbar_tick_labels is not None:
            cbar.set_ticklabels(cbar_tick_labels)
    return fig,ax,cbar

# take the start of each headscan, keep: the xy locations 
def get_event_start_info(beh_df,event_key='off_track_event',drop_multiple=True):
    '''
    get the info at the time bin when some event start
    info: n_event x [x,y,lin,lin_binned,...]
    drop_multiple: for multiple events on the same trial and same position, only keep one
    '''
    event_key = 'off_track_event'
    # beh_df_sub = spk_beh_df_all.loc[ani,sess]
    event = beh_df[event_key]
    inds=np.nonzero(np.diff(event.astype(int),prepend=0)==1)
    info = beh_df.iloc[inds][['x','y','lin','lin_binned','task_index','trial_type','trial']]
    if drop_multiple:
        info = info.groupby(['task_index','trial_type','trial','lin'],group_keys=False).apply(lambda x:x.iloc[0])
    return info

def get_event_start_info_all_sess(beh_df_all,event_key='off_track_event',drop_multiple=True):
    gpb =  beh_df_all.groupby(level=(0,1),group_keys=False)
    info_all = gpb.apply(get_event_start_info,event_key=event_key,drop_multiple=drop_multiple)
    
    return info_all
