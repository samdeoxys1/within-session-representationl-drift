import numpy as np
import scipy
import matplotlib.pyplot as plt
import sys,os,copy,pdb,importlib
from importlib import reload
import pandas as pd
sys.path.append('/mnt/home/szheng/projects/util_code/')
sys.path.append('/mnt/home/szheng/projects/nmf_analysis')
sys.path.append('/mnt/home/szheng/projects/nmf_analysis/scripts')

import h5py

def turn_ascii_array_into_str(ascii):
    return "".join([chr(int(x)) for x in ascii])

def prep_one_category_within_one_exp(f,exp_ind,sess_ind,region='CA1',do_mask_transient=False):
    
    '''
    fn_full = '/mnt/home/szheng/ceph/ad/thomas_data/180301_DG_CA3_CA1.h5'
    f = h5py.File(fn_full, 'r')
    '''
    one_exp=f[f['data'][region][exp_ind,0]]
    n_uid = one_exp['cells'].shape[0]
    dfoy_alltrials_all_cells=[]
    dfot_alltrials_all_cells = []
    onehot_placefield_all_cells= []
    transientmask_alltrials_all_cells = []
    place_field_all_cells = []
    for uid in range(n_uid):
        one_cell = f[one_exp['cells'][uid,0]]
        one_cell_cat = f[one_cell['categories'][sess_ind,0]]
        dfoy_alltrials=np.concatenate([f[dfoy] for dfoy in np.squeeze(one_cell_cat['dFoY'])])
        dfoy_alltrials_all_cells.append(dfoy_alltrials.T)
        
        dfot_alltrials=np.concatenate([f[dfot] for dfot in np.squeeze(one_cell_cat['dFoT'])]).squeeze()
        dfot_alltrials_all_cells.append(dfot_alltrials)
        
        transientmask_alltrials=np.concatenate([f[transientmask] for transientmask in np.squeeze(one_cell_cat['transientmask'])]).squeeze()
        transientmask_alltrials_all_cells.append(transientmask_alltrials)
        
        place_field = np.squeeze(one_cell_cat['placefield'])[::2] # [NB!!!! particular, might need to check] downsample to match the dFoY
        place_field_all_cells.append(place_field)

        # onehot_placefield_all_cells.append(np.squeeze(one_cell_cat['placefield']))
    # concat stuff
    place_field_all_cells = np.stack(place_field_all_cells)
    dfoy_alltrials_all_cells = np.stack(dfoy_alltrials_all_cells)
    # onehot_placefield_all_cells = np.array(onehot_placefield_all_cells)
    dfot_alltrials_all_cells = np.array(dfot_alltrials_all_cells)
    transientmask_alltrials_all_cells = np.array(transientmask_alltrials_all_cells)

    # load stuff common to a session
    lin_all_ref = f[one_exp['metadata']['categories'][sess_ind,0]]['y']
    lin_alltrials = np.concatenate([f[lin_ref] for lin_ref in np.squeeze(lin_all_ref)],axis=0).squeeze()

    moving_all_ref = f[one_exp['metadata']['categories'][sess_ind,0]]['moving']
    moving_alltrials = np.concatenate([f[move_ref] for move_ref in np.squeeze(moving_all_ref)],axis=0).squeeze()

    trial_name_ref = f[one_exp['metadata']['categories'][sess_ind,0]]['filename']
    trial_name=np.array([turn_ascii_array_into_str(f[tn_r]) for tn_r in np.squeeze(trial_name_ref)]).squeeze()

    trial_index_within_alltrials=np.concatenate([tt*np.ones(f[lin_ref].shape[0]) for tt,lin_ref in enumerate(np.squeeze(lin_all_ref))]) # 000001111...nnn

    if do_mask_transient:
        dfot_masked = dfot_alltrials_all_cells * transientmask_alltrials_all_cells
    else:
        dfot_masked = dfot_alltrials_all_cells
    spk_beh_df = pd.DataFrame(dfot_masked).T
    spk_beh_df['lin'] = lin_alltrials
    spk_beh_df['directed_locomotion']=moving_alltrials
    spk_beh_df['index_within'] = trial_index_within_alltrials.astype(int) # trial index within context

    day = sess_ind // 2
    isnovel = sess_ind % 2 # 0 fam, 1 nov
    spk_beh_df['isnovel'] = isnovel

    label_ascii = np.squeeze(one_exp['label'])
    label = turn_ascii_array_into_str(label_ascii)

    animal=str(int(np.squeeze(one_exp['animal'])))

    cell_cols_d = {'pyr':np.arange(n_uid)}
    spk_beh_df['trial_type'] = spk_beh_df['isnovel'] # add trial_type to make compatible with fr_map; other functions might not be compatible since they usually require trialtype to be a tuple; worry about it later
    spk_beh_df['task_index']=0

    prep_res={'spk_beh_df':spk_beh_df,
    'place_field':place_field_all_cells,
    'fr_map_trial':dfoy_alltrials_all_cells,
    'animal':animal,
    'label':label,
    'day':day,
    'isnovel':isnovel,
    'trial_name':trial_name,
    'cell_cols_d':cell_cols_d
    }

    return prep_res

def sort_trials_from_trialnames(trial_name_l):
    '''
    trial_name_l: a list of two lists, each from CA1{1}.metadata.categories{2}.filename
    for some sessions:
        date_animal_{x}c_f{n}, need to concatenate fam and novel together, and sort to get the true order

    '''
    n_ep = len(trial_name_l)
    trial_name_l_concat=np.concatenate(trial_name_l)
    trial_name_df=pd.Series(trial_name_l_concat).to_frame(name='trial_name')
    index_within_l =np.concatenate([np.arange(len(trial_name_l[k])) for k in range(n_ep)])
    isnovel_l = np.concatenate([k*np.ones(len(trial_name_l[k])) for k in range(n_ep)])
    trial_name_df['index_within'] = index_within_l
    trial_name_df['isnovel'] = isnovel_l.astype(int)
    trial_name_df = trial_name_df.sort_values('trial_name').reset_index(drop=True).reset_index().rename({'index':'trial_index'},axis=1)
    index_within_to_trial_index_df = trial_name_df.set_index(['isnovel','index_within'])['trial_index'].sort_index()
    trial_index_to_index_within_df = trial_name_df.set_index(['trial_index'])[['isnovel','index_within']].sort_index()

    return trial_index_to_index_within_df, index_within_to_trial_index_df,trial_name_df


def prep_one_day_within_one_exp(f,exp_ind,day_ind,region='CA1',do_mask_transient=True):
    sess_ind_l = [day_ind * 2,day_ind * 2+1]
    spk_beh_df_l = []
    trial_name_l = []
    gpb_l=[]
    for sess_ind in sess_ind_l:
        try:
            prep_res = prep_one_category_within_one_exp(f,exp_ind,sess_ind,region=region,do_mask_transient=do_mask_transient)
            spk_beh_df = prep_res['spk_beh_df']
            trial_name =  prep_res['trial_name']
            cell_cols_d = prep_res['cell_cols_d'] # assuming two categories within a day will have the same cell_cols_d
            spk_beh_df_l.append(spk_beh_df)
            trial_name_l.append(trial_name)
            gpb = spk_beh_df.groupby('index_within')
            gpb_l.append(gpb)
        except:
            pass
    

    trial_index_to_index_within_df, index_within_to_trial_index_df,trial_name_df = sort_trials_from_trialnames(trial_name_l)

    spk_beh_df_concat = []
    for trial_index,row in trial_index_to_index_within_df.iterrows():
        isnovel=row['isnovel']
        index_within=row['index_within']
        spk_beh_df_one_trial=gpb_l[int(isnovel)].get_group(index_within)
        spk_beh_df_one_trial['trial'] = trial_index
        spk_beh_df_concat.append(spk_beh_df_one_trial)
    spk_beh_df_concat = pd.concat(spk_beh_df_concat,axis=0)
    spk_beh_df_concat = spk_beh_df_concat.reset_index(drop=True)

    

    prep_res = {
        'spk_beh_df':spk_beh_df_concat,
        'trial_index_to_index_within_df':trial_index_to_index_within_df,
        'index_within_to_trial_index_df':index_within_to_trial_index_df,
        'trial_name_df':trial_name_df,
        'cell_cols_d':cell_cols_d,
    }

    return prep_res

# get baseline F0
# consider move to main code in prep_thomas_data
def get_F0_one_exp_one_cat(f,exp_ind=0,day_ind=0,famnov=0,region='CA1'):
    one_exp=f[f['data']['CA1'][exp_ind,0]]
    cells_oneexp=one_exp['cells']
    n_uid = one_exp['cells'].shape[0]
    sess_ind = day_ind * 2 + famnov
    F0_alltrials_allcells = []
    for uid in range(n_uid):
        one_cell = f[one_exp['cells'][uid,0]]
        one_cell_cat = f[one_cell['categories'][sess_ind,0]]
        F0_alltrials=np.concatenate([f[f0] for f0 in np.squeeze(one_cell_cat['F0'])])
        F0_alltrials_allcells.append(np.squeeze(F0_alltrials))
    F0_alltrials_allcells = np.stack(F0_alltrials_allcells)
    return F0_alltrials_allcells
        
def get_roi_one_exp(f,exp_ind,region='CA1'):
    '''
    roi_com_all: n_cell x 2
    '''
    one_exp=f[f['data']['CA1'][exp_ind,0]]
    cells_oneexp=one_exp['cells']
    n_uid = one_exp['cells'].shape[0]
    roi_com_all = []
    for uid in range(n_uid):
        roi=f[cells_oneexp[uid,0]]['roi']
        mnCooordinates=roi['mnCoordinates']
        # f[mnCooordinates]
        mnCooordinates_com=np.array(mnCooordinates).mean(axis=1)
        roi_com_all.append(mnCooordinates_com)
    roi_com_all = np.array(roi_com_all)
    return roi_com_all