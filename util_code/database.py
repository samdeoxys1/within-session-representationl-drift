import numpy as np
import scipy
import scipy.io
import matplotlib.pyplot as plt
import copy,sys,os,pdb,importlib
import pandas as pd
import data_prep_new as dpn
from pathlib import Path
sys.path.append('/mnt/home/szheng/projects/place_variability/code/thomas_18_data')
import prep_thomas_one_region as ptor

# DATABASE_LOC = '/mnt/home/szheng/ceph/ad/database.csv'


DATA_ROOT = '/mnt/home/szheng/ceph/ad'
DATABASE_LOC = os.path.join(DATA_ROOT,'database.csv')
db = pd.read_csv(DATABASE_LOC,index_col=[0,1])

def get_all_paths_info(data_root_dir,get_metadata=False,exclude_existing=False,**kwargs):
    '''
    data_root_dir can be any level of directory containing the .mat structs
    note that if the dir is at the level of animal name then, animal_name="" need to be provided
    return a df with animal name, data_dir_full, sess_name, date plus potentially metadata
    '''
    all_paths_l = []
    
    if exclude_existing:
        db = pd.read_csv(DATABASE_LOC,index_col=[0,1])
        toexclude_from = db['sess_name'].values
    for path in Path(data_root_dir).rglob('*Behavior.mat'):
        if ('Tracking' not in path.name) and ('Linearized' not in path.name):

            if len(list(path.parent.glob('*cell_metrics*.mat')))>0:
                sess_name=str(path.parent.name)
                ifadd = True
                if exclude_existing:
                    if sess_name in toexclude_from:
                        ifadd=False
                if ifadd:
                    all_paths_l.append({'data_dir_full':str(path.parent),'sess_name':sess_name,'animal_name':str(path.parent.parent.name)})
                    date=all_paths_l[-1]['sess_name'].split('_')[-1]
                    if 'sess' in date:
                        date = all_paths_l[-1]['sess_name'].split('_')[-2]
                    date=pd.to_datetime(date,yearfirst=True).date()
                    
                    all_paths_l[-1]['date'] = date
                    


                    if get_metadata:
                        meta_data = get_meta_data(all_paths_l[-1])
                        all_paths_l[-1].update(meta_data)
    if len(all_paths_l)==0:
        return pd.DataFrame([])
    all_paths_l = pd.DataFrame(all_paths_l)
    
    for key,val in kwargs.items():
        all_paths_l[key] = val
    all_paths_l = all_paths_l.groupby('animal_name').apply(lambda group:group.sort_values('date').reset_index(drop=True))
    
    return all_paths_l


def get_acc_per_session(sess_name,data_dir_full=None):
    sess_name=sess_name
#     data_dir=None
    data_dir_full=data_dir_full
    data_fn = f'{sess_name}.Behavior.mat'
    data = dpn.loadmat(os.path.join(data_dir_full,data_fn))['behavior']
    choice_l = data['trials']['choice']
    try:
        bad_trials_l = data['trials']['bad_trials']
        choice_l = choice_l[np.logical_not(bad_trials_l)]
    except:
        print('no bad_trials in behavior.trials')
    acc = np.nanmean(choice_l[1:])
    return acc

# add meta data to database entry
# row = all_paths_l_all_people.iloc[0]
import mat73
# DATABASE_LOC = '/mnt/home/szheng/ceph/ad/database.p'
# DATABASE_LOC = '/mnt/home/szheng/ceph/ad/database.csv'
def get_meta_data(row):

    to_load_path = os.path.join(row['data_dir_full'],f"{row['sess_name']}.Behavior.mat")
    try:
        mat=dpn.loadmat(to_load_path)['behavior']
    except:
        mat = mat73.loadmat(to_load_path)['behavior']

    meta_data = {}
    try:
        meta_data['behavior'] = mat['description']
    except:
        meta_data['behavior'] = None
    # behavior performance; now just a single number of accuracy; could use different later
    try:
        if 'alternation' in mat['description']:
            meta_data['performance']=get_acc_per_session(row['sess_name'],data_dir_full=row['data_dir_full'])

            # meta_data['performance'] = np.nanmean(mat['trials']['choice'][1:])
    except:
        meta_data['performance'] = None
    if meta_data['behavior'] is not None:
        meta_data['ntrials'] = (~np.isnan(mat['trials']['choice'])).sum() # excluding the first (all the nan)
        choice_l = mat['trials']['choice']
        try:
            bad_trials_l = mat['trials']['bad_trials']
            choice_l = choice_l[np.logical_not(bad_trials_l)]
        except:
            print(row['sess_name'],'\n')
            print('no bad_trials in behavior.trials')
        
        meta_data['ngoodtrials'] = (~np.isnan(choice_l)).sum() # excluding the first (all the nan)
    else:
        meta_data['ntrials'] = None
    to_load_path = os.path.join(row['data_dir_full'],f"{row['sess_name']}.cell_metrics.cellinfo.mat")
    try:
        mat=dpn.loadmat(to_load_path)['cell_metrics']
    except:
        mat=mat73.loadmat(to_load_path)['cell_metrics']

    meta_data['n_neurons'] = len(mat['spikes']['times']) # including both pyr and int
    meta_data['n_pyr_putative'] = np.sum(['Pyr' in x for x in mat['putativeCellType']]) # the actual used ones for analysis might be fewer

    return meta_data


def main(dosave=False,data_root_l = ['Chronic_H2','roman_data']):
    '''
    # Loading all folders from all people!
    data_root_l: the person-specific folder names
    need to check 'owner' asignment; ideally each person's data should be in the folder of {name}_data
    '''
    data_root_root = '/mnt/home/szheng/ceph/ad'
    get_metadata=True
    all_paths_l_all_people = []
    # for data_root in os.listdir(data_root_root):
    for data_root in data_root_l:
        data_root_full_dir = os.path.join(data_root_root,data_root)
        if os.path.isdir(data_root_full_dir):
            print(data_root_full_dir)
            all_paths_l = get_all_paths_info(data_root_full_dir,get_metadata=get_metadata)
            
            # need to update according to what names and people are in the ceph/ad folder
            
            if 'Chronic' in data_root:
                all_paths_l['owner']='marisol'
            else:
                owner_name = data_root.split('_')[0]
                all_paths_l['owner']=owner_name
            
            all_paths_l['data_root'] = data_root
            
            all_paths_l_all_people.append(all_paths_l)
    all_paths_l_all_people = pd.concat(all_paths_l_all_people,axis=0)

    database = all_paths_l_all_people
    # database.to_pickle(DATABASE_LOC)
    if dosave:
        database.to_csv(DATABASE_LOC)
        print(f'saved at {DATABASE_LOC}')
    return database

def update_db(new_data_root_dir,**kwargs):
    '''
    create a subdf from the data in new_data_root_dir and concat it to the original db and save
    currently works fine if new_data_root_dir contains two animals, doesn't work for one animal
    also haven't checked how duplicated rows are processed
    
    example kwargs:
    get_metadata=True,owner='ipshita',data_root ='ipshita_data'
    '''
    db_new_portion = get_all_paths_info(new_data_root_dir,**kwargs)
    db = pd.concat([db,db_new_portion],axis=0)
    db = db.groupby('animal_name').apply(lambda group:group.sort_values('date').reset_index(drop=True))
    db.to_csv(DATABASE_LOC)
    print(f'saved at {DATABASE_LOC}')
    return db, db_new_portion


thomas_180301_DG_CA3_CA1_dir = "/mnt/home/szheng/ceph/ad/thomas_data/180301_DG_CA3_CA1"
THOMAS_18_DATABASE_LOC = os.path.join(DATA_ROOT,'thomas_18_db.csv')
try:
    thomas_18_db = pd.read_csv(THOMAS_18_DATABASE_LOC,index_col=0).sort_values(['region','day_ind','exp_ind'])
except:
    pass
def get_thomas_180301_DG_CA3_CA1_db(dosave=False):
    thomas_db = []
    data_dir_full_l = []
    root = thomas_180301_DG_CA3_CA1_dir
    # for x in Path(root).rglob('preprocessed.p'):
    for x in Path(root).rglob(ptor.SAVE_FN):
        x_l = str(x).split('/')
        data_dir_full_l.append(str(x.parent))
        region, exp_ind, day_ind = x_l[-4:-1]
        exp_ind = exp_ind.split('_')[-1]
        day_ind = day_ind.split('_')[-1]
        thomas_db.append([region, exp_ind, day_ind])
    thomas_db = pd.DataFrame(thomas_db,columns=['region','exp_ind','day_ind'])
    thomas_db['data_dir_full']=data_dir_full_l

    if dosave:
        thomas_db.to_csv(THOMAS_18_DATABASE_LOC)


    return thomas_db
