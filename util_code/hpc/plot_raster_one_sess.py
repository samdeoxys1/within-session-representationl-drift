import sys,os
sys.path.append('../')
import plot_raster as pr 
import pandas as pd

DATABASE_LOC = '/mnt/home/szheng/ceph/ad/database.csv'
db = pd.read_csv(DATABASE_LOC,index_col=[0,1])

# sess_to_plot = db.query("owner=='marisol'")
sess_to_plot = db.query("owner=='roman'|owner=='ipshita'")
print(f'{sess_to_plot.shape[0]} sessions!')

def main(i,testmode=False):
    sess_name = db.iloc[i]['sess_name']
    # args = pr.preprocess_for_plot(sess_name,sigma=30,speedmask=5,n_pos_bins=100)
    args = pr.preprocess_for_plot(sess_name,sigma=30,speedmask=1,n_pos_bins=100)
    cell_metric=args[0]
    nunits=len(cell_metric['UID'])
    if testmode:
        upper = 2
    else:
        upper = nunits+1
    for uid in range(1,upper):
        pr.plot_one_cell(uid,*args,savefig=True,sess_name=sess_name)

if __name__ == '__main__':
    args = sys.argv[1:]
    print(args)
    main(int(args[0]))
