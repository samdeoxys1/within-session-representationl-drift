import numpy as np
import scipy

# helper function for clustering
def merge_peaks(pdf_peaks,peak_dist_thresh=6):
    '''
    sequentially merge a monotonic 1d array, if two consecutive ones are smaller than a threshold
    also return the cluster assignment
    '''    
    peaks_left=[]
    cl = np.zeros_like(pdf_peaks)
    i=0
    cl_ind=0
    if len(pdf_peaks)==1:
        return pdf_peaks,cl
    while i <(len(pdf_peaks)-1):
        peaks_to_be_merged=[pdf_peaks[i]]
        cl[i] = cl_ind
        j=0
        while ((pdf_peaks[i+j+1] - pdf_peaks[i+j])<=peak_dist_thresh):
            peaks_to_be_merged.append(pdf_peaks[i+j+1])
            cl[i+j+1] = cl_ind
            j+=1
            if (i+j+1)> len(pdf_peaks)-1:
                break
        peaks_left.append(np.mean(peaks_to_be_merged))
        cl_ind +=1
        
        i=i+j+1
    if cl[-1]==0: # if the last one is not merged, then need to assign the last cluster
        cl[-1]=cl_ind
        peaks_left.append(pdf_peaks[-1])
    return np.array(peaks_left),np.array(cl)




def merge_peaks_and_troughs(pdf_peaks,pdf_troughs,peak_dist_thresh=6):
    peaks_left=[]
    pdf_troughs = pdf_troughs.astype(float)
    cl = np.zeros_like(pdf_peaks)
    i=0
    cl_ind=0
    if len(pdf_peaks)==1:
        return pdf_peaks,cl
    while i <(len(pdf_peaks)-1):
        peaks_to_be_merged=[pdf_peaks[i]]
        cl[i] = cl_ind
        j=0
        while ((pdf_peaks[i+j+1] - pdf_peaks[i+j])<=peak_dist_thresh):
            peaks_to_be_merged.append(pdf_peaks[i+j+1])
            pdf_troughs[i+j] = np.nan
            cl[i+j+1] = cl_ind
            j+=1
            if (i+j+1)> len(pdf_peaks)-1:
                break
        peaks_left.append(np.mean(peaks_to_be_merged))
        cl_ind +=1
        
        i=i+j+1
    if cl[-1]==0: # if the last one is not merged, then need to assign the last cluster
        cl[-1]=cl_ind
        peaks_left.append(pdf_peaks[-1])
    pdf_troughs = pdf_troughs[np.logical_not(np.isnan(pdf_troughs))]
    return np.array(peaks_left),np.array(cl),pdf_troughs    


# mode clustering using kde
class Kde_Peak_Cluster():
    def __init__(self,**kwargs):
        self.bw_method=kwargs['bw_method']
        self.allposbins=kwargs['allposbins']
        self.peak_dist_thresh=kwargs['peak_dist_thresh'] # can set to really high such that no merging
    def fit_predict_with_pdf(self,location_to_be_clustered):
        location_to_be_clustered = np.squeeze(location_to_be_clustered)
        try:
            kernel = scipy.stats.gaussian_kde(location_to_be_clustered,bw_method=self.bw_method)
            pdf = kernel(self.allposbins)
            pdf_peaks = self.allposbins[scipy.signal.find_peaks(pdf)[0]]
            pdf_troughs = self.allposbins[scipy.signal.find_peaks(-pdf)[0]]
            if (self.peak_dist_thresh is not None) and (self.peak_dist_thresh < 100):
                
                pdf_troughs = merge_peaks_and_troughs(pdf_peaks,pdf_troughs,peak_dist_thresh=self.peak_dist_thresh)[-1]
            # old way distance to peak
            # dist_to_peak = np.abs(np.subtract.outer(location_to_be_clustered,pdf_peaks))
            # cl = np.argmin(dist_to_peak,axis=1)
            
            # new way: mode clustering
            n_clusters = len(pdf_troughs) + 1
            pdf_troughs_extended=np.array([self.allposbins[0],*pdf_troughs,self.allposbins[-1]])
            x=location_to_be_clustered
            condlist = [(x>=pdf_troughs_extended[i]) & (x<=pdf_troughs_extended[i+1]) for i in range(len(pdf_troughs_extended)-1)]
            funclist = np.arange(n_clusters)
            cl = np.piecewise(x,condlist,funclist)
            

        except np.linalg.LinAlgError:
            
            cl = np.zeros_like(location_to_be_clustered)
            pdf=None
        return cl,pdf
    def fit_predict(self,location_to_be_clustered):
        cl,pdf=self.fit_predict_with_pdf(location_to_be_clustered)
        return cl
        