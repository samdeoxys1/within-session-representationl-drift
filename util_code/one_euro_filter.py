import numpy as np
import pandas as pd

class OneEuroFilter:
    def __init__(self, min_cutoff=0.2, beta=0.01, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.timestamp_prev = None

    def alpha(self, cutoff,dt):
        r = 2 * np.pi * cutoff*dt
        return r / (r + 1.0)

    def apply(self, x, timestamp):
        if self.x_prev is None:
            result = x
        else:
            dt = timestamp - self.timestamp_prev
            alpha_x = self.alpha(self.min_cutoff + self.beta * np.abs(self.dx_prev),dt)
            result = alpha_x * x + (1 - alpha_x) * self.x_prev

        if self.dx_prev is None:
            dx = 0
        else:
            alpha_dx = self.alpha(self.d_cutoff,dt)
            dx = (result - self.x_prev) / dt
            dx = alpha_dx * dx + (1 - alpha_dx) * self.dx_prev

        self.x_prev = result
        self.dx_prev = dx
        self.timestamp_prev = timestamp

        return result

def get_speed_one_euro_filter(xy,timestamps,min_cutoff=0.05,beta=0.2,d_cutoff=1.):
    
    oef_x = OneEuroFilter(min_cutoff=min_cutoff,beta=beta,d_cutoff=d_cutoff)
    oef_y = OneEuroFilter(min_cutoff=min_cutoff,beta=beta,d_cutoff=d_cutoff)
    ntimes=xy.shape[0]

    xy_smth = []
    for i in range(ntimes):
        x,y,t = xy[i,0],xy[i,1],timestamps[i]
        x_smth=oef_x.apply(x,t)
        y_smth=oef_y.apply(y,t)
        xy_smth.append([x_smth,y_smth])
    xy_smth = np.array(xy_smth)   
    
    dt = np.median(np.diff(timestamps))
    vx = np.gradient(xy_smth[:,0]) / dt 
    vy = np.gradient(xy_smth[:,1]) / dt

    v = np.stack([vx,vy],axis=1)

    return xy_smth, v