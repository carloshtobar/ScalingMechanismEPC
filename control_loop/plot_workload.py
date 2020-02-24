# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as pl

pl.close('all')

def get_workload(tiempo):
    sigma = 0.005
    m = 130.0*(173.29 + 89.83*np.sin(np.pi/12*tiempo+3.08) + 52.6*np.sin(np.pi/6*tiempo+2.08) + 16.68*np.sin(np.pi/4*tiempo+1.13))
    u_t = np.log10(m) #- 0.5*sigma**2
    workload = 1.*st.lognorm.rvs(s=sigma, scale=m)
    return workload


t = np.linspace(0, 24, 100)
w = get_workload(t)

pl.plot(t,w)