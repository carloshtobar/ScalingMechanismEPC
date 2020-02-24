# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

plt.close('all')

def get_workload(tiempo):
    #sigma = 1.3
    
    #factor = 40.0  # 1 instancia
    #factor = 80.0 # 2 instancias
    #factor = 120.0 # 3 instancias
    factor = 160.0  # para 4 instancias
    
    m = factor*(173.29 + 89.83*np.sin(np.pi/12*tiempo+3.08) + 52.6*np.sin(np.pi/6*tiempo+2.08) + 16.68*np.sin(np.pi/4*tiempo+1.13))
    #u_t = np.log10(m) - 0.5*sigma**2
    #workload = st.lognorm.rvs(s=sigma, scale=u_t)
    return m

t = np.linspace(0, 24, 200)
w = np.zeros(len(t))
for i in np.arange(len(t)):
    w[i] = get_workload(t[i])
    
plt.plot(t,w)
plt.grid()
    