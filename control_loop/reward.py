# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import math as mt


class Reward(object):
    
    def __init__(self):
        self.reward = 0
        
    def get_reward(self, x):
        x = 1000*x
        if x < 0.3:
            self.reward = 2/(1+mt.exp(-20*(x-0.20))) - 1
        else:    
            self.reward = -2/(1+mt.exp(-20*(x-0.9))) + 1
        return self.reward
    
import reward as rw

plt.close('all')

r = rw.Reward()

x = np.linspace(0.00001, 0.002, 100)
y = np.zeros(len(x))
for i in np.arange(len(x)):
    y[i] = r.get_reward(x[i])

plt.plot(x,y)
plt.grid()
plt.show()