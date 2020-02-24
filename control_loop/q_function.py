#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 09:35:45 2017

@author: iovisor
"""

import numpy as np

class Q_function(object):
    
    def __init__(self, gamma=0.8, alfa=0.2):
        self.gamma = gamma
        self.alfa = alfa
        self.s = np.array([1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4])
        self.a = np.array([1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4])
        self.Q = np.zeros(16) #0.001*np.random.rand(16)
        
    def max_Q(self, k_prima):
        if k_prima==1:
            return max(self.Q[0], self.Q[1], self.Q[2], self.Q[3])
        elif k_prima==2:
            return max(self.Q[4], self.Q[5], self.Q[6], self.Q[7])
        elif k_prima==3:
            return max(self.Q[8], self.Q[9], self.Q[10], self.Q[11])
        elif k_prima==4:
            return max(self.Q[12], self.Q[13], self.Q[14], self.Q[15])
                
        
 
    def get_Q(self, k, a):
        if k==1:
            return self.Q[a-1]
        elif k==2:
            return self.Q[a+3]
        elif k==3:
            return self.Q[a+7]
        elif k==4:
            return self.Q[a+11]
                    
    
    def update_Q(self, k, a, k_prima, r):
        #print 'k, a, k_prima, r: ', k, a , k_prima, r
        if k==1:
            i = a-1
        elif k==2:
            i = a+3
        elif k==3:
            i = a+7
        elif k==4:
            i = a+11       
        
        self.Q[i] = (1.0-self.alfa)*self.get_Q(k,a) + self.alfa*(r + self.gamma*self.max_Q(k_prima))
         
