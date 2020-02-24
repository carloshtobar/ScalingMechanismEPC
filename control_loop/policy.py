#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 09:19:46 2017

@author: iovisor
"""

import numpy as np

class Policy(object):
    
    def __init__(self):
        self.name = ''
        
    def find_max(self, x):    
        valor_max = np.max(x)
        pos_max = np.where(x == valor_max)
        if (len(pos_max[0]) == 1):
            return valor_max, np.asscalar(pos_max[0])
        else:
            return valor_max, np.asscalar(np.random.choice(pos_max[0]))

        
    def get_policy(self, q, k):
        #print q
        #print k
        if k==1:
            valor, pos = self.find_max(q[k-1:k+3])
            #print 'pos1: ',pos
            return pos+1
        elif k==2:
            valor, pos = self.find_max(q[k+2:k+6])
            #print 'pos2: ',pos
            return pos+1 
        elif k==3:
            valor, pos = self.find_max(q[k+5:k+9])
            #print 'pos3: ',pos
            return pos+1
        elif k==4:
            valor, pos = self.find_max(q[k+8:k+12])
            #print 'pos4: ',pos
            return pos+1
                
#import policy as p
##
#pol = p.Policy()
#print pol.get_policy(np.array([3,1,20,10,30,20,100,1,20,30,1,2,1,2,3,40]), 2)
        