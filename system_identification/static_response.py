# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 13:30:48 2017

MMC model of EPC
C = NUM_NSC (Number of NSCs)

Version final utilizada en el artículo

@author: Carlos Hernán Tobar Arteaga
"""

import random
import numpy as np
import math as mt
import simpy.rt
import matplotlib.pyplot as plt

plt.close('all')

RANDOM_SEED = 33

Ts = []
Tw = []
Tr = []
Tr_teorico = []
np.inf
class Front_End(object):
    
    def __init__(self, env, u_fe):
        self.env = env
        self.u_fe = u_fe
        self.queue = simpy.Resource(env, 1) 
               
    def request(self):
        yield self.env.timeout(random.expovariate(self.u_fe))

class Service_Layer(object):
    
    def __init__(self, env, num_instances, u_sl):
        self.env = env
        self.u_sl = u_sl
        self.queue = simpy.Resource(env, num_instances) 
               
    def request(self):
        yield self.env.timeout(random.expovariate(self.u_sl))

class Data_Base(object):
    
    def __init__(self, env, u_db):
        self.env = env
        self.u_db = u_db
        self.queue = simpy.Resource(env, 1) 
               
    def request(self):
        yield self.env.timeout(random.expovariate(self.u_db))
        
class Output_Interface(object):
    
    def __init__(self, env, u_oi):
        self.env = env
        self.u_oi = u_oi
        self.queue = simpy.Resource(env, 1) 
               
    def request(self):
        yield self.env.timeout(random.expovariate(self.u_oi))
        
class Delay(object):
    
    def __init__(self, env, time):
        self.env = env
        self.time = time
        self.queue = simpy.Resource(env, np.Inf) 
               
    def request(self):
        yield self.env.timeout(self.time)
            
def nas_event(env, front_end, service_layer, data_base, output_interface, delay):
    arrival_time = env.now    
    with front_end.queue.request() as request: 
        yield request     
        initial_processing_time = env.now
        #print 'time: ', env.now
        yield env.process(front_end.request()) 
        
    with service_layer.queue.request() as request:
        yield request
        yield env.process(service_layer.request())
        
    with data_base.queue.request() as request:
        yield request
        yield env.process(data_base.request())
    
    with output_interface.queue.request() as request:
        yield request
        yield env.process(output_interface.request())
    
    with delay.queue.request() as request:
        yield request
        yield env.process(delay.request())
        
    final_processing_time = env.now
    
    service_time = final_processing_time - initial_processing_time
    waiting_time = initial_processing_time - arrival_time
    response_time = service_time + waiting_time
    Ts.append(service_time)
    Tw.append(waiting_time)
    Tr.append(response_time)

SIM_TIME = 1.0        
num_sl = 1         # Number of NSCs
u_fe = 120000  # packets per second
u_sl = 10167   # control messages per second
u_db = 100000 # transactions per second
u_oi = 5000000 # packets per second
lmbda = 814

def setup(env, lmbda):
    front_end = Front_End(env, u_fe)
    service_layer = Service_Layer(env, num_sl, u_sl)
    data_base = Data_Base(env, u_db)
    output_interface = Output_Interface(env, u_oi)
    delay = Delay(env, 0.0)
    
    while True:        
        yield env.timeout(random.expovariate(lmbda))   # tiempo de arrivo total
        #yield env.timeout(1/LMBDA)
        env.process(nas_event(env, front_end, service_layer, data_base, output_interface, delay))        

# Setup and start the simulation
print 'EPC Control Plane. Overall simulation time: ', SIM_TIME
random.seed(RANDOM_SEED)


numero_usuarios = np.arange(10000, 10000, 10000)
lmbda = np.linspace(100, 9400, 13)
Tr_average = np.zeros(len(lmbda))


for i in np.arange(len(lmbda)):
    env= simpy.rt.RealtimeEnvironment(factor=0.00001, strict=False)
    env.process(setup(env, lmbda[i]))
    env.run(until=SIM_TIME)
    tr_av = sum(Tr)/len(Tr)
    Tr_average[i] = tr_av
    print 'Tr_average = ', tr_av
    Tr = []

num_sl = 2
lmbda2 = np.linspace(100, 19700, 13)
Tr_average2 = np.zeros(len(lmbda2))
Tr = []

for i in np.arange(len(lmbda2)):
    env= simpy.rt.RealtimeEnvironment(factor=0.00001, strict=False)
    env.process(setup(env, lmbda2[i]))
    env.run(until=SIM_TIME)
    tr_av = sum(Tr)/len(Tr)
    Tr_average2[i] = tr_av
    print 'Tr_average= ', tr_av
    Tr = []    

num_sl = 3
lmbda3 = np.linspace(100, 30100, 13)
Tr_average3 = np.zeros(len(lmbda3))
Tr = []

for i in np.arange(len(lmbda3)):
    env= simpy.rt.RealtimeEnvironment(factor=0.00001, strict=False)
    env.process(setup(env, lmbda3[i]))
    env.run(until=SIM_TIME)
    tr_av = sum(Tr)/len(Tr)
    Tr_average3[i] = tr_av
    print 'Tr_average= ', tr_av
    Tr = []  

num_sl = 4
lmbda4 = np.linspace(100, 40100, 13)
Tr_average4 = np.zeros(len(lmbda4))
Tr = []

for i in np.arange(len(lmbda4)):
    env= simpy.rt.RealtimeEnvironment(factor=0.00001, strict=False)
    env.process(setup(env, lmbda4[i]))
    env.run(until=SIM_TIME)
    tr_av = sum(Tr)/len(Tr)
    Tr_average4[i] = tr_av
    print 'Tr_average= ', tr_av
    Tr = [] 


plt.figure()

f1, = plt.semilogy(lmbda, Tr_average, 'b^:', linewidth=0.9, markersize=3, label='MRT for one SL instance')
f2, = plt.semilogy(lmbda2, Tr_average2, 'rs:', linewidth=0.9, markersize=3, label='MRT for two SL instances')
f3, = plt.semilogy(lmbda3, Tr_average3, 'go:', linewidth=0.9, markersize=3, label='MRT for three SL instances')
f4, = plt.semilogy(lmbda4, Tr_average4, 'kx:', linewidth=0.9, markersize=3, label='MRT for four SL instances')

#plt.title('Mean Response Time Variarion in 24 hours')
plt.xlabel('Total arriving rate (control messages per second)')
plt.ylabel('Mean response time (seconds)')
plt.xlim(0, 40100.0)
plt.ylim(0.0001, 0.003)
plt.xticks(np.linspace(0, 40000, 5, endpoint=True))
plt.grid()
plt.legend(loc=2, handles=[f1, f2, f3, f4])
plt.show()

    