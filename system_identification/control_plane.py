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
import scipy.stats as st

plt.close('all')

RANDOM_SEED = 31

Ts = []
Tw = []
Tr = []
t = []
w = []
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
        
    def modify_queue(self, num_instances):
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

SIM_TIME = 24        
num_sl = 4         # Number of NSCs
u_fe = 120000  # packets per second
u_sl = 10167   # control messages per second
u_db = 100000 # transactions per second
u_oi = 5000000 # packets per second
lmbda = 814

def get_workload(tiempo):
    #sigma = 1.3
    #factor = 40.0   # 1 instancia
    #factor = 80 # 2 instancias
    #factor = 120.0 # 3 instancias
    factor = 160.0 # 4 instancias
    
    m = factor*(173.29 + 89.83*np.sin(np.pi/12*tiempo+3.08) + 52.6*np.sin(np.pi/6*tiempo+2.08) + 16.68*np.sin(np.pi/4*tiempo+1.13))
    #m = factor*(173.29 + 89.83*np.sin(np.pi/12*tiempo+3.08))
    #u_t = np.log10(m) - 0.5*sigma**2
    #workload = m + st.lognorm.rvs(s=sigma, scale=u_t)
    return m

   
    
    
    

def setup(env, lmbda):
    front_end = Front_End(env, u_fe)
    service_layer = Service_Layer(env, num_sl, u_sl)
    data_base = Data_Base(env, u_db)
    output_interface = Output_Interface(env, u_oi)
    delay = Delay(env, 0.0)
    
    while True:          
        tiempo = env.now
        print 't:', tiempo
        
        t.append(tiempo)
                
                
        workload = get_workload(tiempo)
        print 'workload', workload
        w.append(workload)
        yield env.timeout(random.expovariate(workload))   # tiempo de arrivo total
        #yield env.timeout(1/LMBDA)
        env.process(nas_event(env, front_end, service_layer, data_base, output_interface, delay))        

# Setup and start the simulation
print 'EPC Control Plane. Overall simulation time: ', SIM_TIME
random.seed(RANDOM_SEED)  
env = simpy.rt.RealtimeEnvironment(factor=0.000001, strict=False)
env.process(setup(env, num_sl))

# Execute!
env.run(until=SIM_TIME)

plt.figure()
plt.plot(t,w)
plt.figure()
plt.semilogy(t[0:len(Tr)],Tr)

Tr_average = []
t_average = []
w_average = []
delta = len(t)/(2*360)      #360
for i in np.arange(2*360):
    Tr_average.append(sum(Tr[i*delta:(i+1)*delta])/delta)
    t_average.append(t[i*delta])

# Workload
for i in np.arange(len(t_average)):
    w_average.append(get_workload(t_average[i]))
    
plt.figure()
plt.semilogy(t_average, Tr_average, '.-')
#plt.subplot(2,1,2)
#plt.semilogy(Tr_teorico, '+--') 

plt.figure()
plt.plot(t_average, w_average)

# Guardo los datos en un archivo
archivo = open("sys-data-4sl.txt", 'a')
for i in np.arange(len(t_average)):
    line = str(t_average[i])+'\t'+str(w_average[i])+'\t'+str(Tr_average[i])+'\n'
    archivo.write(line)
archivo.close()
