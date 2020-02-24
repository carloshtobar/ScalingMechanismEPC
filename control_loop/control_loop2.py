# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 13:30:48 2017

MMC model of EPC
C = NUM_NSC (Number of NSCs)

Version final utilizada en el artículo

Este archivo usa Q-learning sólo

@author: Carlos Hernán Tobar Arteaga
"""

import random
import numpy as np
import math as mt
import simpy.rt
import matplotlib.pyplot as plt
import scipy.stats as st
import policy as pol
import q_function as q_function

plt.close('all')

RANDOM_SEED = 33

Ts = []
Tw = []
Tr = []
Tr_average = []
R_average = []
C_average = []
t_average = []
w_average = []
actions = []

t = []
w = []
Tr_teorico = []
np.inf

gamma = 0.8   # 0.8
alfa = 0.1   #0.2

policy = pol.Policy()
q = q_function.Q_function(gamma, alfa)


class Monitor(object):
    def __init__(self):
        self.Tr_average = []        
    def get_Tr_average(self): 
        mrt = sum(self.Tr_average)/len(self.Tr_average)
        
        return mrt
    def collect(self, tr):
        self.Tr_average.append(tr)
    def reset(self):        
        self.Tr_average = []
        
monitor = Monitor()

class Front_End(object):
    
    def __init__(self, env, u_fe):
        self.env = env
        self.u_fe = u_fe
        self.queue = simpy.Resource(env, 1) 
               
    def request(self):
        yield self.env.timeout(random.expovariate(self.u_fe))

class Service_Layer(object):
    
    def __init__(self, env, num_instances, u_sl):
        self.num_instances = num_instances
        self.env = env
        self.u_sl = u_sl
        self.queue = simpy.Resource(env, num_instances) 
        
    def modify_queue(self, num_instances):
        self.num_instances = num_instances
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
    monitor.collect(response_time)

SIM_TIME = 24        
sl_t = 0        # Number of NSCs
sl_tp1 = 0
tr_t = 0
tr_tp1 = 0
a_t = 0
workload_t = 0
num_sl = 4  # valor inicial
u_fe = 120000  # packets per second
u_sl = 10167   # control messages per second
u_db = 100000 # transactions per second
u_oi = 5000000 # packets per second
lmbda = 814

def get_workload(tiempo):
    #sigma = 1.3
    m = 130.0*(173.29 + 89.83*np.sin(np.pi/12*tiempo+3.08) + 52.6*np.sin(np.pi/6*tiempo+2.08) + 16.68*np.sin(np.pi/4*tiempo+1.13))
    #u_t = np.log10(m) - 0.5*sigma**2
    #workload = m + st.lognorm.rvs(s=sigma, scale=u_t)
    return m


def get_reward(sl_prima, tr_prima, workload):
    mu = 10167.0
    rho = workload/(sl_prima*mu)
    reward = 0
    if sl_prima == 1:
        if tr_prima < 0.001:
            reward = 1
        else:
            reward = -1
    elif sl_prima == 2:
        if tr_prima >= 0.001:
            reward = -1
        else:
            if rho >= 0.45 and rho < 1:
                reward = 1
            else:
                reward = -1
    elif sl_prima == 3:
        if tr_prima >= 0.001:
            reward = -1
        else:
            if rho >= 0.65 and rho < 1:
                reward = 1
            else:
                reward = -1
    elif sl_prima == 4:
        if rho >= 0.7:
            reward = 1
        else:
            reward = -1 
    return reward        
    
def get_reward7(sl_prima, tr_prima, workload):
    mu = 10167.0
    reward = 0
    if sl_prima == 4:
        if workload >= 0.9*(sl_prima-1)*mu:
            reward = 1
        else:
            reward = -1
    else:
        
        if workload >= 0.9*(sl_prima-1)*mu and workload < 0.9*sl_prima*mu:
            reward = 1
        else:
            reward = -1  
    return reward        

def get_reward2(sl_prima, tr_prima, workload):
    print 'w:', workload
    reward = (0.001 - tr_prima)*(1/sl_prima)
    
    
        
    print 'sl_prima, tr_prima, workload, reward: ', sl_prima, tr_prima, workload, reward
    return reward 
    

def setup(env, lmbda):
    front_end = Front_End(env, u_fe)
    service_layer = Service_Layer(env, num_sl, u_sl)
    data_base = Data_Base(env, u_db)
    output_interface = Output_Interface(env, u_oi)
    delay = Delay(env, 0.0)
    
    
    
    delta = 1.0/12.0  # Cada 10 minutos  # 1.0 Cada hora (1/6)
    flag = delta  
    i = delta
    
    while True:    
        tiempo = env.now
        #print 't:', tiempo
        t.append(tiempo)
        workload = get_workload(tiempo)
        
        if tiempo >= i and flag == i:
            #print 't:', tiempo
            
            if flag == delta:
                tr_t = monitor.get_Tr_average() 
                Tr_average.append(tr_t)
                monitor.reset()
                sl_t = service_layer.num_instances
                a_t = 4
                service_layer.modify_queue(a_t)
                
                workload_t = workload
                print 't-->, tr_t, sl_t, a_t:', tr_t, sl_t, a_t
                #print 'action = ', a_t
            else:
                tr_tp1 = monitor.get_Tr_average()
                Tr_average.append(tr_t)
                monitor.reset()
                sl_tp1 = service_layer.num_instances
                                
                r = get_reward(sl_tp1, tr_tp1, workload)
                R_average.append(r)
                q.update_Q(sl_t, a_t, sl_tp1, r)
                print 't, tr, sl, r:', tiempo, tr_t, sl_t, r 
                
                sl_t = sl_tp1
                action = policy.get_policy(q.Q, sl_t) 
                service_layer.modify_queue(action)
                
                
                a_t = action
                tr_t = tr_tp1
                workload_t = workload
                #print 't-->, tr_t, sl_t, a_t:', tr_t, sl_t, a_t          
            flag = i + delta
            i = i + delta 
                
                
            
            t_average.append(tiempo)
            actions.append(sl_t)#sl_t
            w_average.append(workload)
            
            
            
        
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


# Gráficas
fig = plt.figure()


plt.xlabel('Time (hours)') 
plt.plot(t_average, actions, 'b-', linewidth=1.1) 
plt.ylabel('Number of SL instances') 

#plt.ylim(0.0, 0.0006)

#plt.annotate('One instance', xy = (15,0.00005), xycoords = 'data', xytext = (5, 0.0001), textcoords = 'data', arrowprops = dict(arrowstyle = "->"))
#plt.annotate('Two instances', xy = (40, 0.00055), xycoords = 'data', xytext = (52, 0.0005), textcoords = 'data', arrowprops = dict(arrowstyle = "->"))
#plt.annotate('One instance', xy = (55,0.00005), xycoords = 'data', xytext = (60, 0.0001), textcoords = 'data', arrowprops = dict(arrowstyle = "->"))

plt.twinx() 
plt.semilogy(t_average, Tr_average, 'go-', linewidth=0.9) 
plt.ylabel('Mean response time (seconds)') 
#plt.ylim(0.9, 2.1)

plt.grid() 

plt.figure()
plt.plot(t_average, w_average)

#plt.figure()
#plt.semilogy(t_average, C_average)
#plt.grid()

plt.figure()
plt.plot(R_average, 'mo-')
plt.title('Reward')
plt.grid()

import os
try:
    os.remove('results-caso1.txt')
except Exception:
    print 'No existe el archivo'    

archivo = open("results-caso1.txt", 'a')
for i in np.arange(len(t_average)):
    line = str(t_average[i])+'\t'+str(Tr_average[i])+'\t'+str(actions[i])+'\n'
    archivo.write(line)
archivo.close()
