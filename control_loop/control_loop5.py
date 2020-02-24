# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 13:30:48 2017

MMC model of EPC
C = NUM_NSC (Number of NSCs)

Version final utilizada en el artículo

Este archivo usa Q-learning solo con el modelo

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
import system_model as sm

plt.close('all')

RANDOM_SEED = 31

Ts = []
Tw = []
Tr = []
Tr_average = []
C_average = []
Tr_predicted = []
t_average = []
w_average = []
actions = []
num_iter_training = []

t = []
w = []
Tr_teorico = []
np.inf

gamma = 0.8    # 0.8
alfa = 0.1    #0.2

policy = pol.Policy()
q = q_function.Q_function(gamma, alfa)
system_model = sm.System_Model()
system_model.create_model_sl1()
system_model.create_model_sl2()
system_model.create_model_sl3()
system_model.create_model_sl4()

q_previo = np.zeros(16)
q_actual = np.zeros(16)

def set_Q_previo(q):
    q_previo[0] = q[0]
    q_previo[1] = q[1]
    q_previo[2] = q[2]
    q_previo[3] = q[3]
    q_previo[4] = q[4]
    q_previo[5] = q[5]
    q_previo[6] = q[6]
    q_previo[7] = q[7]
    q_previo[8] = q[8]
    q_previo[9] = q[9]
    q_previo[10] = q[10]
    q_previo[11] = q[11]
    q_previo[12] = q[12]
    q_previo[13] = q[13]
    q_previo[14] = q[14]
    q_previo[15] = q[15]
    
def set_Q_actual(q):
    q_actual[0] = q[0]
    q_actual[1] = q[1]
    q_actual[2] = q[2]
    q_actual[3] = q[3]
    q_actual[4] = q[4]
    q_actual[5] = q[5]
    q_actual[6] = q[6]
    q_actual[7] = q[7]
    q_actual[8] = q[8]
    q_actual[9] = q[9]
    q_actual[10] = q[10]
    q_actual[11] = q[11]
    q_actual[12] = q[12]
    q_actual[13] = q[13]
    q_actual[14] = q[14]
    q_actual[15] = q[15]

def get_mse(x, y):    
    s = 0.0    
    for i in np.arange(len(x)):        
        s = s + (x[i]-y[i])**2
    return s/len(x) 

class Monitor(object):
    def __init__(self):
        self.Tr_average = []     
        self.C_average = []
    def get_Tr_average(self, num_sl, workload): 
        mrt = sum(self.Tr_average)/len(self.Tr_average)
        c = (mrt-1/10167.0)*(10167.0*num_sl-workload)
        if c < 0.0 or c > 1.0:
            c = 1.0
        return mrt, c
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
        self.queue = simpy.Resource(env, self.num_instances) 
        
    def modify_queue(self, num_instances):
        self.num_instances = num_instances
        self.queue = simpy.Resource(env, self.num_instances)  
                
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

SIM_TIME =24        
sl_t = 0        # Number of NSCs
sl_tp1 = 0
tr_t = 0
tr_tp1 = 0
a_t = 0
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
#    sigma = 0.005
#    m = 130.0*(173.29 + 89.83*np.sin(np.pi/12*tiempo+3.08) + 52.6*np.sin(np.pi/6*tiempo+2.08) + 16.68*np.sin(np.pi/4*tiempo+1.13))
#    u_t = np.log10(m) #- 0.5*sigma**2
#    workload = 1.*st.lognorm.rvs(s=sigma, scale=m)
#    return workload


def setup(env, lmbda):
    front_end = Front_End(env, u_fe)
    service_layer = Service_Layer(env, num_sl, u_sl)
    data_base = Data_Base(env, u_db)
    output_interface = Output_Interface(env, u_oi)
    delay = Delay(env, 0.0)
    
    
    
    delta = 1.0/6.0  # Cada 10 minutos (1/6)  # 1.0 Cada hora
    flag = delta  
    i = delta
    
    while True:    
        tiempo = env.now
        #print 't:', tiempo
        t.append(tiempo)
        workload = get_workload(tiempo)
        
        
        
        
        if tiempo >= i and flag == i:
            print 't:', tiempo
            
            if flag == delta:
                tr_t, c = monitor.get_Tr_average(service_layer.num_instances, workload) 
                C_average.append(c)
                Tr_average.append(tr_t)
                monitor.reset()
                #sl_t = 2
                #a_t = 2
                tr_predicted = system_model.predict_tr(service_layer.num_instances, workload)
                Tr_predicted.append(tr_predicted)
                num_iter_training.append(0)
                #print 'tr_t, sl_t, a_t:', tr_t, sl_t, a_t
                #print 'action = ', a_t
            else:
                
                
                
                num_train = 0
                error = 100
                threshold = .000001
                
                sl_t = service_layer.num_instances
                
                while error > threshold:
                    set_Q_previo(q.Q)
                    #print 'sl_t: ', sl_t
                    a2 = policy.get_policy(q.Q, sl_t)
                    #a2 = np.random.randint(1,3,1)
                    sl_prima, tr_prima = system_model.get_next_states(a2, workload)     
                    #print 'sl_prima, tr_prima: ', sl_prima, tr_prima                    
                    r = system_model.get_reward()
                    
                    q.update_Q(sl_t, a2, sl_prima, r)
                    set_Q_actual(q.Q)
                    error = get_mse(q_previo, q_actual)
                    #print 'error:', error
                    num_train += 1
                    sl_t = a2
                print 'Q: ', q.Q
                print 'num train', num_train
                num_iter_training.append(num_train)
                
                #action = policy.get_policy(q.Q, service_layer.num_instances)
                #sl_t = a2
                action = policy.get_policy(q.Q, sl_t)
                
                tr_tp1, c = monitor.get_Tr_average(service_layer.num_instances, workload)
                Tr_average.append(tr_tp1)
                C_average.append(c)
                monitor.reset()
                
                tr_predicted = system_model.predict_tr(service_layer.num_instances, workload)
                Tr_predicted.append(tr_predicted)
                #sl_tp1 = a_t
                
                #action = policy.get_policy(q.Q, sl_t) 
                
                #r = get_reward(sl_tp1, tr_tp1)
                #q.update_Q(sl_t, action, sl_tp1, r)
                
                service_layer.modify_queue(action)
                
                #sl_t = a2
                #a_t = action
                
                
                #print 'action = ', a_t
                tr_t = tr_tp1
                           
            flag = i + delta
            i = i + delta 
                
                
            
            t_average.append(tiempo)
            actions.append(service_layer.num_instances)
            w_average.append(workload)
            
            
            
        
        w.append(workload)
        yield env.timeout(random.expovariate(workload))   # tiempo de arrivo total
        #yield env.timeout(1/LMBDA)
        env.process(nas_event(env, front_end, service_layer, data_base, output_interface, delay))        

# Setup and start the simulation
print 'EPC Control Plane. Overall simulation time: ', SIM_TIME
random.seed(RANDOM_SEED)  
env = simpy.rt.RealtimeEnvironment(factor=0.001, strict=False)

env.process(setup(env, num_sl))

# Execute!
env.run(until=SIM_TIME)


# Gráficas
fig = plt.figure()

plt.semilogy(t_average, Tr_average, 'g-', linewidth=0.9) 
plt.semilogy(t_average, Tr_predicted, 'r.-', linewidth=0.9)
plt.xlabel('Time (hours)') 
plt.ylabel('Mean response time (seconds)') 
#plt.ylim(0.0, 0.0006)

#plt.annotate('One instance', xy = (15,0.00005), xycoords = 'data', xytext = (5, 0.0001), textcoords = 'data', arrowprops = dict(arrowstyle = "->"))
#plt.annotate('Two instances', xy = (40, 0.00055), xycoords = 'data', xytext = (52, 0.0005), textcoords = 'data', arrowprops = dict(arrowstyle = "->"))
#plt.annotate('One instance', xy = (55,0.00005), xycoords = 'data', xytext = (60, 0.0001), textcoords = 'data', arrowprops = dict(arrowstyle = "->"))

plt.twinx() 
plt.plot(t_average, actions, 'b-', linewidth=1.1) 
plt.ylabel('Number of SL instances') 
#plt.ylim(0.9, 2.1)

plt.grid() 

plt.figure()
plt.plot(t_average, w_average)

plt.figure()
plt.plot(t_average, num_iter_training)

import os
try:
    os.remove('results-caso3.txt')
except Exception:
    print 'No existe el archivo'    

archivo = open("results-caso3.txt", 'a')
for i in np.arange(len(t_average)):
    line = str(t_average[i])+'\t'+str(Tr_average[i])+'\t'+str(Tr_predicted[i])+'\t'+str(actions[i])+'\t'+str(num_iter_training[i])+'\n'
    archivo.write(line)
archivo.close()
