#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 13:35:12 2017

@author: carlosh

Simulación final del modelado de la cadena de servicios como un sistema
dinámico used in the paper, con los datos generados por control_plane

Utilizo un modelo para cada K.

Pasos:
    1.  Static response of the system
    2.  Worload para 24 horas. 1 muestra por minuto
    3.  Mean response time for K=1 and K=2
    4.  Training samples. 60. Tomadas de forma regular del MRT
    5.  Validation samples. 100. Tomadas de forma regular del MRT
    6.  Random actions
    7.  MRT en t+1
    8.  Training
    9.  Validation

"""
import numpy as np
import matplotlib.pyplot as plt
import pyGPs as gp
import string
from cycler import cycler

np.random.seed(31)  #31
plt.close('all')

fig, ax0 = plt.subplots(ncols=1)
ax0.grid(color='#cfcfcf', linestyle=':', linewidth=1)
#ax0.grid(color='#cfcfcf', linestyle=':', linewidth=1)
#ax1.grid(color='#cfcfcf', linestyle=':', linewidth=1)

# Azul, naranja, gris claro, gris oscuro, azul medio, naranja oscuro, gris medio, azul claro, carne, gris mas claro             
CB_color_cycle = ['#006ba4', '#ff800e', '#ababab',
                  '#595959', '#5f9ed1', '#c85200',
                  '#898989', '#a2c8ec', '#ffbc79', '#cfcfcf']

plt.rc('axes', prop_cycle=(cycler('color', CB_color_cycle)))

# 1. Static response is in control_plane4.py


# Leer el valor del archivo
def find_valor(linea):
    indice = string.find(linea, "\t")
    valor = linea[0:indice]
    linea = linea[indice+1:len(linea)]
    return float(valor), linea

arch = open('sys-data-1sl.txt', 'r')
t = []
workload = []
tr1 = []
linea = arch.readline()
while linea!="":  
    
    tiempo, linea = find_valor(linea)
    t.append(tiempo)
    carga, linea = find_valor(linea)
    workload.append(carga)
    tr1.append(float(linea))
    linea=arch.readline()
arch.close()

arch = open('sys-data-4sl.txt', 'r')
t4 = []
workload4 = []
tr4 = []
linea = arch.readline()
while linea!="":  
    
    tiempo, linea = find_valor(linea)
    t4.append(tiempo)
    carga, linea = find_valor(linea)
    workload4.append(carga)
    tr4.append(float(linea))
    linea=arch.readline()
arch.close()

t = np.array(t)
tr1 = np.array(tr1)
workload = np.array(workload)


# Data for training K = 1
points_training = np.concatenate((np.arange(0,len(t)/3,1), np.arange(len(t)/3,len(t),30)), axis=0)
#points_training = np.arange(0,len(t),8)
t_training = t[points_training]
w_training = workload[points_training]
#w_training = w_training/max(w_training)

tr_training = tr1[points_training]


# Data for validation
points_validation = np.arange(0,len(t),5)
t_validation = t[points_validation]
w_validation = workload[points_validation]
tr_validation = tr1[points_validation]

# Training of the GP model
# Learning
gp_system = gp.GPR()
gp_system.setOptimizer("Minimize", num_restarts=10)
gp_system.getPosterior(w_training, tr_training)
gp_system.optimize(w_training, tr_training)

#plt.figure()
gp_system.predict(np.sort(w_validation))
#gp_system.plot()

# Validation
tr_predicted = np.zeros(len(t_validation))
for i in np.arange(len(t_validation)):
    gp_system.predict(np.array([w_validation[i]]))
    tr_predicted[i] = np.asscalar(gp_system.ym)



# Para sl=4

t4 = np.array(t4)
tr4 = np.array(tr4)
workload4 = np.array(workload4)

# Data for training K = 4
points_training4 = np.concatenate((np.arange(0,len(t4)/3,1), np.arange(len(t4)/3,len(t4),40)), axis=0)
t_training4 = t4[points_training4]
w_training4 = workload4[points_training4]
tr_training4 = tr4[points_training4]


# Data for validation
points_validation4 = np.arange(0,len(t4), 5) #5
t_validation4 = t4[points_validation4]
w_validation4 = workload4[points_validation4]
tr_validation4 = tr4[points_validation4]

# Training of the GP model
# Learning
gp_system4 = gp.GPR()

w_action4 = np.array([w_training4]).T
#try:
#m = gp.mean.Const()
kernel = gp.cov.RBF(log_ell=3000)
gp_system4.setPrior(kernel=kernel)
#gp_system.setPrior(mean=m, kernel=kernel)
gp_system4.setOptimizer("Minimize", num_restarts=10)
gp_system4.getPosterior(w_training4, tr_training4)
gp_system4.optimize(w_training4, tr_training4)

#plt.figure()
gp_system4.predict(np.sort(w_validation4))
#gp_system4.plot()

# Validation
tr_predicted4 = np.zeros(len(t_validation4))
for i in np.arange(len(t_validation4)):
    #gp_system.predict((np.array([[w_validation[i]],[a_validation[i]]]).T))
    gp_system4.predict(np.array([w_validation4[i]]))
    tr_predicted4[i] = np.asscalar(gp_system4.ym)


mse1 = (1.0/len(tr_validation))*sum(np.power((tr_validation-tr_predicted),2))
print 'mse1', mse1
#mse2 = (1.0/len(tr_validation2))*sum(np.power((tr_validation2-tr_predicted2),2))
#print 'mse2', mse2
#mse3 = (1.0/len(tr_validation3))*sum(np.power((tr_validation3-tr_predicted3),2))
#print 'mse3', mse3
mse4 = (1.0/len(tr_validation4))*sum(np.power((tr_validation4-tr_predicted4),2))
print 'mse4', mse4


# Plot
#plt.figure()

f1, = ax0.semilogy(t_validation, tr_validation, '^:', linewidth=0.9, color='#006ba4', markersize=3, label='Measured MRT for one SL instance')
f2, = ax0.semilogy(t_validation, tr_predicted, '+:', linewidth=0.9, color='#ff800e', markersize=4, label='Estimated MRT for one SL instance')

f3, = ax0.semilogy(t_validation4, tr_validation4, 's:', linewidth=0.9, color='#c85200', markersize=3, label='Measured MRT for four SL instances')
f4, = ax0.semilogy(t_validation4, tr_predicted4, 'x:', linewidth=0.9, color='#5f9ed1', markersize=4, label='Estimated MRT for four SL instances')

#plt.title('Mean Response Time Variarion in 24 hours')
plt.xlabel('Time of the day (hours)')
plt.ylabel('Mean response time (seconds)')

plt.ylim(0.00009, 0.006)
plt.xticks(np.linspace(0, 24, 13, endpoint=True))
#plt.grid()

plt.twinx() 

f0, = plt.plot(t4, workload4, '-', linewidth=1.2, color='#595959', label='Total arriving rate of control messages') 
plt.xlabel('Time of the day (hours)') 
plt.ylabel('Workload (control messages per second)') 
plt.ylim(0.0, 42000.0)
plt.xlim(-0.2, 24.0)

plt.legend(loc=2, handles=[f0, f1, f2, f3, f4])
plt.show()