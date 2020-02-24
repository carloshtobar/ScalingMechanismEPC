# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import string
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib import colors as mcolors
from cycler import cycler


np.random.seed(31)  #31
plt.close('all')

# Azul, naranja, gris claro, gris oscuro, azul medio, naranja oscuro, gris medio, azul claro, carne, gris mas claro             
CB_color_cycle = ['#006ba4', '#ff800e', '#ababab',
                  '#595959', '#5f9ed1', '#c85200',
                  '#898989', '#a2c8ec', '#ffbc79', '#cfcfcf']

plt.rc('axes', prop_cycle=(cycler('color', CB_color_cycle)))

# Leer el valor del archivo
def find_valor(linea):
    indice = string.find(linea, "\t")
    valor = linea[0:indice]
    linea = linea[indice+1:len(linea)]
    return float(valor), linea


arch = open('results-caso1.txt', 'r')
t1 = []
tr1 = []
actions1 = []
linea = arch.readline()
while linea!="":  
    
    tiempo, linea = find_valor(linea)
    t1.append(tiempo)
    response, linea = find_valor(linea)
    tr1.append(response)
    actions1.append(float(linea))
    linea=arch.readline()
arch.close()

arch = open('results-caso2.txt', 'r')
t2 = []
tr2 = []
actions2 = []
linea = arch.readline()
while linea!="":  
    
    tiempo, linea = find_valor(linea)
    t2.append(tiempo)
    response, linea = find_valor(linea)
    tr2.append(response)
    actions2.append(float(linea))
    linea=arch.readline()
arch.close()

arch = open('results-caso3.txt', 'r')
t3 = []
tr3 = []
tr3_predicted = []
actions3 = []
linea = arch.readline()
num_ite_training = []
while linea!="":  
    
    tiempo, linea = find_valor(linea)
    t3.append(tiempo)
    response, linea = find_valor(linea)
    tr3.append(response)
    predicted, linea = find_valor(linea)
    tr3_predicted.append(predicted)    
    action, linea = find_valor(linea)
    actions3.append(action)    
    num_ite_training.append(float(linea))
    
    linea=arch.readline()
arch.close()

# Gráfica
fig = plt.figure()
plt.semilogy(t1, tr1, 'g-', linewidth=0.9) 
plt.semilogy(t2, tr2, 'r-', linewidth=0.9) 
plt.semilogy(t3, tr3, 'm-', linewidth=0.9) 
plt.xlabel('Time (hours)') 
plt.ylabel('Mean response time (seconds)') 
#plt.ylim(0.0, 0.0006)
#plt.annotate('One instance', xy = (15,0.00005), xycoords = 'data', xytext = (5, 0.0001), textcoords = 'data', arrowprops = dict(arrowstyle = "->"))
#plt.annotate('Two instances', xy = (40, 0.00055), xycoords = 'data', xytext = (52, 0.0005), textcoords = 'data', arrowprops = dict(arrowstyle = "->"))
#plt.annotate('One instance', xy = (55,0.00005), xycoords = 'data', xytext = (60, 0.0001), textcoords = 'data', arrowprops = dict(arrowstyle = "->"))
plt.twinx() 
plt.plot(t1, actions1, 'b-', linewidth=1.1) 
plt.plot(t2, actions2, 'k-', linewidth=1.1) 
plt.plot(t3, actions3, 'c-', linewidth=1.1) 
plt.ylabel('Number of SL instances') 
#plt.ylim(0.9, 2.1)
plt.grid()


fig = plt.figure()
ax = fig.gca(projection='3d')

def cc(arg):
    return mcolors.to_rgba(arg, alpha=1)


verts = []
zs = [0.0, 2.0, 4.0]

verts.append(list(zip(t1, tr1)))
verts.append(list(zip(t2, tr2)))
verts.append(list(zip(t3, tr3)))

#for z in zs:
#    ys = t
    #ys = np.random.rand(len(xs))
    #print 'ys:', ys
    #ys[0], ys[-1] = 0, 0
#    verts.append(list(zip(xs, ys)))

poly = PolyCollection(verts, facecolors=[cc('r'), cc('g'), cc('b')])
poly.set_alpha(0.7)
ax.add_collection3d(poly, zs=zs, zdir='y')

ax.set_xlabel('X')
ax.set_xlim3d(0, 24)
ax.set_ylabel('Y')
ax.set_ylim3d(-1, 5)
ax.set_zlabel('Z')

ax.set_zlim3d(0.000001, 0.001)

plt.show() 

# Gráfica principal (caso 3, uso del modelo del sistema)

fig, ax0 = plt.subplots(ncols=1)
ax0.grid(color='#cfcfcf', linestyle=':', linewidth=1)

f1, = ax0.semilogy(t3, tr3, '--', color='#006ba4', linewidth=1.0, label='MRT') 
#f2, = ax0.semilogy(t3, tr3_predicted, '.--', color='#898989', linewidth=0.9, label='Estimated $t_r$') 
plt.xlabel('Time (hours)') 
plt.ylabel('Mean response time (seconds)') 
#plt.grid()
plt.ylim(0.0001, 0.002)
#plt.annotate('One instance', xy = (15,0.00005), xycoords = 'data', xytext = (5, 0.0001), textcoords = 'data', arrowprops = dict(arrowstyle = "->"))
#plt.annotate('Two instances', xy = (40, 0.00055), xycoords = 'data', xytext = (52, 0.0005), textcoords = 'data', arrowprops = dict(arrowstyle = "->"))
#plt.annotate('One instance', xy = (55,0.00005), xycoords = 'data', xytext = (60, 0.0001), textcoords = 'data', arrowprops = dict(arrowstyle = "->"))
plt.twinx() 
f3, = plt.plot(t3, actions3, '-', color='#c85200', linewidth=1.5, label='Number of SL instances') 
plt.ylabel('Number of SL instances') 
plt.xlim(0.0, 24.0)
plt.xticks(np.linspace(0, 24, 13, endpoint=True))
plt.yticks(np.linspace(1, 4, 4, endpoint=True))
plt.legend(loc=2, handles=[f1, f3])
plt.show()

# Comparación caso 3 con caso 1 (q-learning direct)
fig = plt.figure()
plt.semilogy(t3, tr3, 'b-', linewidth=1.2) 
plt.semilogy(t1, tr1, 'r--', linewidth=1.2) 
plt.xlabel('Time (hours)') 
plt.ylabel('Mean response time (seconds)') 
plt.grid()
plt.ylim(0.0001, 0.002)
#plt.annotate('One instance', xy = (15,0.00005), xycoords = 'data', xytext = (5, 0.0001), textcoords = 'data', arrowprops = dict(arrowstyle = "->"))
#plt.annotate('Two instances', xy = (40, 0.00055), xycoords = 'data', xytext = (52, 0.0005), textcoords = 'data', arrowprops = dict(arrowstyle = "->"))
#plt.annotate('One instance', xy = (55,0.00005), xycoords = 'data', xytext = (60, 0.0001), textcoords = 'data', arrowprops = dict(arrowstyle = "->"))
plt.twinx() 
plt.plot(t3, actions3, 'k-', linewidth=1.5) 
plt.plot(t1, actions1-0.02*np.ones(len(actions1)), 'g-', linewidth=1.5) 

plt.ylabel('Number of SL instances') 
plt.xlim(0.0, 24.0)
plt.xticks(np.linspace(0, 24, 13, endpoint=True))
plt.yticks(np.linspace(1, 2, 2, endpoint=True))

# Comparación caso 3 con caso 2 (threshold rules)
fig = plt.figure()
plt.semilogy(t3, tr3, '-', color='gray', linewidth=1.2) 
plt.semilogy(t2, tr2, 'r-', linewidth=1.2) 
plt.xlabel('Time (hours)') 
plt.ylabel('Mean response time (seconds)') 
plt.grid()
plt.ylim(0.0001, 0.002)
#plt.annotate('One instance', xy = (15,0.00005), xycoords = 'data', xytext = (5, 0.0001), textcoords = 'data', arrowprops = dict(arrowstyle = "->"))
#plt.annotate('Two instances', xy = (40, 0.00055), xycoords = 'data', xytext = (52, 0.0005), textcoords = 'data', arrowprops = dict(arrowstyle = "->"))
#plt.annotate('One instance', xy = (55,0.00005), xycoords = 'data', xytext = (60, 0.0001), textcoords = 'data', arrowprops = dict(arrowstyle = "->"))
plt.twinx() 
plt.plot(t3, actions3, '-', color='orange', linewidth=1.5) 
plt.plot(t2, actions2-0.02*np.ones(len(actions2)), 'g-', linewidth=1.5) 

plt.ylabel('Number of SL instances') 
plt.xlim(0.0, 24.0)
plt.xticks(np.linspace(0, 24, 13, endpoint=True))
plt.yticks(np.linspace(1, 2, 2, endpoint=True))

# Gráficas separadas caso 1 y 2
# Caso 1 (q-learning direct)
fig, ax0 = plt.subplots(ncols=1)
ax0.grid(color='#cfcfcf', linestyle=':', linewidth=1)
f1, = ax0.semilogy(t1, tr1, '--', color='#006ba4', linewidth=1.0, label='MRT') 
plt.xlabel('Time (hours)') 
plt.ylabel('Mean response time (seconds)') 
#plt.grid()
plt.ylim(0.0001, 0.002)
#plt.annotate('One instance', xy = (15,0.00005), xycoords = 'data', xytext = (5, 0.0001), textcoords = 'data', arrowprops = dict(arrowstyle = "->"))
#plt.annotate('Two instances', xy = (40, 0.00055), xycoords = 'data', xytext = (52, 0.0005), textcoords = 'data', arrowprops = dict(arrowstyle = "->"))
#plt.annotate('One instance', xy = (55,0.00005), xycoords = 'data', xytext = (60, 0.0001), textcoords = 'data', arrowprops = dict(arrowstyle = "->"))
plt.twinx() 
f2, = plt.plot(t1, actions1, '-', color='#c85200', linewidth=1.5, label='Number of SL instances') 
plt.ylabel('Number of SL instances') 
plt.xlim(0.0, 24.0)
plt.xticks(np.linspace(0, 24, 13, endpoint=True))
plt.yticks(np.linspace(1, 4, 4, endpoint=True))
plt.legend(loc=2, handles=[f1, f2])
plt.show()

# Caso 2 (threshold rules)
fig, ax0 = plt.subplots(ncols=1)
ax0.grid(color='#cfcfcf', linestyle=':', linewidth=1)
f1, = ax0.semilogy(t2, tr2, '--', color='#006ba4', linewidth=1.0, label='MRT') 
plt.xlabel('Time (hours)') 
plt.ylabel('Mean response time (seconds)') 
#plt.grid()
plt.ylim(0.0001, 0.002)
#plt.annotate('One instance', xy = (15,0.00005), xycoords = 'data', xytext = (5, 0.0001), textcoords = 'data', arrowprops = dict(arrowstyle = "->"))
#plt.annotate('Two instances', xy = (40, 0.00055), xycoords = 'data', xytext = (52, 0.0005), textcoords = 'data', arrowprops = dict(arrowstyle = "->"))
#plt.annotate('One instance', xy = (55,0.00005), xycoords = 'data', xytext = (60, 0.0001), textcoords = 'data', arrowprops = dict(arrowstyle = "->"))
plt.twinx() 
f2, = plt.plot(t2, actions2, '-', color='#c85200', linewidth=1.5, label='Number of SL instances') 
plt.ylabel('Number of SL instances') 
plt.xlim(0.0, 24.0)
plt.xticks(np.linspace(0, 24, 13, endpoint=True))
plt.yticks(np.linspace(1, 4, 4, endpoint=True))
plt.legend(loc=2, handles=[f1, f2])
plt.show()

# Numero de trainings caso 3
fig, ax0 = plt.subplots(ncols=1)
ax0.grid(color='#cfcfcf', linestyle=':', linewidth=1)
f1, = plt.plot(t3[1:len(t3)], num_ite_training[1:len(t3)], '.:', color='#006ba4', linewidth=1.0, label='Number of iterations') 
plt.xlabel('Time (hours)') 
plt.ylabel('Number of iterations') 
plt.legend(loc=1, handles=[f1])
plt.ylim(-5, 55)
plt.yticks(np.linspace(1, 218, 6, endpoint=True))
plt.xlim(0.0, 24.0)
plt.xticks(np.linspace(0, 24, 13, endpoint=True))
plt.show()