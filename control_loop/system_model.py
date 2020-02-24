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
import math as mt
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor


# 1. Static response is in control_plane4.py

class System_Model(object):
    
    def __init__(self):
        np.random.seed(31)  #31
        plt.close('all')    
        kernel1 = 1.0 * RBF(length_scale=1000, length_scale_bounds=(10000, 10000))
        kernel2 = 1.0 * RBF(length_scale=1000, length_scale_bounds=(10000, 10000))
        kernel3 = 1.0 * RBF(length_scale=4000, length_scale_bounds=(4000, 4000))
        kernel4 = 1.0 * RBF(length_scale=1000, length_scale_bounds=(10000, 10000))
        self.gp_system1 = GaussianProcessRegressor(kernel1)
        self.gp_system2 = GaussianProcessRegressor(kernel2)
        self.gp_system3 = GaussianProcessRegressor(kernel3)
        self.gp_system4 = GaussianProcessRegressor(kernel4)
        self.reward = 0
        self.sl = 0

    def create_model_sl1(self):
        t, workload, tr = self.__load_data('sys-data-1sl.txt')
        self.__plot_data(t, workload, tr)
        t_training, w_training, tr_training = self.__get_data_training(t, workload, tr, 1)
        self.__plot_data_training(t_training, w_training, tr_training)
        t_validation, w_validation, tr_validation = self.__get_data_validation(t, workload, tr, 2)        
        self.__train_sl1(w_training, tr_training)
        self.__validate_gp_sl1(t_validation, w_validation, tr_validation)
    
    def create_model_sl2(self):
        t, workload, tr = self.__load_data('sys-data-2sl.txt')
        self.__plot_data(t, workload, tr)
        t_training, w_training, tr_training = self.__get_data_training(t, workload, tr, 2)
        self.__plot_data_training(t_training, w_training, tr_training)
        t_validation, w_validation, tr_validation = self.__get_data_validation(t, workload, tr, 2)        
        self.__train_sl2(w_training, tr_training)
        self.__validate_gp_sl2(t_validation, w_validation, tr_validation)

    def create_model_sl3(self):
        t, workload, tr = self.__load_data('sys-data-3sl.txt')
        self.__plot_data(t, workload, tr)
        t_training, w_training, tr_training = self.__get_data_training(t, workload, tr, 3)
        self.__plot_data_training(t_training, w_training, tr_training)
        t_validation, w_validation, tr_validation = self.__get_data_validation(t, workload, tr, 2)        
        self.__train_sl3(w_training, tr_training)
        self.__validate_gp_sl3(t_validation, w_validation, tr_validation)

    def create_model_sl4(self):
        t, workload, tr = self.__load_data('sys-data-4sl.txt')
        self.__plot_data(t, workload, tr)
        t_training, w_training, tr_training = self.__get_data_training(t, workload, tr, 4)
        self.__plot_data_training(t_training, w_training, tr_training)
        t_validation, w_validation, tr_validation = self.__get_data_validation(t, workload, tr, 2)        
        self.__train_sl4(w_training, tr_training)
        self.__validate_gp_sl4(t_validation, w_validation, tr_validation)

    
    # Leer el valor del archivo
    def __find_valor(self, linea):
        indice = string.find(linea, "\t")
        valor = linea[0:indice]
        linea = linea[indice+1:len(linea)]
        return float(valor), linea

    def __load_data(self, nombre_file):
        arch = open(nombre_file, 'r')
        t = []
        workload = []
        tr1 = []
        linea = arch.readline()
        while linea!="":  
            
            tiempo, linea = self.__find_valor(linea)
            t.append(tiempo)
            carga, linea = self.__find_valor(linea)
            workload.append(carga)
            tr1.append(float(linea))
            linea=arch.readline()
        arch.close()
        return np.array(t), np.array(workload), np.array(tr1)

    def __plot_data(self, t, workload, tr):
        plt.figure()        
        plt.plot(t, workload, 'g.-', linewidth=0.9) 
        plt.xlabel('Time of the day (hours)') 
        plt.ylabel('Total arriving rate (control messages per second)') 
        plt.grid()
        #plt.ylim(0.0, 0.0006)
        
        plt.twinx() 
        plt.semilogy(t, tr, 'b+-', linewidth=0.9) 
        plt.ylabel('Mean response time (seconds)') 
        plt.grid()
        #plt.xlim(1.9, 13.1)        

    def __get_data_training(self, t, workload, tr, model):                
        # Data for training K = 1
        if model == 1 or model == 2:
            points_training = np.concatenate((np.arange(0,len(t)/3,1), np.arange(len(t)/3,len(t),30)), axis=0)
        elif model == 3:
            #points_training = np.concatenate((np.arange(0,len(t)/3,2), np.arange(len(t)/3,len(t)*2/3,100), np.arange(len(t)*2/3,len(t),10)), axis=0)
            points_training = np.concatenate((np.arange(0,len(t)/3,1), np.arange(len(t)/3,len(t),20)), axis=0)
        elif model == 4:    
            points_training = np.concatenate((np.arange(0,len(t)/3,2), np.arange(len(t)/3,len(t),40)), axis=0)
        
        t_training = t[points_training]
        w_training = workload[points_training]
        #w_training = w_training/max(w_training)        
        tr_training = tr[points_training]        
        return t_training, w_training, tr_training

    def __plot_data_training(self, t, workload, tr):
        # Plot
        plt.figure()
        f1, = plt.semilogy(t, workload, '.-', label='W training')
        f2, = plt.semilogy(t, tr, '+--', label='Tr training')        
        plt.title('Training data')
        plt.xlabel('t (min)')
        plt.ylabel('Training data')
        plt.grid()
        plt.legend(handles=[f1, f2])
        plt.show()

    def __get_data_validation(self, t, workload, tr, step_length):
        # Data for validation
        points_validation = np.arange(0,len(t),1)
        t_validation = t[points_validation]
        w_validation = workload[points_validation]
        #w_validation = w_validation/max(w_validation)
        tr_validation = tr[points_validation]
        return t_validation, w_validation, tr_validation

    def __train_sl1(self, w_training, tr_training):
        self.gp_system1.fit(w_training[:, np.newaxis], tr_training)
        print 'model 1 construido'

    def __train_sl2(self, w_training, tr_training):
        self.gp_system2.fit(w_training[:, np.newaxis], tr_training)
        print 'model 2 construido'
        
    def __train_sl3(self, w_training, tr_training):
        self.gp_system3.fit(w_training[:, np.newaxis], tr_training)
        print 'model 3 construido'
        
    def __train_sl4(self, w_training, tr_training):
        self.gp_system4.fit(w_training[:, np.newaxis], tr_training)
        print 'model 4 construido'         

    def __validate_gp_sl1(self, t_validation, w_validation, tr_validation):
        
        tr_predicted, y_std = self.gp_system1.predict(w_validation[:, np.newaxis], return_std=True)
        # Plot
        plt.figure()
        f1, = plt.semilogy(t_validation, tr_validation, 'b.-', label='Actual MRT')
        f2, = plt.semilogy(t_validation, tr_predicted, 'r+--', label='Predicted MRT')
        #plt.title('Mean Response Time Variarion in 24 hours')
        plt.xlabel('Time of the day (hours)')
        plt.ylabel('Mean response time (seconds)')
        #plt.xlim(1.9, 13.1)
        plt.grid()
        plt.legend(handles=[f1, f2])
        plt.show()
        
        mse = (1.0/len(tr_validation))*sum(np.power((tr_validation-tr_predicted),2))
        print 'mse', mse

    def __validate_gp_sl2(self, t_validation, w_validation, tr_validation):
        # Validation
        tr_predicted, y_std = self.gp_system2.predict(w_validation[:, np.newaxis], return_std=True)

        # Plot
        plt.figure()
        f1, = plt.semilogy(t_validation, tr_validation, 'b.-', label='Actual MRT')
        f2, = plt.semilogy(t_validation, tr_predicted, 'r+--', label='Predicted MRT')
        #plt.title('Mean Response Time Variarion in 24 hours')
        plt.xlabel('Time of the day (hours)')
        plt.ylabel('Mean response time (seconds)')
        #plt.xlim(1.9, 13.1)
        plt.grid()
        plt.legend(handles=[f1, f2])
        plt.show()
        
        mse = (1.0/len(tr_validation))*sum(np.power((tr_validation-tr_predicted),2))
        print 'mse', mse

    def __validate_gp_sl3(self, t_validation, w_validation, tr_validation):
        # Validation
        tr_predicted, y_std = self.gp_system3.predict(w_validation[:, np.newaxis], return_std=True)
        
        # Plot
        plt.figure()
        f1, = plt.semilogy(t_validation, tr_validation, 'b.-', label='Actual MRT')
        f2, = plt.semilogy(t_validation, tr_predicted, 'r+--', label='Predicted MRT')
        #plt.title('Mean Response Time Variarion in 24 hours')
        plt.xlabel('Time of the day (hours)')
        plt.ylabel('Mean response time (seconds)')
        #plt.xlim(1.9, 13.1)
        plt.grid()
        plt.legend(handles=[f1, f2])
        plt.show()
        
        mse = (1.0/len(tr_validation))*sum(np.power((tr_validation-tr_predicted),2))
        print 'mse', mse

    def __validate_gp_sl4(self, t_validation, w_validation, tr_validation):
        # Validation
        tr_predicted, y_std = self.gp_system4.predict(w_validation[:, np.newaxis], return_std=True)

        # Plot
        plt.figure()
        f1, = plt.semilogy(t_validation, tr_validation, 'b.-', label='Actual MRT')
        f2, = plt.semilogy(t_validation, tr_predicted, 'r+--', label='Predicted MRT')
        #plt.title('Mean Response Time Variarion in 24 hours')
        plt.xlabel('Time of the day (hours)')
        plt.ylabel('Mean response time (seconds)')
        #plt.xlim(1.9, 13.1)
        plt.grid()
        plt.legend(handles=[f1, f2])
        plt.show()
        
        mse = (1.0/len(tr_validation))*sum(np.power((tr_validation-tr_predicted),2))
        print 'mse', mse


    def predict_tr(self, sl, lmbda):
        
        if sl == 1:
#            if lmbda < 100:
#                lmbda = 100
            if lmbda > 9865:
                lmbda = 9865
            #self.gp_system1.predict(np.array([lmbda]))
            #tr_predicted, y_std = self.gp_system1.predict(lmbda[:, np.newaxis], return_std=True)
            tr_predicted, y_std = self.gp_system1.predict(lmbda, return_std=True)
            return np.asscalar(tr_predicted)
        elif sl == 2:
#            if lmbda < 100:
#                lmbda = 100
            if lmbda > 20845:
                lmbda = 20845
            tr_predicted, y_std = self.gp_system2.predict(lmbda, return_std=True)
            return np.asscalar(tr_predicted)
        elif sl == 3:
#            if lmbda < 100:
#                lmbda = 100
            if lmbda > 29878:
                lmbda = 29878
            tr_predicted, y_std = self.gp_system3.predict(lmbda, return_std=True)
            return np.asscalar(tr_predicted)
        elif sl == 4:
#            if lmbda < 100:
#                lmbda = 100
            if lmbda > 41643:
                lmbda = 41643
            tr_predicted, y_std = self.gp_system4.predict(lmbda, return_std=True)
            return np.asscalar(tr_predicted)

    def get_next_states(self, action, lmbda):
        sl_prima = action
        tr_prima = self.predict_tr(action, lmbda)
        self.__calculate_reward(sl_prima, tr_prima, lmbda)
        return action, tr_prima

    def __calculate_reward2(self, sl_prima, tr_prima):
        if sl_prima == 1:
            if tr_prima < 0.001:
                self.reward = 1
            else:
                self.reward = -1
        elif sl_prima == 2:
            if tr_prima >= 0.00014 and tr_prima < 0.001: #0.14
                self.reward = 1
            else:
                self.reward = -1
        elif sl_prima == 3:
            if tr_prima >= 0.00014 and tr_prima < 0.001:  #0.15
                self.reward = 1
            else:
                self.reward = -1 
        elif sl_prima == 4:
            if tr_prima >= 0.00014:
                self.reward = 1
            else:
                self.reward = -1        
        
    def __calculate_reward3(self, sl_prima, tr_prima):
        reward = 0
        if sl_prima == 1:
            if tr_prima < 0.001:
                reward = 1
            else:
                reward = -1
        elif sl_prima == 2:
            if tr_prima >= 0.00016 and tr_prima < 0.001:
                reward = 1
            elif tr_prima < 0.00014 or tr_prima >= 0.001:
                reward = -1
        elif sl_prima == 3:
            if tr_prima >= 0.00016 and tr_prima < 0.001:
                reward = 1
            elif tr_prima < 0.00014 or tr_prima >= 0.001:
                reward = -1 
        elif sl_prima == 4:
            if tr_prima >= 0.00016:
                reward = 1
            elif tr_prima < 0.00014:
                reward = -1    
        self.reward = reward
    
    def __calculate_reward3(self, sl_prima, tr_prima):
        reward = 0
        tr_prima = 1000*tr_prima
        if sl_prima == 1:
            reward = -2/(1+mt.exp(-20*(tr_prima-0.9))) + 1
        elif sl_prima == 4:
            reward = 2/(1+mt.exp(-20*(tr_prima-0.3))) - 1
        else:        
            if tr_prima < 0.4:
                reward = 2/(1+mt.exp(-20*(tr_prima-0.3))) - 1
            else:    
                reward = -2/(1+mt.exp(-20*(tr_prima-0.9))) + 1
        print 'reward:', reward    
        self.reward = reward

    def __calculate_reward7(self, sl_prima, tr_prima, workload):
        
        mu = 10167.0
        rho = workload/(sl_prima*mu)
                          
        if sl_prima == 1:
            if tr_prima < 0.001:
                self.reward = 1
            else:
                self.reward = -1
        elif sl_prima == 2:
            if tr_prima >= 0.001:
                self.reward = -1
            else:
                if rho >= 0.45:
                    self.reward = 1
                else:
                    self.reward = -1
        elif sl_prima == 3:
            if tr_prima >= 0.001:
                self.reward = -1
            else:
                if rho >= 0.65:
                    self.reward = 1
                else:
                    self.reward = -1
        elif sl_prima == 4:
            if rho >= 0.75:
                self.reward = 1
            else:
                self.reward = -1 

    def __calculate_reward(self, sl_prima, tr_prima, workload):
        mu = 10167.0
        rho = workload/(sl_prima*mu)
                          
        if sl_prima == 1:
            if tr_prima < 0.001:
                self.reward = 1
            else:
                self.reward = -1
        elif sl_prima == 2:
            if tr_prima >= 0.001:
                self.reward = -1
            else:
                if rho >= 0.45 and rho < 1:
                    self.reward = 1
                else:
                    self.reward = -1
        elif sl_prima == 3:
            if tr_prima >= 0.001:
                self.reward = -1
            else:
                if rho >= 0.65 and rho < 1:
                    self.reward = 1
                else:
                    self.reward = -1
        elif sl_prima == 4:
            if rho >= 0.7:
                self.reward = 1
            else:
                self.reward = -1
            
        
                    
            
    def get_reward(self):
        print 'reward: ', self.reward
        return self.reward
    

    
#sm = System_Model()
#sm.create_model_sl1()
#sm.create_model_sl2()
#
#print 's: ', '1', sm.predict_tr(1, 12000)
#print 's prima para action: ', '1', sm.get_next_states(1, 12000)
#print 'reward: ', sm.get_reward()
#print 's prima para action: ', '2', sm.get_next_states(2, 12000)
#print 'reward: ', sm.get_reward()
#
#
