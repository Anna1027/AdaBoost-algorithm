#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
from numpy import linalg as la
import pandas as pd
from scipy import stats
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
import math
from scipy.stats import multivariate_normal as mvnorm
from matplotlib import pyplot as plt 
import sys


# In[25]:


def find_gmm(x_data, num_gmm):
    gmm_info = []
    #find mean and covariance
    mean = np.mean(x_data, axis = 0)
    cov = np.cov(x_data.T)
    gau_mean = np.random.multivariate_normal(mean, cov, num_gmm)
    
    for i in gau_mean:
        gmm_gau_mean = i
        gmm_gau = [i, cov]
        gmm_info.append(gmm_gau)
    return gmm_info


# In[26]:


#find sigma
def find_sigma(data, phi_list, uk, nk):
    sum = np.repeat(float(0), 10*10).reshape((10,10))
    for r in range(data.shape[0]):
        multiply = ((data[r, :] - uk).reshape(data[r, :].shape[0],1)).dot((data[r, :]-uk).reshape(1, data[r, :].shape[0]))
        
        multiply = phi_list[r]*multiply
        
        sum += multiply
    sigmak = (1 / nk)*sum
    return sigmak


# In[27]:


def get_phi(data,gmm,mix_weight,k):
    prob = mvnorm.pdf(data, gmm[k][0], gmm[k][1],allow_singular = True)
    # print(prob)
    numerator = mix_weight[k] * prob

    
    norm_list = []
    
    for j in gmm:
        norm_list.append(mvnorm.pdf(data, j[0], j[1],allow_singular = True))
    norm_list = np.array(norm_list)
    mix_weight = np.array(mix_weight)
    denominator = mix_weight.dot(norm_list.T)
    phi_k = numerator / denominator
        
    return phi_k


# In[28]:


def get_multip(data,num_gmm,mix_weight,gmm):
    multipl = 0
    for k in range(num_gmm):
        multipl += mix_weight[k] * mvnorm.pdf(data, gmm[k][0], gmm[k][1])
    return  np.log(multipl)


# In[29]:


def get_multip_pred(data, num_gmm, mix_weight, gmm):
    multipl = 0
    for k in range(num_gmm):
        multipl += mix_weight[k]*mvnorm.pdf(data,gmm[k][0], gmm[k][1])
    return multipl


# In[30]:


def find_TP(y_true, y_pred):
    return sum((y_true == 1)& (y_pred == 1))

def find_TN(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 0))

def find_FP(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 1))

def find_FN(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 0))


# In[8]:


num_gmm = 3
x_array = np.genfromtxt('hw3-data/Prob3_Xtrain.csv',delimiter = ',')
y_array = np.genfromtxt('hw3-data/Prob3_ytrain.csv',delimiter = ',')

index1 = np.argwhere(y_array == 1)
index0 = np.argwhere(y_array == 0)
index1 = index1.flatten()
x1_data = x_array[index1]

index0 = index0.flatten()
x0_data = x_array[index0]


# In[9]:


iteration = 30
run = 10

def cal_objective_function(data, iteration, gmm, num_gmm, mix_weight):
    total_L = []
    
    for iter in range(iteration):
        phi_list = []
        for i in range(num_gmm):
            phi_list.append(np.apply_along_axis(get_phi, 1, data, gmm, mix_weight, i))
            
        for k in range(num_gmm):
            nk = phi_list[k].sum()
                
            mix_weight[k] = nk / len(phi_list[k])
                
            gmm[k][0] = (1/nk)*np.sum(((phi_list[k]*data.T).T), axis = 0)
                
            gmm[k][1] = find_sigma(data, phi_list[k], gmm[k][0], nk)
                
        prob_list = np.apply_along_axis(get_multip, 1,data,num_gmm, mix_weight, gmm)
        L = np.sum(prob_list)
        total_L.append(L)
    return total_L
            
    


# In[10]:


max0 = -sys.float_info.max
max1 = -sys.float_info.max
final_weight0 = []
final_weight1 = []
final_gmm0 = []
final_gmm1 = []

for times in range(run):
    print('run', times)
    
    gmm0 = find_gmm(x0_data, num_gmm)
    
    mix_weight0 = [1 / num_gmm for i in range (num_gmm)]
    plt.figure(1)
    Loss0 = cal_objective_function(x0_data, iteration, gmm0, num_gmm, mix_weight0)
    if Loss0[-1] > max0:
        max0 = Loss0[-1]
        final_gmm0 = gmm0
        final_weight0 = mix_weight0
    plt.plot(range(5, 31), Loss0[4:30], label = 'run %d'%(times))
    
plt.xticks([int(x) for x in np.linspace(5, 30)])
plt.xlabel('Iterations')
plt.ylabel('Log marginal objective function')
plt.title ('Class 0 data log marginal objective function for 10 runs')
plt.legend()

plt.show()


# In[11]:


for times in range(run):
    print("run", times)
    
    gmm1 = find_gmm(x1_data, num_gmm)
    
    mix_weight1 = [1 / num_gmm for i in range(num_gmm)]
    
    plt.figure(2)
    Loss1 = cal_objective_function(x1_data, iteration, gmm1, num_gmm, mix_weight1)
    if Loss1[-1] > max1:
        max1 = Loss1[-1]
        final_gmm1 = gmm1
        final_weight1 = mix_weight1
        
    plt.plot(range(5, 31), Loss1[4:30], label = 'run %d' %(times))
plt.xticks([int(x) for x in np.linspace(5,30)])
plt.xlabel('Iterations')
plt.ylabel('Log marginal objective function')
plt.title('Class 1 data log marginal objective function for 10 runs')
plt.legend()

plt.show()


# In[12]:


#part b: Using the best run for each class after 30 iterations, predict the testing data using a Bayes classifier
#and show the result in a 2  2 confusion matrix, along with the accuracy percentage. Repeat
#this process for a 1-, 2-, 3- and 4-Gaussian mixture model. Show all results nearby each other,
#and don’t repeat Part (a) for these other cases

print(len(index0))
print(len(index1))
prior0 = len(index0) / x_array.shape[0]
prior1 = len(index1) / x_array.shape[0]
print(prior0)
print(prior1)

x_test = np.genfromtxt('hw3-data/Prob3_Xtest.csv',delimiter = ',')
y_test = np.genfromtxt('hw3-data/Prob3_ytest.csv',delimiter = ',')

likelihood0 = prior0 * np.apply_along_axis(get_multip_pred,1,x_test,num_gmm,final_weight0,final_gmm0)
likelihood1 = prior1 * np.apply_along_axis(get_multip_pred,1,x_test,num_gmm,final_weight1,final_gmm1)

p = np.greater(likelihood1, likelihood0)
y_pred = p.astype(int)
# print(y_pred)
TP = find_TP(y_test, y_pred)
FP = find_FP(y_test, y_pred)
TN = find_TN(y_test, y_pred)
FN = find_FN(y_test, y_pred)
print("TP",TP)
print("FP",FP)
print("TN",TN)
print("FN",FN)

acc = (TP + TN) / x_test.shape[0]
print("Accuracy", acc)


# In[9]:


num_gmm = 1
x_array = np.genfromtxt('hw3-data/Prob3_Xtrain.csv',delimiter = ',')
y_array = np.genfromtxt('hw3-data/Prob3_ytrain.csv',delimiter = ',')

index1 = np.argwhere(y_array == 1)
index0 = np.argwhere(y_array == 0)
index1 = index1.flatten()
x1_data = x_array[index1]

index0 = index0.flatten()
x0_data = x_array[index0]


# In[10]:


iteration = 30
run = 10

def cal_objective_function(data, iteration, gmm, num_gmm, mix_weight):
    total_L = []
    
    for iter in range(iteration):
        phi_list = []
        for i in range(num_gmm):
            phi_list.append(np.apply_along_axis(get_phi, 1, data, gmm, mix_weight, i))
            
        for k in range(num_gmm):
            nk = phi_list[k].sum()
                
            mix_weight[k] = nk / len(phi_list[k])
                
            gmm[k][0] = (1/nk)*np.sum(((phi_list[k]*data.T).T), axis = 0)
                
            gmm[k][1] = find_sigma(data, phi_list[k], gmm[k][0], nk)
                
        prob_list = np.apply_along_axis(get_multip, 1,data,num_gmm, mix_weight, gmm)
        L = np.sum(prob_list)
        total_L.append(L)
    return total_L
            


# In[11]:


max0 = -sys.float_info.max
max1 = -sys.float_info.max
final_weight0 = []
final_weight1 = []
final_gmm0 = []
final_gmm1 = []

for times in range(run):
    print('run', times)
    
    gmm0 = find_gmm(x0_data, num_gmm)
    
    mix_weight0 = [1 / num_gmm for i in range (num_gmm)]
    plt.figure(1)
    Loss0 = cal_objective_function(x0_data, iteration, gmm0, num_gmm, mix_weight0)
    if Loss0[-1] > max0:
        max0 = Loss0[-1]
        final_gmm0 = gmm0
        final_weight0 = mix_weight0
    plt.plot(range(5, 31), Loss0[4:30], label = 'run %d'%(times))
    
plt.xticks([int(x) for x in np.linspace(5, 30)])
plt.xlabel('Iterations')
plt.ylabel('Log marginal objective function')
plt.title ('Class 0 data log marginal objective function for 10 runs')
plt.legend()

#plt.show()


# In[12]:


for times in range(run):
    print("run", times)
    
    gmm1 = find_gmm(x1_data, num_gmm)
    
    mix_weight1 = [1 / num_gmm for i in range(num_gmm)]
    
    plt.figure(2)
    Loss1 = cal_objective_function(x1_data, iteration, gmm1, num_gmm, mix_weight1)
    if Loss1[-1] > max1:
        max1 = Loss1[-1]
        final_gmm1 = gmm1
        final_weight1 = mix_weight1
        
    plt.plot(range(5, 31), Loss1[4:30], label = 'run %d' %(times))
plt.xticks([int(x) for x in np.linspace(5,30)])
plt.xlabel('Iterations')
plt.ylabel('Log marginal objective function')
plt.title('Class 1 data log marginal objective function for 10 runs')
plt.legend()

# plt.show()


# In[13]:


print(len(index0))
print(len(index1))
prior0 = len(index0) / x_array.shape[0]
prior1 = len(index1) / x_array.shape[0]
print(prior0)
print(prior1)

x_test = np.genfromtxt('hw3-data/Prob3_Xtest.csv',delimiter = ',')
y_test = np.genfromtxt('hw3-data/Prob3_ytest.csv',delimiter = ',')

likelihood0 = prior0 * np.apply_along_axis(get_multip_pred,1,x_test,num_gmm,final_weight0,final_gmm0)
likelihood1 = prior1 * np.apply_along_axis(get_multip_pred,1,x_test,num_gmm,final_weight1,final_gmm1)

p = np.greater(likelihood1, likelihood0)
y_pred = p.astype(int)
# print(y_pred)
TP = find_TP(y_test, y_pred)
FP = find_FP(y_test, y_pred)
TN = find_TN(y_test, y_pred)
FN = find_FN(y_test, y_pred)
print("TP",TP)
print("FP",FP)
print("TN",TN)
print("FN",FN)

acc = (TP + TN) / x_test.shape[0]
print("Accuracy", acc)


# In[14]:


num_gmm = 2
x_array = np.genfromtxt('hw3-data/Prob3_Xtrain.csv',delimiter = ',')
y_array = np.genfromtxt('hw3-data/Prob3_ytrain.csv',delimiter = ',')

index1 = np.argwhere(y_array == 1)
index0 = np.argwhere(y_array == 0)
index1 = index1.flatten()
x1_data = x_array[index1]

index0 = index0.flatten()
x0_data = x_array[index0]


# In[15]:


iteration = 30
run = 10

def cal_objective_function(data, iteration, gmm, num_gmm, mix_weight):
    total_L = []
    
    for iter in range(iteration):
        phi_list = []
        for i in range(num_gmm):
            phi_list.append(np.apply_along_axis(get_phi, 1, data, gmm, mix_weight, i))
            
        for k in range(num_gmm):
            nk = phi_list[k].sum()
                
            mix_weight[k] = nk / len(phi_list[k])
                
            gmm[k][0] = (1/nk)*np.sum(((phi_list[k]*data.T).T), axis = 0)
                
            gmm[k][1] = find_sigma(data, phi_list[k], gmm[k][0], nk)
                
        prob_list = np.apply_along_axis(get_multip, 1,data,num_gmm, mix_weight, gmm)
        L = np.sum(prob_list)
        total_L.append(L)
    return total_L
            


# In[16]:


max0 = -sys.float_info.max
max1 = -sys.float_info.max
final_weight0 = []
final_weight1 = []
final_gmm0 = []
final_gmm1 = []

for times in range(run):
    print('run', times)
    
    gmm0 = find_gmm(x0_data, num_gmm)
    
    mix_weight0 = [1 / num_gmm for i in range (num_gmm)]
    plt.figure(1)
    Loss0 = cal_objective_function(x0_data, iteration, gmm0, num_gmm, mix_weight0)
    if Loss0[-1] > max0:
        max0 = Loss0[-1]
        final_gmm0 = gmm0
        final_weight0 = mix_weight0
    plt.plot(range(5, 31), Loss0[4:30], label = 'run %d'%(times))
    
plt.xticks([int(x) for x in np.linspace(5, 30)])
plt.xlabel('Iterations')
plt.ylabel('Log marginal objective function')
plt.title ('Class 0 data log marginal objective function for 10 runs')
plt.legend()

#plt.show()


# In[17]:


for times in range(run):
    print("run", times)
    
    gmm1 = find_gmm(x1_data, num_gmm)
    
    mix_weight1 = [1 / num_gmm for i in range(num_gmm)]
    
    plt.figure(2)
    Loss1 = cal_objective_function(x1_data, iteration, gmm1, num_gmm, mix_weight1)
    if Loss1[-1] > max1:
        max1 = Loss1[-1]
        final_gmm1 = gmm1
        final_weight1 = mix_weight1
        
    plt.plot(range(5, 31), Loss1[4:30], label = 'run %d' %(times))
plt.xticks([int(x) for x in np.linspace(5,30)])
plt.xlabel('Iterations')
plt.ylabel('Log marginal objective function')
plt.title('Class 1 data log marginal objective function for 10 runs')
plt.legend()

#plt.show()


# In[18]:


print(len(index0))
print(len(index1))
prior0 = len(index0) / x_array.shape[0]
prior1 = len(index1) / x_array.shape[0]
print(prior0)
print(prior1)

x_test = np.genfromtxt('hw3-data/Prob3_Xtest.csv',delimiter = ',')
y_test = np.genfromtxt('hw3-data/Prob3_ytest.csv',delimiter = ',')

likelihood0 = prior0 * np.apply_along_axis(get_multip_pred,1,x_test,num_gmm,final_weight0,final_gmm0)
likelihood1 = prior1 * np.apply_along_axis(get_multip_pred,1,x_test,num_gmm,final_weight1,final_gmm1)

p = np.greater(likelihood1, likelihood0)
y_pred = p.astype(int)
# print(y_pred)
TP = find_TP(y_test, y_pred)
FP = find_FP(y_test, y_pred)
TN = find_TN(y_test, y_pred)
FN = find_FN(y_test, y_pred)
print("TP",TP)
print("FP",FP)
print("TN",TN)
print("FN",FN)

acc = (TP + TN) / x_test.shape[0]
print("Accuracy", acc)


# In[31]:


num_gmm = 4
x_array = np.genfromtxt('hw3-data/Prob3_Xtrain.csv',delimiter = ',')
y_array = np.genfromtxt('hw3-data/Prob3_ytrain.csv',delimiter = ',')

index1 = np.argwhere(y_array == 1)
index0 = np.argwhere(y_array == 0)
index1 = index1.flatten()
x1_data = x_array[index1]

index0 = index0.flatten()
x0_data = x_array[index0]


# In[32]:


iteration = 30
run = 10

def cal_objective_function(data, iteration, gmm, num_gmm, mix_weight):
    total_L = []
    
    for iter in range(iteration):
        phi_list = []
        for i in range(num_gmm):
            phi_list.append(np.apply_along_axis(get_phi, 1, data, gmm, mix_weight, i))
            
        for k in range(num_gmm):
            nk = phi_list[k].sum()
                
            mix_weight[k] = nk / len(phi_list[k])
                
            gmm[k][0] = (1/nk)*np.sum(((phi_list[k]*data.T).T), axis = 0)
                
            gmm[k][1] = find_sigma(data, phi_list[k], gmm[k][0], nk)
                
        prob_list = np.apply_along_axis(get_multip, 1,data,num_gmm, mix_weight, gmm)
        L = np.sum(prob_list)
        total_L.append(L)
    return total_L
            


# In[33]:


max0 = -sys.float_info.max
max1 = -sys.float_info.max
final_weight0 = []
final_weight1 = []
final_gmm0 = []
final_gmm1 = []

for times in range(run):
    print('run', times)
    
    gmm0 = find_gmm(x0_data, num_gmm)
    
    mix_weight0 = [1 / num_gmm for i in range (num_gmm)]
    plt.figure(1)
    Loss0 = cal_objective_function(x0_data, iteration, gmm0, num_gmm, mix_weight0)
    if Loss0[-1] > max0:
        max0 = Loss0[-1]
        final_gmm0 = gmm0
        final_weight0 = mix_weight0
    plt.plot(range(5, 31), Loss0[4:30], label = 'run %d'%(times))
    
plt.xticks([int(x) for x in np.linspace(5, 30)])
plt.xlabel('Iterations')
plt.ylabel('Log marginal objective function')
plt.title ('Class 0 data log marginal objective function for 10 runs')
plt.legend()

#plt.show()


# In[34]:


for times in range(run):
    print("run", times)
    
    gmm1 = find_gmm(x1_data, num_gmm)
    
    mix_weight1 = [1 / num_gmm for i in range(num_gmm)]
    
    plt.figure(2)
    Loss1 = cal_objective_function(x1_data, iteration, gmm1, num_gmm, mix_weight1)
    if Loss1[-1] > max1:
        max1 = Loss1[-1]
        final_gmm1 = gmm1
        final_weight1 = mix_weight1
        
    plt.plot(range(5, 31), Loss1[4:30], label = 'run %d' %(times))
plt.xticks([int(x) for x in np.linspace(5,30)])
plt.xlabel('Iterations')
plt.ylabel('Log marginal objective function')
plt.title('Class 1 data log marginal objective function for 10 runs')
plt.legend()

#plt.show()


# In[35]:


print(len(index0))
print(len(index1))
prior0 = len(index0) / x_array.shape[0]
prior1 = len(index1) / x_array.shape[0]
print(prior0)
print(prior1)

x_test = np.genfromtxt('hw3-data/Prob3_Xtest.csv',delimiter = ',')
y_test = np.genfromtxt('hw3-data/Prob3_ytest.csv',delimiter = ',')

likelihood0 = prior0 * np.apply_along_axis(get_multip_pred,1,x_test,num_gmm,final_weight0,final_gmm0)
likelihood1 = prior1 * np.apply_along_axis(get_multip_pred,1,x_test,num_gmm,final_weight1,final_gmm1)

p = np.greater(likelihood1, likelihood0)
y_pred = p.astype(int)
# print(y_pred)
TP = find_TP(y_test, y_pred)
FP = find_FP(y_test, y_pred)
TN = find_TN(y_test, y_pred)
FN = find_FN(y_test, y_pred)
print("TP",TP)
print("FP",FP)
print("TN",TN)
print("FN",FN)

acc = (TP + TN) / x_test.shape[0]
#print("Accuracy", acc)


# In[ ]:




