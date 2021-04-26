#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm 
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["figure.figsize"] = (10,8)


# In[2]:


class LeastSquareRegression(object):
    def __init__(self):
        self.weights = []
    def calcLeastSquare(self, X, y ):
        xtranspose = np.transpose(X)
        xtransx = np.dot(xtranspose, X)
        if xtransx.shape[0] != xtransx.shape[1]:
            raise ValueError("Needs to be square matrix for inverse")
        matinv = np.linalg.inv(xtransx)
        xtransy = np.dot(xtranspose, y)
        self.weights = np.dot(matinv, xtransy)
        
    def makePredictions(self, X):
        class_output = np.dot(X, self.weights)
        return np.sign(class_output)


# In[3]:


class Boosting(object):
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.N = self.X_train.shape[0]
        self.y_train = y_train
        self.weights = np.ones(self.N)/self.N
        self.epsilont = []
        self.alphas = []
        self.classifiers = []
        self.stemplot = {}
        self.weights_t = np.zeros((2500, self.N))
        
    def doBoosting(self):
        #output = []
        for t in range(2500):
            output = np.random.choice(self.N, self.N, p = self.weights)
            #output.append(np.random.choice(self.N, self.N, p = self.weights))
        #print(len(output))
            #for t in output:
                #self.stemplot[t] = self.stemplot.get(t, 0) + 1
                
            
            B_Xtrain = self.X_train[output]
            B_ytrain = self.y_train[output]
            
            ls = LeastSquareRegression()
            ls.calcLeastSquare(B_Xtrain, B_ytrain)
            
            #print (ls.weights)
            Y_pred = ls.makePredictions(self.X_train)
            
            e_t = np.sum((Y_pred != self.y_train)* self.weights)
            
            if e_t > 0.5:
                ls.weights = -ls.weights
                Y_pred = ls.makePredictions(self.X_train)
                e_t = np.sum((Y_pred != self.y_train)*self.weights)
            
                #print e_t
            self.epsilont.append(e_t)
            alpha_t = 0.5*np.log((1 - e_t) / e_t)
            self.alphas.append(alpha_t)
            self.classifiers.append(ls) 
            #print(len(self.alphas))
                
                #print alpha_t
            self.weights = self.weights * np.exp(-alpha_t * Y_pred * self.y_train)
            self.weights = self.weights / np.sum(self.weights)
            self.weights_t[t, :] = self.weights / np.sum(self.weights)
            #print(Y_pred.shape)


# In[4]:


X_train = np.genfromtxt('hw3-data/Prob1_X.csv',delimiter = ',')
Y_train = np.genfromtxt('hw3-data/Prob1_y.csv',delimiter = ',')


# In[5]:


#split the data 
#X_train, x_test, Y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[6]:


#print(len(Y_train))


# In[9]:


training_error = []
testing_error = []

boost = Boosting(X_train, Y_train)
boost.doBoosting()

#print(len(boost.alphas))

for t in tqdm(range(1, 2501)):
    sum_train = np.zeros(X_train.shape[0])
    #sum_test = np.zeros(x_test.shape[0])
    for i in range(t):
        alpha = boost.alphas[i]
        classifier = boost.classifiers[i]
        
        sum_train += (alpha * classifier.makePredictions(X_train))
        #sum_test +=(alpha * classifier.makePredictions(x_test))
        
    fboost_train_pred = np.sign(sum_train)
    #fboost_test_pred = np.sign(sum_test)
    
    training_error.append(np.sum(fboost_train_pred != Y_train) / Y_train.shape[0])
    #testing_error.append(np.sum(fboost_test_pred != y_test) / y_test.shape[0])
    
    


# In[10]:


#show upper bound of the training error as a function of t 
training_upper_bound = []
for t in tqdm(range(1, 2501)):
    ub = 0
    for i in range(t):
        epsilon = boost.epsilont[i]
        ub += np.power((0.5 - epsilon), 2)
    training_upper_bound.append(np.exp(-2 * ub))


# In[11]:


#Plotting the empirical training error: 

plt.figure()
plt.plot(range(1, 2501), training_error, label = "Training error")
plt.plot(range(1,2501), training_upper_bound,label = "upper bound line")
plt.title("Empirical Training Error of fboost (t) for t = 1,2,....2500 and Upper bound of training error as a function of t")
plt.legend()
plt.show


# In[12]:


#show a stem plot of the average of the distribution on the data across all 2500 iterations (the empirical average of wt over t)

plt.figure()
plt.stem(range(boost.N), np.mean(boost.weights_t, axis = 0))
plt.xlabel("Data Index")
plt.ylabel("Average of Wt over t")
plt.title('Average of the distribution of the data across 2500 iterations')


# In[14]:


#plot epsilon as a function of t 
plt.figure
plt.plot(range(1,2501), boost.epsilont)
plt.title("Epsilon as a function of t")
plt.show()


# In[15]:


#plot alpha as a function of t 
plt.figure
plt.plot(range(1, 2501), boost.alphas)
plt.title("Alphas as a function of t")
plt.show()


# In[ ]:




