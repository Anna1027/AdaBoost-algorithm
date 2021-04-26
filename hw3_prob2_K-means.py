#!/usr/bin/env python
# coding: utf-8

# import numpy as np
# import numpy.linalg as nplg
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# % matplotlib inline

# In[6]:


#Generating Gaussian Samples 
models = np.array([1,2,3])
weights = np.array([0.2,0.5,0.3])

mean1 = np.array([0,0])
cov1 = np.array([[1,0],[0,1]])

mean2 = np.array([3,0])
cov2 = np.array([[1,0],[0,1]])

mean3 = np.array([0,3])
cov3 = np.array([[1,0],[0,1]])

samples = []

for i in range(500):
    select = np.random.choice(models, p = weights)
    if select ==1:
        samples.append(np.random.multivariate_normal(mean1,cov1))
    if select ==2:
        samples.append(np.random.multivariate_normal(mean2,cov2))
    if select ==3:
        samples.append(np.random.multivariate_normal(mean3,cov3))
        


# In[14]:


samples = np.array(samples)
K = range(2,6)
obj_all_k = []
centroids_k = []
final_clusters = []
plt.figure(figsize=(8,6))
for k in K:
    
    # Initialize centroids
    index = np.random.choice(range(500),k)
    centroids = samples[index]

    # Intialize clusters array
    c = np.zeros(500)
    obj_all=[]
    
    # Start Iterations
    for iteration in range(20):
        for i in range(500):
            norm=[]
            for cent in centroids:
                dist = (nplg.norm(samples[i]-cent,ord=2))**2
                norm.append(dist)

            norm=np.array(norm)
            cluster = np.argmin(norm)
            c[i]=cluster

    # Calculate objective function
        obj = 0
        for m in range(len(centroids)):
            diff = samples[np.where(c==m)] - np.repeat([centroids[m]],samples[np.where(c==m)].shape[0],axis=0)
            obj_cluster = np.sum(nplg.norm(diff,ord=2,axis=1)**2)
            obj = obj + obj_cluster

        obj_all.append(obj)

    # Re-calculate Centroids
        for j in range(len(centroids)):
            centroids[j] = np.sum(samples[np.where(c==j)],axis=0)/samples[np.where(c==j)].shape[0]
            
    final_clusters.append(c)
    centroids_k.append(centroids)  
    

    plt.plot(range(20),obj_all,label='clusters='+str(k))
    plt.xlabel("iterations")
    plt.ylabel("Objective")
    plt.legend()
            


# In[16]:


#For K = 3;5, plot the 500 data points and indicate the cluster of each for the final iteration by marking it in some way 
cluster3 = pd.DataFrame(np.concatenate((samples,np.transpose(np.matrix(final_clusters[1]))),1),columns = ['x','y','cluster'])
cluster3['cluster'] = cluster3.cluster.astype('int').astype('category')


cluster5 = pd.DataFrame(np.concatenate((samples,np.transpose(np.matrix(final_clusters[3]))),1),columns = ['x','y','cluster'])
cluster5['cluster'] = cluster5.cluster.astype('int').astype('category')

_=sns.lmplot('x','y', data = cluster3, hue = 'cluster', fit_reg = False)
_=sns.lmplot('x','y',data = cluster5, hue = 'cluster', fit_reg = False)

                        


# In[ ]:




