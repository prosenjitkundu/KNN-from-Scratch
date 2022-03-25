#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
import operator


# In[2]:


datatrain = pd.read_csv("project2.csv").values
datatest = pd.read_csv("project2_test.csv").values


# In[3]:


end = len(datatrain[0])-1
X_train = datatrain[:,:end]
X_test = datatest
Y_train = datatrain[:,-1]


# In[9]:


class KNN:
    def __init__(self):
        self.k = None
        
    def distance(self, X1, X2):
        eucl_dist = np.sqrt(np.sum((X1-X2)**2))
        return eucl_dist
    
    def KNN_classification(self,X_train, X_test, Y_train, k):
        distances = []
        para1 = X_test
        for j in range((X_train).shape[0]):
            para2 = X_train[j]
            distances.append((Y_train[j], self.distance(para1, para2)))
        distances.sort(key=operator.itemgetter(1))
        k_closest = []
        k_closest = distances[:k]
        return k_closest
        
    def fit(self, X_train, X_test, Y_train, k):
        closest_dist = []
        file = open('21CS60R22_P2.out','a')
        k_val = 'k = ' + str(k) + '       \n'
        file.write(k_val)
        #calling the KNN Classifier for all the test row
        for i in range(len(X_test)):
            result = self.KNN_classification(X_train, X_test[i], Y_train, k)
            pred = self.prediction(result)
            print('Test Instance ', i , ': ', pred)
            file.write(str(int(pred))+' ')
        file.write('\n')
        file.close()
            
    def prediction(self, KclosNeigh):
        closest = [x[0] for x in KclosNeigh]
        one = 0
        zero = 0
        #print(closest)
        for cnt in range(len(closest)):
            if int(closest[cnt]) == 0:
                zero = zero + 1
            else:
                one = one + 1
        if one > zero:
            return 1
        else:
            return 0


# In[10]:


print("Enter any K value:")
print("Enter 17 for optimal value floor(root(300)) = 17")
K = int(input())
classifier = KNN()
classifier.fit(X_train, X_test, Y_train, K)


# In[ ]:




