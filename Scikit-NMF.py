
# coding: utf-8

# In[2]:

import pandas as pa
import numpy as np
import nimfa

train_data = pa.read_csv("../bytecup2016data/train_data.txt", sep ="\t", header='infer')

train_features = pa.read_csv("../bytecup2016data/question_info.txt", sep ="\t")

print train_data.shape


# In[3]:

#make a user question matrix
'''
q/e 1 2 3
1
2
3
'''
question_list = list(set(train_data.iloc[:,0]))
expert_list = list(set(train_data.iloc[:,1]))

r = len(question_list) #num of distinct questions
c = len(expert_list) #num of distinct users

matrix = np.zeros((r,c))


# In[4]:

print matrix.shape


# In[5]:

#bring data in the question, user matrix format
train_data = train_data.ix[1:]

#print train_data

for row in range(train_data.shape[0]):
    row_info = train_data.iloc[row]
    curr_q , curr_u, label =  row_info[0], row_info[1], row_info[2] 

    #print curr_q, curr_u
    question_index = question_list.index(curr_q)  
    user_index = expert_list.index(curr_u)
    
    matrix[question_index][user_index] =  label

print matrix


# In[ ]:

import numpy as np
from sklearn.decomposition import NMF

model = NMF( n_components = 100, init='random', random_state=0,max_iter=20000, eta=0.01, alpha = 0.1)
model.fit(matrix) 


# In[ ]:

#np.savetxt("sknmf.txt", model.components_)
print model.reconstruction_err_
print model.n_iter_


# In[49]:

print rHat.shape
print 'original matrix shape: ', matrix.shape
#get validation data
#for each entry in validation set, get prediction from the newly composed matrix


# In[ ]:



