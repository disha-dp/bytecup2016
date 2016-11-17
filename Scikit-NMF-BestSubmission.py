
# coding: utf-8

# In[55]:

import pandas as pa
import numpy as np
import numpy as np
from sklearn.decomposition import ProjectedGradientNMF
import numpy as np
import pickle

from sklearn.metrics import roc_auc_score

train_data = pa.read_csv("train_data.txt", sep ="\t", header='infer')


print train_data.shape


# In[56]:

#make a user question matrix
question_list = list(set(train_data.iloc[:,0]))
expert_list = list(set(train_data.iloc[:,1]))

r = len(question_list) #num of distinct questions
c = len(expert_list) #num of distinct users

#bring data in the question, user matrix format
train_data = train_data.ix[1:]

matrix = np.zeros((r,c))

print 'forming matrix...'
for row in range(train_data.shape[0]):
    row_info = train_data.iloc[row]
    curr_q , curr_u, label =  row_info[0], row_info[1], row_info[2] 

    #print curr_q, curr_u
    question_index = question_list.index(curr_q)  
    user_index = expert_list.index(curr_u)

    matrix[question_index][user_index] =  label
#print matrix


# In[57]:
print 'running model...'

model = ProjectedGradientNMF( n_components = 50, init='nndsvda', random_state=0,max_iter=300, eta=0.01, alpha = 0.01)
W = model.fit_transform(matrix) 
H = model.components_
rHat = np.dot(W,H)
print 'recon error: ', model.reconstruction_err_

#np.savetxt("rHat.txt",rHat)

#pickle.dump(question_list, 'qList.txt')
# np.savetxt("qList.txt",question_list)
#np.savetxt( user_list,"uList.txt")


# In[61]:

#matrix = pa.read_csv("rHat.txt")
#rHat = np.array(matrix)

validation_data = pa.read_csv("validate_nolabel.txt", sep =",", header='infer')

prob = []
for row in range(validation_data.shape[0]):
    row_info = validation_data.iloc[row]
    curr_q , curr_u=  row_info[0], row_info[1] 
    if curr_q in question_list and curr_u in expert_list:
        question_index = question_list.index(curr_q)  
        user_index = expert_list.index(curr_u)
        cur_prob = rHat[question_index][user_index]

        prob.append(cur_prob)
    else:
        prob.append(0)


#prob = ComputeValidationSet(rHat, validation_data)

#print prob


# In[68]:


final_val = np.c_[validation_data, prob]

#c = open('val.txt','w')
#pickle.dump(final_val, c )

c = ( pa.DataFrame(final_val)).to_csv('v1.txt')
