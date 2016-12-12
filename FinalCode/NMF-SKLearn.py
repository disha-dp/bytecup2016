
# coding: utf-8

# In[55]:

import pandas as pa
from sklearn.decomposition import ProjectedGradientNMF
import numpy as np
import pickle

from sklearn.metrics import roc_auc_score

train_data = pa.read_csv("../bytecup2016data/invited_info_train.txt", sep ="\t", header='infer')


test_data = pa.read_csv("../bytecup2016data/test_no_label.txt", sep =",", header='infer')

validation_data = pa.read_csv("../bytecup2016data/validate_nolabel.txt", sep =",", header='infer')


#print train_data.shape


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

model = ProjectedGradientNMF( n_components = 45, init='nndsvda', random_state=0,max_iter=300, eta=0.01, alpha = 0.01)
W = model.fit_transform(matrix)
H = model.components_
rHat = np.dot(W,H)
print 'recon error: ', model.reconstruction_err_

test_prob = []
for row in range(test_data.shape[0]):
    row_info = test_data.iloc[row]
    curr_q , curr_u=  row_info[0], row_info[1]
    if curr_q in question_list and curr_u in expert_list:
        question_index = question_list.index(curr_q)
        user_index = expert_list.index(curr_u)
        cur_prob = rHat[question_index][user_index]

        test_prob.append(cur_prob)
    else:
        test_prob.append(0)


validation_prob = []
for row in range(validation_data.shape[0]):
    row_info = validation_data.iloc[row]
    curr_q , curr_u=  row_info[0], row_info[1]
    if curr_q in question_list and curr_u in expert_list:
        question_index = question_list.index(curr_q)
        user_index = expert_list.index(curr_u)
        cur_prob = rHat[question_index][user_index]

        validation_prob.append(cur_prob)
    else:
        validation_prob.append(0)

train_prob = []
for row in range(train_data.shape[0]):
    row_info = train_data.iloc[row]
    curr_q , curr_u=  row_info[0], row_info[1]
    if curr_q in question_list and curr_u in expert_list:
        question_index = question_list.index(curr_q)
        user_index = expert_list.index(curr_u)
        cur_prob = rHat[question_index][user_index]

        train_prob.append(cur_prob)
    else:
        train_prob.append(0)




train_info = train_prob#np.c_[train_data, train_prob]
train_ = ( pa.DataFrame(train_info)).to_csv('train_nmf.csv')



test_info = np.c_[test_data, test_prob]
test_ = ( pa.DataFrame(test_info)).to_csv('final_nmf.csv')


validate_info = np.c_[validation_data, validation_prob]
c = ( pa.DataFrame(validate_info)).to_csv('temp_nmf.csv')

