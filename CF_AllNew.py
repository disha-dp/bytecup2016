
# coding: utf-8

# In[1]:

import pandas as pa
import numpy as np
import sklearn
from sklearn import preprocessing
import math
from itertools import izip


# In[2]:

training_data = pa.read_csv("../bytecup2016data/invited_info_train.txt", sep = "\t")
question_data =  pa.read_csv("../bytecup2016data/question_info.txt", sep="\t")
user_data =  pa.read_csv("../bytecup2016data/user_info.txt", sep = "\t")
validation_info = pa.read_csv("../bytecup2016data/validate_nolabel.txt", sep = ",")


# In[4]:

'''
u/q matrix 
'''
question_list = list(set(training_data.iloc[:,0]))
expert_list = list(set(training_data.iloc[:,1]))

row_train = training_data.shape[0]

eq_matrix = np.zeros((len(expert_list), len(question_list)))

eq_matrix.fill(np.nan) #----fill with nans

lines = []

for row in range(training_data.shape[0]):
    row_info = training_data.iloc[row]
    curr_q , curr_u, label =  row_info[0], row_info[1], row_info[2] 
    question_index = question_list.index(curr_q)  
    expert_index = expert_list.index(curr_u)    
    eq_matrix[expert_index][question_index] =  label


# In[5]:

print eq_matrix[12][24]


# In[5]:

'''
{
u1: [0,1,2,3],
}
'''
prediction_dict = {}

#print validation_info

for row in range(validation_info.shape[0]):
    row_info = validation_info.iloc[row]
    curr_q , curr_u  =  row_info[0], row_info[1] 

    if curr_u not in expert_list:
        expert_list.append(curr_u)
    
    if curr_q not in question_list:
        question_list.append(curr_q)
        
    question_index = question_list.index(curr_q)  
    expert_index = expert_list.index(curr_u)    
    

    if expert_index not in prediction_dict:
        prediction_dict[expert_index]=[question_index]
    else:
        prediction_dict[expert_index].append(question_index)


# In[10]:

'''
r_bar = []
'''

r_bar = []

#for row in range(eq_matrix.shape[0]):
#0: 0 0 0 0 0 1 1 nan nan
ru_bar  =  np.nanmean(eq_matrix , axis = 1)

rq_bar  =  np.nanmean(eq_matrix , axis = 0)

prior_expert = ru_bar

prior_item = rq_bar


# In[7]:

'''
w_a_u = expert * expert
'''
num_experts = eq_matrix.shape[0]
wau = np.zeros((num_experts,num_experts))

rated_questions = []

for exp in range(num_experts):
    faced = np.argwhere(~np.isnan(eq_matrix[exp]))
                        
    c=map(float, faced)
    #print type(c)
    rated_questions.append(c)


# In[25]:

#print rated_questions


# In[8]:

print (rated_questions[exp1])


# In[ ]:


for exp1 in range(num_experts):
    for exp2 in range(exp1+1):
        #print '.',
        wau[exp1][exp2] = 0
        
        common_qs = set().intersection(set(rated_questions[exp2]))
        nr = 0
        dr1 = 0
        dr2 = 0
        a_bar = ru_bar[exp1]
        u_bar = ru_bar[exp2]
        for q in common_qs:
            pa = (eq_matrix[exp1][q] - a_bar )
            pu =  (eq_matrix[exp2][q] - u_bar)
            nr +=  pa*pu
            dr1 += (pa)**2
            dr2 += (pu)**2
        
        dr = math.sqrt(dr1 * dr2)
        
        '''if  (a_bar == 0.0 and u_bar == 1.0)  or  (a_bar == 1.0 and u_bar == 0.0) :#when all ratings are the same 
            wau[exp1][exp2] = 0
            wau[exp2][exp1] = 0


        elif  (a_bar == 0.0 and u_bar == 0.0) or  (a_bar == 1.0 and u_bar == 1.0)  :#when all ratings are the same 
            wau[exp1][exp2] = 1
            wau[exp2][exp1] = 1
        
        else: 
       '''     #print nr, dr , a_bar, u_bar
        if dr == 0: 
            dr = 0.0005#??????????????
                
            wau[exp1][exp2] = 1.0*nr/dr
            wau[exp2][exp1] = wau[exp1][exp2]


# In[ ]:

'''
Time to make predicitons
'''
for u in prediction_dict.keys():
    for q in prediction_dict[u]:
        if eq_matrix[u][q]!=nan:
            r = predict_rating(u,q)
            eq_matrix[u][q] = r


# In[ ]:

avg_matrix = eq_matrix.mean()


# In[ ]:

import heapq
import numpy
K = 10000
def predict_ratings(a,q):
    global K
    
    if eq_matrix[a][q]!=nan:
        return eq_matrix[a][q]
    
    pred = 0
    a_bar = ru_bar[a]
    part2 = 0
    p2num = 0
    p2den = 0
    
    usrs = waq[a] #all users with curr user comb
    topK = heapq.nlargest(K, range(len(usrs)), usrs.take)
    for u_idx in topK:
        curr_wau = wau[a][u_idx]
        curr_dev = (eq_matrix[u_idx][q] - ru_bar[u_idx])
        p2num += curr_wau * curr_dev
        p2_den = wau[q][u_idx]
        
        part2 += p2num/p2den
        
    return a_bar + part2


# In[ ]:

#------can be repeated for training data
val_r_count = validation_data.shape[0]
probs = []
lines = []
for vrow in range(val_r_count):
    qid, uid = validation_data.ix[vrow][0] ,  validation_data.ix[vrow][1]
    question_index = question_list.index(curr_q)  
    expert_index = expert_list.index(curr_u)    
    curr_prob = 0
    
    if expert_index < eq_matrix.shape[0]: #user rating we have
        curr_prob = eq_matrix[uid][qid]
    else:
        if question_index < eq_matrix.shape[1]:
            curr_prob = prior_q[qid]
        else:
            print 'neither user nor q found'
            curr_prob = avg_matrix
        
    lines.append(qid+','+uid+curr_prob)


# In[ ]:

with open('CF_NEW_Validation.csv','w') as f:
    f.write('\n'.join(lines))

