import nimfa

import pandas as pa
import numpy as np

train_data = pa.read_csv("train_data.txt", sep ="\t")

train_features = pa.read_csv("question_info.txt", sep ="\t")

print train_data.shape
import os
os.system("taskset -p 0xff %d" % os.getpid())


# In[119]:

'''
1. make a table with users as rows
2. columns = categories
3. entry = num of times user has answered question in category= col 
'''
#get all distinct users from the train file

allUsers = train_data.ix[:,1]#getting second column
allQuestions = train_data.ix[:,0]#getting first column

allUniqueUsers = allUsers.unique()
print 'total users: ',len(allUsers)
print 'unique users: ',len(allUniqueUsers)
print 'total questions in training set: ',len(allQuestions)


numExperts = len(allUniqueUsers)


# In[298]:

#using the question info, get its category from the other dataframe
#we have a question and category

expert_category_map = train_data.ix[:,[0,1]].merge(train_features.ix[:,[0,1]], left_on='question_ID', right_on='question_ID1', how='left').ix[:,[1,3]]
print expert_category_map.shape


# In[299]:

#convert above into [expert,  ]
maxCat = max(list(expert_category_map.ix[:,1]))
print maxCat

expert_list = (list(expert_category_map.ix[:,0].unique()))

print len(expert_category_map[expert_category_map.isnull().any(axis=1)])


# In[301]:

expert_category_matrix = np.zeros((numExperts, int(maxCat)+1))


experts_found = []

expert_category_map = expert_category_map.dropna()
total_expert_cat_entries = len(expert_category_map)

for row_num in range(total_expert_cat_entries-1):
    #print expert_category_map.iloc[row_num][0]
    exp = expert_list.index( expert_category_map.iloc[row_num][0])    
    cat = int(expert_category_map.iloc[row_num][1])  
    expert_category_matrix[exp][cat] += 1
        


# In[302]:

print expert_category_matrix.shape


# In[303]:

def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in xrange(steps):
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                    for k in xrange(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = np.dot(P,Q)
        e = 0
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in xrange(K):
                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        if e < 0.01:
            break
    return P, Q.T


# In[84]:
print '#################    computing the R hat matrix'
print 'approach SGD for computing R hat'

R = np.array(expert_category_matrix)
N = len(R)
M = len(R[0])
K = 10

P = np.random.rand(N,K)
Q = np.random.rand(M,K)
nP, nQ = matrix_factorization(R, P, Q, K)
nR = np.dot(nP, nQ.T)



print '#################computing the R hat matrix'
print 'approach nimfa , given library for computing R hat'

V = R   #nimfa.examples.medulloblastoma.read(normalize=True)

lsnmf = nimfa.Lsnmf(V, seed='random_vcol', rank=50, max_iter=100)
lsnmf_fit = lsnmf()

print('Rss: %5.4f' % lsnmf_fit.fit.rss())
print('Evar: %5.4f' % lsnmf_fit.fit.evar())
print('K-L divergence: %5.4f' % lsnmf_fit.distance(metric='kl'))
print('Sparseness, W: %5.4f, H: %5.4f' % lsnmf_fit.fit.sparseness())


print '#################   Finished computing the R hat matrix'


# for each expert in train set, output the score in the table for that [expert][category]
#compare it to the output in the train file
#this gives train accuracy
print nR
#convert test data into expert_tag table
# for each expert, output the score in the table for that [expert][category]
#compare it to the output in the test file


# In[306]:
print 'saving file'

np.savetxt("myCalcArr.txt",nR)


# In[307]:

print nR.shape
print type(nR)
x= nR
x_normed = x / x.max(axis=0)
y_normed = x/x.max(axis=1).reshape(nR.shape[0],1)
print y_normed


# In[308]:

print x_normed
print x[0]


# In[312]:

#get training accuracy 


#we have q, user 

trainAccSet = train_data.iloc[:,[0,1]]
#find cat for the q and read the val from the yhat matrix

#for i in range(train_data.shape(0)):
    
#op raw values and actual values side by side
#now try to set a threshold
#compare the number of 0s and 1s predicted to the actual data values
#measure accuracy


# In[313]:

#print expert_category_map
prob = []
for i in range(int(expert_category_map.shape[0])):
    expertID = (expert_category_map.iloc[i][0])
    category = int(expert_category_map.iloc[i][1])
    exp = expert_list.index(expertID)
    prob.append(y_normed[exp][category])  
    
    
print len(prob)



# In[336]:

print expert_category_matrix.shape


# In[337]:

#now validate your array results 
#validation_data = pa.read_csv("validation_mini.txt",sep ='\t')

validation_data = pa.read_csv("validate_nolabel.txt", sep =",")


# In[362]:

rValid = validation_data.shape[0]
cValid = validation_data.shape[1]

validation_QEC_map = validation_data.ix[:,[0,1]].merge(train_features.ix[:,[0,1]], left_on='question_ID2', right_on='question_ID1', how='left').ix[:,[0,1,3]]

#print validation_QEC_map

rV = validation_QEC_map.shape[0]
cV = validation_QEC_map.shape[1]

out = []
for entry in range(rV):
    #print  validation_QEC_map.ix[entry][1],  validation_QEC_map.ix[entry][2]
    
    exp = validation_QEC_map.ix[entry][1]
    cat =  validation_QEC_map.ix[entry][2]
    
    try:
        catInt = int(cat)

        if exp in expert_list:
            exp_idx = expert_list.index(exp)
            prob = x_normed[exp_idx][catInt]
        else: 
            prob = 0
    except:
        prob = 0
    out.append(prob)

print len(out)


# In[392]:

import math
import numpy as np
#print map(abs,out)

final_res = np.c_[validation_data, out]

cn = np.array(final_res)

print type(cn)

final_DF = pa.DataFrame(cn)

final_DF.to_csv("final.csv", sep =",")


# In[ ]:



