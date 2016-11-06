
# coding: utf-8

# In[115]:

import pandas as pa
import numpy as np
#collaborative filtering
#train_file = open("../bytecup2016data/train_data.txt")
train_data = pa.read_csv("../bytecup2016data/train_data_mini.txt", sep ="\t")
#print train_data

train_features = pa.read_csv("../bytecup2016data/question_info.txt", sep ="\t")

print train_data.shape


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


# In[117]:

#using the question info, get its category from the other dataframe
#we have a question and category

expert_category_map = train_data.ix[:,[0,1]].merge(train_features.ix[:,[0,1]], left_on='question_ID', right_on='question_ID1', how='left').ix[:,[1,3]]
print expert_category_map.shape


# In[79]:

#convert above into [expert,  ]
maxCat = max(list(expert_category_map.ix[:,1]))
print maxCat

expert_list = (list(expert_category_map.ix[:,0].unique()))

print len(expert_category_map[expert_category_map.isnull().any(axis=1)])


# In[121]:

expert_category_matrix = np.zeros((numExperts, maxCat+1))

total_expert_cat_entries = len(expert_category_map)

experts_found = []

expert_category_map = expert_category_map.dropna()

for row_num in range(total_expert_cat_entries-1):
    #print expert_category_map.iloc[row_num][0]
    exp = expert_list.index( expert_category_map.iloc[row_num][0])    
    cat = expert_category_map.iloc[row_num][1]      
    expert_category_matrix[exp][cat] += 1
        


# In[122]:

#print expert_category_matrix[:5]
print expert_category_matrix.shape


# In[83]:

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
        if e < 0.1:
            break
    return P, Q.T


# In[84]:

R = np.array(expert_category_matrix)
N = len(R)
M = len(R[0])
K = 2

P = np.random.rand(N,K)
Q = np.random.rand(M,K)
nP, nQ = matrix_factorization(R, P, Q, K)
nR = np.dot(nP, nQ.T)


# In[ ]:

# for each expert in train set, output the score in the table for that [expert][category]
#compare it to the output in the train file
#this gives train accuracy
print nR


# In[16]:

print R
#convert test data into expert_tag table
# for each expert, output the score in the table for that [expert][category]
#compare it to the output in the test file


# In[19]:

np.savetxt("myCalcArr.txt",nR)


# In[128]:

print nR.shape
print type(nR)
x= nR
x_normed = x / x.max(axis=0)
y_normed = x/x.max(axis=1).reshape(nR.shape[0],1)
print y_normed


# In[126]:

print x_normed
print x[0]


# In[87]:

#now validate your array results 
#validation_data = pa.read_csv("validation_mini.txt",sep ='\t')

validation_data = pa.read_csv("../bytecup2016data/validation_mini.txt", sep ="\t")


# In[90]:

#get training accuracy 


#we have q, user 

trainAccSet = train_data.iloc[:,[0,1]]
#find cat for the q and read the val from the yhat matrix

#for i in range(train_data.shape(0)):
    
#op raw values and actual values side by side
#now try to set a threshold
#compare the number of 0s and 1s predicted to the actual data values
#measure accuracy


# In[129]:

#print expert_category_map
prob = []
for i in range(expert_category_map.shape[0]):
    expertID = expert_category_map.iloc[i][0]
    category = expert_category_map.iloc[i][1]
    exp = expert_list.index(expertID)
    prob.append(y_normed[exp][category])  
    #exp = expert_list.index( expertID)    
print prob



# In[130]:

print expert_category_matrix.shape


# In[132]:

c =np.c_[expert_category_map, prob]
#print len(prob)
#print expert_category_map.shape
#print nR.shape
#print c

cd = pa.DataFrame(c,columns = ["expert_ID", "category","probability"])

print cd


# In[154]:

expert_question_map = train_data.ix[:,[0,1,2]].merge(cd.ix[:,[0,1,2]], left_on='expert_ID', right_on='expert_ID', how='inner').ix[:,[0,1,2,4]]



cj = train_data.ix[:,[0,1,2]].join(cd.ix[:,[0,1,2]], on='expert_ID', how='inner',  lsuffix='_left', rsuffix='_right', sort=False)
  
print cj.shape
print cj


print expert_question_map.shape

print train_data.shape

print cd.shape
#print expert_question_map#.ix[:,[0,1]]


# In[149]:

gv= np.array(expert_question_map)
print type(gv)
np.savetxt("myFinalPreds.txt",gv)


# In[ ]:



