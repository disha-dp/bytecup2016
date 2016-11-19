
# coding: utf-8

# In[1]:

import pandas as pa
import numpy as np
import sklearn
from sklearn import preprocessing


# In[2]:

all_data = pa.read_csv("../bytecup2016data/training_data.txt")


# In[3]:

#only_pos = all_data.loc[all_data['label'] == 1]


# In[3]:

train_data = all_data


# In[6]:

to_norm =  train_data.iloc[:,146:149]
print max(to_norm.iloc[:,0])


# In[7]:

min_max_scaler = preprocessing.MinMaxScaler()
norm_cols = min_max_scaler.fit_transform(to_norm)


# In[23]:

#print pa.DataFrame(norm_cols)


# In[4]:

train_half2 = train_data.iloc[:,149:171]


# In[5]:

train_half3 = train_data.iloc[:,0:146]


# In[8]:

train_half1 = pa.DataFrame(norm_cols)


# In[9]:

#print train_half1


# In[10]:

#print train_half2


# In[11]:

train_inter = pa.concat([train_half1, train_half2], axis = 1)


# In[12]:

#print train_inter


# In[13]:

final_train_data = pa.concat([train_half3, train_inter], axis = 1)


# In[14]:

#print final_train_data


# In[15]:

pa.DataFrame(final_train_data).to_csv("final_normalized_train.csv")


# In[47]:

#train_features = final_train_data.ix[:, final_train_data.columns != 'label']
train_features = train_inter
train_labels = final_train_data.ix[:, final_train_data.columns == 'label']


# In[48]:

from sklearn import datasets
from sklearn import metrics
from sklearn import svm


# In[49]:

model = svm.SVC(C=0.01, kernel='rbf', degree=3, gamma=0.25, coef0=0.0, shrinking=True, probability=True, tol=0.0001, 
cache_size=200000, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None, random_state=None)


# In[ ]:

#model =  svm.SVC( probability=True , verbose = True)


model.fit(train_features, train_labels )

print(model)

# make predictions
expected = train_labels

predicted = model.predict(train_features)
# summarize the fit of the model

print(metrics.classification_report(expected, predicted))

print'auc: ',(metrics.auc(expected, predicted))


#print(metrics.confusion_matrix(expected, predicted))


# In[15]:

print predicted


# In[ ]:

print model


# In[ ]:

save_train = all_data[:,140:143]


# In[ ]:

for ro

