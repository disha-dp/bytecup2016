
# coding: utf-8

# In[6]:

import graphlab as gl
import pandas as pa


# In[7]:

training_data = pa.read_csv("../bytecup2016data/invited_info_train.txt", sep = "\t", names=["qid","uid","label"] )
question_data = pa.read_csv("../bytecup2016data/question_info.txt", sep = "\t", names=["qid","cat","word_tags","char_tags","upvotes","tot_ans","top_q_ans"])
test_data = pa.read_csv("../bytecup2016data/test.csv", sep = ",")#,names = ["qid","uid","label"])
validation_data = pa.read_csv("../bytecup2016data/validate_nolabel.txt", sep = ",")

user_data = pa.read_csv("../bytecup2016data/user_info.txt", sep = "\t", names = ["uid","tags","word_seq","char_seq"])


# In[8]:

qList =  training_data['qid'].tolist()
eList =  training_data['uid'].tolist()
labelList =  training_data['label'].tolist()


# In[9]:

all_data = {}
all_data['user_id'] = eList
all_data['item_id'] = qList
all_data['rating'] = labelList


# In[10]:

#item info
'''
---> column popularAnswers
--->        number of total answers
'''
upvotes = []
tot_ans = []
top_q_ans = []
category = []
category_str = []
######### list of lists, each list corresponding to one user
'''user_tags = []
   user_tags.append(qinfo_row.iloc[1][1].split('/'))
'''
word_seq = []

rows_train = training_data.shape[0]

for i in range(rows_train):
    qid = training_data.ix[i][0]
    qinfo_row = question_data.loc[question_data['qid'] == qid]
    if not qinfo_row.empty:
        word_seq.append(qinfo_row.iloc[0][2].split('/'))
#         print word_seq
#         raw_input()
        upvotes.append(qinfo_row.iloc[0][4])
        tot_ans.append(qinfo_row.iloc[0][5])
        top_q_ans.append(qinfo_row.iloc[0][6])
        category.append(qinfo_row.iloc[0][1])
        category_str.append(str(qinfo_row.iloc[0][1]))

    else:
        word_seq.append([])
        upvotes.append(0)
        tot_ans.append(0)
        top_q_ans.append(0)
        category.append(0)
        category_str.append("0")


# In[11]:

from collections import Counter

#user info
user_interests=[]
user_interests_Str = []
user_interests_diction = []
char_seq = []
word_seq=[]
word_seq_list=[]
rows_train = training_data.shape[0]

for i in range(rows_train):
    uid = training_data.ix[i][1]
    uinfo_row = user_data.loc[user_data['uid'] == uid]
    if not uinfo_row.empty:
        str_list = uinfo_row.iloc[0][1].split('/')
        str_diction = Counter(str_list)
        word_list = uinfo_row.iloc[0][2].split('/')
        word_dictionary = Counter(word_list)
#         print dictionary
#         raw_input()
#         print str_list
#         print type(str_list)
        int_list = map(int,str_list)
#         print int_list
#         raw_input()

        user_interests.append(int_list)
        user_interests_Str.append(str_list)
        word_seq.append(word_dictionary)
        user_interests_diction.append(str_diction)
        word_seq_list.append(word_list)
    else:
        user_interests.append([])
        user_interests_Str.append([])
        word_seq.append({})
        word_seq_list.append([])
        user_interests_diction.append({})
        
# Counter({'red': 3, 'apple': 2, 'pear': 1})


# In[12]:

sf = gl.SFrame({'user_id': eList, 'item_id': qList, 'rating': labelList})


# In[ ]:

#-------- make prediction without any side features, only latent features 
#train, test = gl.recommender.util.random_split_by_user(sf, max_num_users= 20000, item_test_proportion=0.2, random_seed=0)

#m = gl.factorization_recommender.create(train, target='rating',item_data=None, max_iterations=500, regularization=1e-01)
#print 'MEASUREMENT ON TEST, PLAIN'
#evals = m.evaluate(test)


# In[13]:

user_info = gl.SFrame({'user_id': eList,'word_seq': word_seq})


# In[14]:

#item_info  = gl.SFrame({'item_id': qList,'total_ans' : tot_ans ,  'top_q_ans': top_q_ans ,'upv': upvotes , 'cat' : category,'ques_desc':word_seq})# , 'cat': catStr, 'top_q_ans' : topQ ,'tot_ans': totAns  })#,'cat_str' : catStrTrain  })

# item_info  = gl.SFrame({'item_id': qList,'total_ans' : tot_ans ,  'top_q_ans': top_q_ans ,'upv': upvotes  })


item_info  = gl.SFrame({'item_id': qList,'total_ans' : tot_ans ,  'top_q_ans': top_q_ans ,'upv': upvotes  })


# In[15]:

sf= gl.SFrame({'item_id': qList, 'user_id': eList, 'rating': labelList})


# In[ ]:

#-------- make prediction with side features 
#trainI, testI = gl.recommender.util.random_split_by_user( sf , max_num_users= 15000, item_test_proportion= 0.2, random_seed=0)

#regularization=1e-09, linear_regularization=1e-09, side_data_factorization=True, ranking_regularization=0.25, 
#mI = gl.factorization_recommender.create(trainI, target='rating',item_data= item_info, max_iterations=1000, regularization=1e-02, linear_regularization=1e-09, side_data_factorization=True, ranking_regularization=0.25) 

'''
ranking_factorization_recommender.
'''

# binary_target , ranking
#mI = gl.factorization_recommender.create(trainI, target='rating',user_data = user_info, item_data= item_info, num_factors=20, regularization=0.01, max_iterations=1000 ,sgd_step_size=0.4  )

#print 'MEASUREMENT ON TEST WITH SIDE FEATURES-------------'

#evalsI = mI.evaluate(testI)


# In[16]:

#generate validate results
qvList =  validation_data['qid'].tolist()
evList =  validation_data['uid'].tolist()

sfValidate = gl.SFrame({'user_id': evList,
                       'item_id': qvList})


# In[18]:

#generate train results
#qtrList =  training_data['qid'].tolist()
#etrList =  training_data['uid'].tolist()

#sfTrain = gl.SFrame({'user_id': etrList,
#                       'item_id': qtrList})


# In[19]:

# with feature and reg
item_info_val  = gl.SFrame({'item_id': qList,'total_ans' : tot_ans ,  'top_q_ans': top_q_ans ,'upv': upvotes  })

m_full_train = gl.factorization_recommender.create(sf, target='rating',user_data = user_info, item_data= item_info, num_factors=20, regularization=0.01, max_iterations=1000 ,sgd_step_size=0.4  )


# In[22]:

validation_preds = m_full_train.predict(  sfValidate  )


# In[23]:

qtList =  test_data['qid'].tolist()
etList =  test_data['uid'].tolist()

sfTest = gl.SFrame({'user_id': etList ,'item_id': qtList})

test_preds = m_full_train.predict(  sfTest  )


# In[20]:

#train_preds = m_full_train.predict(  sfTrain  )


# In[24]:

with open('finally.csv','w') as f:
    f.write('\n'.join(map(str,test_preds)))


# In[25]:

with open('temporarily.csv','w') as f:
    f.write('\n'.join(map(str,validation_preds)))


# In[21]:

#with open('train_vals.csv','w') as f:
#    f.write('\n'.join(map(str,train_preds)))


# In[ ]:



