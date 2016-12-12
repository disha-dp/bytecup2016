
# coding: utf-8

# In[1]:

import graphlab as gl
import pandas as pa


# In[2]:

training_data = pa.read_csv("../bytecup2016data/invited_info_train.txt", sep = "\t", names=["qid","uid","label"] )
question_data = pa.read_csv("../bytecup2016data/question_info.txt", sep = "\t", names=["qid","cat","word_tags","char_tags","upvotes","tot_ans","top_q_ans"])
test_data = pa.read_csv("../bytecup2016data/test.csv", sep = ",")#,names = ["qid","uid","label"])
validation_data = pa.read_csv("../bytecup2016data/validate_nolabel.txt", sep = ",")

user_data = pa.read_csv("../bytecup2016data/user_info.txt", sep = "\t", names = ["uid","tags","word_seq","char_seq"])


# In[3]:

qList =  training_data['qid'].tolist()
eList =  training_data['uid'].tolist()
labelList =  training_data['label'].tolist()


# In[4]:

all_data = {}
all_data['user_id'] = eList
all_data['item_id'] = qList
all_data['rating'] = labelList


# In[5]:

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
word_seq = []

rows_train = training_data.shape[0]

for i in range(rows_train):
    qid = training_data.ix[i][0]
    qinfo_row = question_data.loc[question_data['qid'] == qid]
    if not qinfo_row.empty:
        word_seq.append(qinfo_row.iloc[0][2].split('/'))
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


# In[6]:

from collections import Counter

#user info
word_seq=[]
rows_train = training_data.shape[0]

for i in range(rows_train):
    uid = training_data.ix[i][1]
    uinfo_row = user_data.loc[user_data['uid'] == uid]
    if not uinfo_row.empty:
        str_list = uinfo_row.iloc[0][1].split('/')
        str_diction = Counter(str_list)
        word_list = uinfo_row.iloc[0][2].split('/')
        word_dictionary = Counter(word_list)
        int_list = map(int,str_list)

        word_seq.append(word_dictionary)
        word_seq_list.append(word_list)
    else:
        word_seq.append({})
        


# In[13]:

sf = gl.SFrame({'user_id': eList, 'item_id': qList, 'rating': labelList})


# In[14]:

user_info = gl.SFrame({'user_id': eList,'word_seq': word_seq})


# In[15]:

item_info  = gl.SFrame({'item_id': qList,'total_ans' : tot_ans ,  'top_q_ans': top_q_ans ,'upv': upvotes  })


# In[18]:

sf= gl.SFrame({'item_id': qList, 'user_id': eList, 'rating': labelList})


# In[31]:

#-------- make prediction with side features 
#trainI, testI = gl.recommender.util.random_split_by_user( sf , max_num_users= 15000, item_test_proportion= 0.2, random_seed=0)

# binary_target , ranking
#mI = gl.factorization_recommender.create(trainI, target='rating',user_data = user_info, item_data= item_info, num_factors=20, regularization=0.01, max_iterations=1000 ,sgd_step_size=0.4  )

#print '----------------MEASUREMENT ON TEST WITH SIDE FEATURES-------------'

#evalsI = mI.evaluate(testI)


# In[20]:

#generate validate results
qvList =  validation_data['qid'].tolist()
evList =  validation_data['uid'].tolist()

sfValidate = gl.SFrame({'user_id': evList,
                       'item_id': qvList})


# In[21]:

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


# In[24]:

with open('FINAL_1.csv','w') as f:
    f.write('\n'.join(map(str,test_preds)))


# In[25]:

with open('temp_LAST.csv','w') as f:
    f.write('\n'.join(map(str,validation_preds)))

