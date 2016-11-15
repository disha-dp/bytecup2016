import pandas as pa

train_data = pa.read_csv("./bytecup2016data/invited_info_train.txt", sep ="\t", names=["qid", "uid", "label"])
train_question_features = pa.read_csv("./bytecup2016data/question_info.txt", sep ="\t", names = ["qid", "tag", "wordid", "charid", "upvotes", "totalanswers", "topanswers"])
train_user_features = pa.read_csv("./bytecup2016data/user_info.txt", sep ="\t", names=["uid", "etags", "wordid", "charid"])

# print train_question_features.shape
# print train_user_features.shape
# print train_data.shape

''' Notes '''
'''
#df['col_label']
#df.loc[row_index, col_label/col_index]
#df.iloc = ??
'''

print '====== Modifying Question dataframe ======='
print str(train_question_features['qid'].size) + 'size'
q_tag_arr = train_question_features['tag'].unique()
#print q_tag_arr
for j in q_tag_arr:
    train_question_features[j] = 0

i=0
for i in range(train_question_features['qid'].size):
    train_question_features.loc[i, train_question_features.loc[i, 'tag']] = 1
    i += 1
# print train_question_features
'''================================================='''


print '======= Modifying Expert features ========='
print str(train_user_features['uid'].size) + 'size'
e_tag_arr = train_user_features['etags'].str.split('/', expand=True).stack().unique()

for j in e_tag_arr:
    train_user_features[j]=0

i=0
size =  train_user_features['uid'].size
while(i < size):
    temp_int_arr = train_user_features.loc[i, 'etags'].split('/')
    for temp_tag in temp_int_arr:
        train_user_features.loc[i, temp_tag] = 1
    i += 1

# print train_user_features
''' ================================================= '''


''' Drop the aggregated 'tag' columns from both user and question data frames '''
train_user_features = train_user_features.drop('etags', axis=1)
train_question_features = train_question_features.drop('tag', axis =1)
# print train_user_features
# print train_question_features

train_data = pa.merge(train_data, train_user_features, on='uid')
print train_data