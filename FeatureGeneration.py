import pandas as pa

train_data = pa.read_csv("./bytecup2016data/mini_train_data.txt", sep ="\t", names=["qid", "uid", "label"])
train_question_features = pa.read_csv("./bytecup2016data/mini_ques_info.txt", sep ="\t", names = ["qid", "tag", "wordid", "charid", "upvotes", "totalanswers", "topanswers"])
train_user_features = pa.read_csv("./bytecup2016data/mini_user_info.txt", sep ="\t", names=["uid", "etags", "wordid", "charid"])

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
print train_question_features
'''================================================='''


print '======= Modifying Expert features ========='
print str(train_user_features['uid'].size) + 'size'
e_tag_arr = train_user_features['etags'].str.split('/', expand=True).stack().unique()

for j in e_tag_arr:
    train_user_features[j]=0

i=0
while i < train_user_features['uid'].size:
    train_user_features
