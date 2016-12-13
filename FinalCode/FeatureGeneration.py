import pandas as pa

train_data = pa.read_csv("./bytecup2016data/invited_info_train.txt", sep ="\t", names=["qid", "uid", "label"])
validation_data = pa.read_csv("./bytecup2016data/validate_nolabel.txt", sep =",", names=["qid", "uid"])
test_data = pa.read_csv("./bytecup2016data/test_nolabel.txt", sep =",", names=["qid", "uid"])

# print train_question_features.shape
# print train_user_features.shape
# print train_data.shape

''' Notes '''
'''
#df['col_label']
#df.loc[row_index, col_label/col_index]
#df.iloc = ??
'''

def main():
    generateFeatures(train_data, 'training_data.txt')
    generateFeatures(validation_data, 'validation_data.txt')
    generateFeatures(test_data, 'test_data.txt')

def generateFeatures(data, filename):
    train_question_features = pa.read_csv("./bytecup2016data/question_info.txt", sep="\t",
                                          names=["qid", "tag", "q_wordid", "q_charid", "upvotes", "totalanswers", "topanswers"])
    train_user_features = pa.read_csv("./bytecup2016data/user_info.txt", sep="\t",
                                      names=["uid", "etags", "u_wordid", "u_charid"])

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

    ''' ================================================= '''

    # print '======== Adding q_wordid to training data ========'
    # print str(train_question_features['qid'].size) + 'size'
    # q_desc_arr = train_question_features['q_wordid'].str.split('/', expand=True).stack().unique()
    # print '# of unique Q words: '+ str(len(q_desc_arr))
    # for j in q_desc_arr:
    #     train_user_features[j] = 0
    #
    # i = 0
    # size = train_user_features['uid'].size
    # while (i < size):
    #     temp_int_arr = train_user_features.loc[i, 'q_wordid'].split('/')
    #     for temp_tag in temp_int_arr:
    #         train_user_features.loc[i, temp_tag] = 1
    #     i += 1
    # '''=========================================================='''

    print('====== Drop the aggregated tag columns from both user and question data frames')

    train_question_features = train_question_features.drop('q_wordid', axis=1)
    train_question_features = train_question_features.drop('q_charid', axis=1)
    train_user_features = train_user_features.drop('u_wordid', axis=1)
    train_user_features = train_user_features.drop('u_charid', axis=1)
    train_user_features = train_user_features.drop('etags', axis=1)
    train_question_features = train_question_features.drop('tag', axis =1)

    # print train_user_features
    # print train_question_features

    '''================================================================================='''
    label = None
    print '============== separate label from training data================='
    if 'label' in data.columns:
        label = data['label']
        print label
        data = data.drop('label', axis=1)
    '''======================================================================'''

    print '=========== Merge training info with user and question info =================='

    data = pa.merge(data, train_user_features, on='uid')
    data = pa.merge(data, train_question_features, on='qid')
    if label != None:
        data = pa.concat([data, label], axis = 1)
    '''==================================================================================='''

    print '============= Write training data to File ============'
    print data
    data.to_csv("./bytecup2016data/"+str(filename))
    print '======================================================= '''

main()