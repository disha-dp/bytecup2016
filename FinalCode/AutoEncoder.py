import pandas as pd
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model

def main():
    X = getQuestionWordDescription()
    getReducedFeature(X)

def getReducedFeature(X):
    X = X.transpose()
    print "Shape of transpose word matrix: " +str(X.shape)
    # this is the size of our encoded representations
    encoding_dim = 200  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

    # this is our input placeholder
    input_img = Input(shape=(8087, 1))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='sigmoid')(input_img)
    # print encoded

    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(8087, activation='sigmoid')(encoded)

    # this model maps an input to its reconstruction

    # input_img = Input(shape=(784,))
    # encoded = Dense(128, activation='relu')(input_img)
    # encoded = Dense(64, activation='relu')(encoded)
    # encoded = Dense(32, activation='relu')(encoded)
    #
    # decoded = Dense(64, activation='relu')(encoded)
    # decoded = Dense(128, activation='relu')(decoded)
    # decoded = Dense(784, activation='sigmoid')(decoded)
    # Let's try this:

    autoencoder = Model(input=input_img, output=decoded)
    # TO DO: Optimizer - to be changed to RProp or something else
    autoencoder.compile(optimizer='sgd', loss='mean_squared_error')

    '''
    # CODE FOR SPLITTING TEST DATA FROM TRAINING DATA

    # df = all_data # pd.DataFrame(np.random.randn(100, 2))
    # msk = np.random.rand(len(df)) < 0.08
    # training_data = df[msk]
    # test = df[~msk]
    # gt_labels = []
    # (pa.DataFrame(test)).to_csv("validation.csv")
    '''

    msk = np.random.rand(len(X)) < 0.08
    x_train = X[msk]
    x_test = X[~msk]

    autoencoder.fit(x_train, x_train,
                    nb_epoch=100,
                    batch_size=256,
                    shuffle=True,
                    validation_data = (x_test, x_test))


def getQuestionWordDescription():
    train_question_features = pd.read_csv("./bytecup2016data/question_info.txt", sep="\t",
                                          names=["qid", "tag", "q_wordid", "q_charid", "upvotes", "totalanswers",
                                                 "topanswers"])

    print str(train_question_features['qid'].size) + 'size'
    q_word_arr = train_question_features['q_wordid'].str.split('/', expand=True).stack().unique()

    for j in q_word_arr:
        train_question_features[j] = 0

    i = 0
    size = train_question_features['qid'].size
    while (i < size):
        temp_int_arr = train_question_features.loc[i, 'q_wordid'].split('/')
        for temp_tag in temp_int_arr:
            train_question_features.loc[i, temp_tag] = 1
        i += 1
    # df.drop('column_name', axis=1, inplace=True)
    train_question_features.drop(["qid", "tag", "q_wordid", "q_charid", "upvotes", "totalanswers",
                                                 "topanswers"],  axis=1, inplace=True)
    print " Shape of the Word Matrix: "+ str(train_question_features.shape)
    return train_question_features

main()