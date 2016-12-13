import pandas as pd
from sklearn import tree
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split

def main():
    X, Y = getTrainingData()
    clf = trainModel(X, Y)
    train_acc = calculateTrainingAccuracy(clf, X, Y)
    print 'train_acc: ' + str(train_acc)

    X_Valid = getValidationData()
    pred_OnValidation = predict(clf, X_Valid)
    df_predictedValidation = pd.DataFrame(X_Valid, columns=['qid', 'uid'])
    df_predictedValidation['label'] = pred_OnValidation[:,[1]]
    generateFile(df_predictedValidation)


    X_Test = getTestData()
    pred_OnTest = predict(clf, X_Test)
    df_predictedTest = pd.DataFrame(X_Test, columns=['qid', 'uid'])
    df_predictedTest['label'] = pred_OnTest[:,[1]]
    generateFile(df_predictedTest)

def getTrainingData():
    X = pd.read_csv('./bytecup2016data/training_data.txt', sep=',')
    X.drop(X.columns[[0, 1, 2]], axis=1, inplace=True)

    Y = map(float,X['label'])
    X.drop('label', axis=1, inplace=True)

    return X, Y

def calculateTrainingAccuracy(clf, X, Y):
    label = predict(clf, X)

    # print type(label)
    label = label.astype(int)
    # print label[:5]
    # print np.array_equal(Y, label)
    np.savetxt('decisiontree/truelabel.txt', Y)
    np.savetxt('decisiontree/predlabel.txt', label)
    # print label.shape
    noOfMatched = (Y == label).sum()
    # print str(noOfMatched) + 'match count'
    totalSamples = len(X.index)
    # print totalSamples
    trainingAccuracy = (noOfMatched/totalSamples)*100
    print trainingAccuracy
    return trainingAccuracy

def trainModel(X, Y):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)

    return clf

def predict(model, data):
    label = model.predict_proba(data)
    print '@@@',label.sum(axis=0)
    # print label

    return label

def calculateTestAccuracy(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=0)
    clf = trainModel(X_train, y_train)
    scores = clf.score(X_test, y_test)

    return scores

def calculateCrossValidationScore(model, X, Y):
    cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
    clf = tree.DecisionTreeClassifier()
    scores = cross_val_score(model, X, Y, cv=cv, scoring='roc_auc')
    return scores

def getValidationData():
    X = pd.read_csv('./bytecup2016data/validation_data.txt', sep=',')
    X.drop(X.columns[[0, 1, 2]], axis=1, inplace=True)

    return X

def getTestData():
    X = pd.read_csv('./bytecup2016data/test_data.txt', sep=',')
    X.drop(X.columns[[0, 1, 2]], axis=1, inplace=True)

    return X

def generateFile(y_pred_validation):
    validation_data = pd.read_csv("./bytecup2016data/validate_nolabel.txt", sep = "\t", names=["qid","uid","label"] )
    pred_df= y_pred_validation['label']
    result = pd.concat([validation_data.ix[:,0:2], pred_df], axis=1)
    result.to_csv('DecisionTree_ValidationOutput.csv', index=False, columns=["qid","uid","label"])

def generateTestPredictions(y_pred_test):
    test_data = pd.read_csv("./bytecup2016data/test_nolabel.txt", sep="\t",
                                          names=["qid", "uid", "label"])
    pred_df = y_pred_test['label']
    result = pd.concat([test_data.ix[:, 0:2], pred_df], axis=1)
    result.to_csv('DecisionTree_TestOutput.csv', index=False, columns=["qid", "uid", "label"])

main()