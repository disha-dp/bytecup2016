import numpy as numpy
import pandas as pa
from sklearn import svm
from sklearn.model_selection import cross_val_score

print 'reading training data'

model_data = pa.read_csv("training_data.txt",sep = ",")

trimmed_data = model_data

trimmed_data.drop(trimmed_data.columns[[0,1,2]], axis=1, inplace=True)

all_gammas = [1.0,0.1,0.01,0.001]
Cs = [1,10,100]
print 'starting SVM'
for gamma in all_gammas:
	for C in  Cs:

		clf = svm.SVC(gamma=gamma, C=C)

		X = trimmed_data.ix[:, trimmed_data.columns != 'label']
		Y =  trimmed_data['label']
		print '-----------------For Gamma: {0} and C: {1}---------------------------'.format(gamma, C)

		clf.fit(X,Y) 

		print clf.accuracy_score

		print clf.roc_auc_score

		print cross_val_score(clf, X, Y, scoring='roc_auc')
		print cross_val_score(clf, X, Y, scoring='average_precision')
