from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
import pandas as pa
import numpy as np


all_data = pa.read_csv("../../bytecup2016data/train_data.txt", sep = "\t")

df = all_data # pd.DataFrame(np.random.randn(100, 2))

msk = np.random.rand(len(df)) < 0.08

training_data = df[msk]
test = df[~msk]

gt_labels = []

(pa.DataFrame(test)).to_csv("validation.csv")

training_data =  training_data.reset_index(drop=True)
train_size = training_data.shape[0]

question_list = list(set(training_data.iloc[:,0]))
expert_list = list(set(training_data.iloc[:,1]))


for idx in range(train_size):
    q , u, label = training_data.ix[idx][0], training_data.ix[idx][1], training_data.ix[idx][2]
    qidx = question_list.index(q)
    uidx = expert_list.index(u)
    gt_labels.append(str(qidx)+","+str(uidx)+","+str(label))



with open('../../lists2/gt_labels.txt','w') as f:
    f.write('\n'.join(gt_labels))

with open('../../lists2/question_list.txt','w') as f:
    f.write('\n'.join(question_list))

with open('../../lists2/user_list.txt','w') as f:
    f.write('\n'.join(expert_list))




#---Spark can start here if data gets loaded fully
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
import pandas as pa
import numpy as np

ALS.checkpointInterval = 2
                  
sc.setCheckpointDir('checkpoint/')
# Load and parse the data
data = sc.textFile("../../lists2/gt_labels.txt")
ratings = data.map(lambda l: l.split(','))\
    .map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

# Build the recommendation model using Alternating Least Squares
rank = 40
numIterations = 350
model = ALS.train(ratings, rank, numIterations)



#Evaluate the model on training data
testdata = ratings.map(lambda p: (p[0], p[1]))
predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Mean Squared Error = " + str(MSE))

# Save and load model
model.save(sc, "target/tmp/myCollaborativeFilter102")



sameModel = MatrixFactorizationModel.load(sc, "target/tmp/myCollaborativeFilter102")

#validation
validation_data = pa.read_csv("../../validation.csv", sep = ",")







sameModel = MatrixFactorizationModel.load(sc, "target/tmp/myCollaborativeFilter101")
validation_data = pa.read_csv("../../bytecup2016data/validate_nolabel.csv", sep = ",")

#***********after this step remove the first column



with open('../../lists2/question_list.txt') as f:
    question_list = f.read().splitlines()

with open('../../lists2/user_list.txt') as f:
    user_list = f.read().splitlines()

validation_size = validation_data.shape[0]
prob_train = []

for idx in range(validation_size):
	try:
			q , u = validation_data.ix[idx][0], validation_data.ix[idx][1]
			toAppend = 0
			if q in question_list and u in user_list:
				qidx = question_list.index(q)
				uidx = user_list.index(u)
				toAppend = sameModel.predict(qidx,uidx)
				if toAppend < 0:
					toAppend = 0
				if toAppend > 1:
					toAppend = 1
				prob_train.append(toAppend)
				print 'found in the matrix and the list'
			else:
				prob_train.append(0)
	except:
			pass

final_validation = np.c_[ np.array(validation_data) , prob_train]
pa.DataFrame(final_validation).to_csv('final1.csv')

