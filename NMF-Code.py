import pandas as pa
import numpy as np
from pandas import DataFrame
import numpy as np
from sklearn.decomposition import ProjectedGradientNMF

train_data = pa.read_csv("train_data.txt", sep ="\t", header='infer')
#create the matrix with  np.nan values


question_list = list(set(train_data.iloc[:,0]))
expert_list = list(set(train_data.iloc[:,1]))


r = len(question_list) #num of distinct questions
c = len(expert_list) #num of distinct users




matrix = np.zeros((r,c))
matrix.fill(np.nan)

for row in range(train_data.shape[0]):
    row_info = train_data.iloc[row]
    curr_q , curr_u, label =  row_info[0], row_info[1], row_info[2] 

    #print curr_q, curr_u
    question_index = question_list.index(curr_q)  
    user_index = expert_list.index(curr_u)
    
    matrix[question_index][user_index] =  label
    #print  question_index, user_index

print matrix[1053][7094]
#matrix[question_index][user_index] , matrix[question_index][user_index+1] #


####THEIRS- not needed
# Example data matrix X

###MINE
X = DataFrame(matrix)
X_imputed = X.copy()
X = pa.DataFrame(matrix)# DataFrame(toy_vals, index = range(nrows), columns = range(ncols))
###use some way to mask only a few vals.... thst too either 0 or 1
msk = (X.values + np.random.randn(*X.shape) - X.values) < 0.8
X_imputed.values[~msk] = 0


##THEIRS

# Hiding values to test imputation
# Initializing model
nmf_model = ProjectedGradientNMF(n_components = 600, init='nndsvda', random_state=0,max_iter=300, eta=0.01, alpha = 0.01)
nmf_model.fit(X_imputed.values)

# iterate model
#while nmf_model.reconstruction_err_**2 > 10:
    #nmf_model = NMF( n_components = 600, init='nndsvda', random_state=0,max_iter=300, eta=0.01, alpha = 0.01)
W = nmf_model.fit_transform(X_imputed.values)
X_imputed.values[~msk] = W.dot(nmf_model.components_)[~msk]
print nmf_model.reconstruction_err_

H = nmf_model.components_
rHat = np.dot(W,H)
np.savetxt("rHat.txt" ,rHat) 
