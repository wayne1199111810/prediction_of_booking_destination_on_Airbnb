# import matplotlib.pyplot as plt
from include import preProcessing as pp
from include.utility import *
import include.crossValidation as cv
from sklearn.metrics import accuracy_score
from include.multiclass import multiclass_trainer as mt
import include.crossValidation as cv
import include.utility
import math
# size_of_training = 5000;
# createNewTrainingFileWithSize(size_of_training)



import xgboost as xgb
# read in data

k = 8 
data = cv.CV(k)
X_train, Y_train, X_valid, Y_valid = data.iteration(2)
# specify parameters via 
Y_train = np.ravel(pp.convertUStoBinary(Y_train))
Y_valid = np.ravel(pp.convertUStoBinary(Y_valid))

from sklearn import svm
clf = svm.SVC(probability=True)
clf.fit(X_train, Y_train)
print(len((clf.support_vectors_)))


# X_train, Y_train = downSampling(X_train, Y_train)

# dtrain = xgb.DMatrix(X_train, label = Y_train)

# dtest = xgb.DMatrix(X_valid, label = Y_valid)

# param = {'bst:max_depth':2, 'bst:eta':1, 'silent':1, 'objective':'binary:logistic' }
# param['nthread'] = 4
# param['eval_metric'] = 'auc'

# evallist  = [(dtest,'eval'), (dtrain,'train')]

# num_round = 1000
# bst = xgb.train( param, dtrain, num_round, evallist )



# idx1 = np.where(Y_valid == 1)
# idx0 = np.where(Y_valid == 0)
# ratio = len(idx1[0]) / (len(idx1[0]) + len(idx0[0]))
# print(ratio)

# model = xgb.XGBClassifier()
# model.fit(X_train, Y_train)
# result = model.predict(X_train)
# print(result)
# print('')
# print(Y_valid)

# predictions = [round(value) for value in result]
# accuracy = accuracy_score(Y_train, predictions)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))

#createNewTrainingFileWithSize(50000)


k = 3
data = cv.CV(k)
X_train, Y_train, X_valid, Y_valid = data.iteration(2)


trainer1 = mt.multiclass_trainer()
trainer1.train(X_train, Y_train, 3)
#result1 = trainer1.predict(X_valid)


trainer2 = mt.multiclass_trainer()
trainer2.train(X_train, Y_train, 3)
#result2 = trainer2.predict(X_valid)

trainers = [trainer1, trainer2]
x = X_valid