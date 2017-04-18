# import matplotlib.pyplot as plt
import include.crossValidation as cv
import include.binary.binaryClassifier as bc
from include.binary import binary_trainer as bt
from sklearn import svm
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from include.preProcessing import *
from include.utility import *
from sklearn.metrics import accuracy_score

# get data
label, instance = readRawData()

# print (label.shape)
# print (instance.shape)
# exit()

data_file = 'data.txt'
f = open(data_file, 'w')
for i in range(label.shape[0]):
	label_str = label[i, 0]
	if label_str == 'US':
		lbl = 1
	else:
		lbl = 0
	f.write('{}'.format(lbl))
	for j in range(instance.shape[1]):
		if instance[i, j] != 0:
			f.write(' {}:{}'.format(j+1, instance[i, j]))
	f.write('\n')
f.close()
exit()

# divide data into training and testing
num = len(instance)
X_train = instance[:(int)(0.8*num)]
X_valid = instance[(int)(0.8*num):]
Y_train = label[:(int)(0.8*num)]
Y_valid = label[(int)(0.8*num):]

Y_train = pp.convertUStoBinary(Y_train)
Y_valid = pp.convertUStoBinary(Y_valid)

#print(Y_train)


#print(sum(Y_train))
#print(len(Y_train))


# down sampling
data_dm, label_dm, data_ud, label_ud = downSample(X_train, Y_train, 1)
X_train = np.append(data_ud,data_dm[:len(data_ud)],axis=0)
Y_train = np.append(label_ud,label_dm[:len(label_ud)],axis=0)


# create training model
trainer = svm.SVC(C=1e5, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)


# train
Y_train_b = np.ravel(Y_train)
trainer.fit(X_train, Y_train_b)


# predict
result = trainer.predict(X_valid)


print(result)


# evaluation
Y_valid_b = np.ravel(Y_valid)
score = binaryEvaluation(result, Y_valid_b)


print(Y_valid)

#score = accuracy_score(Y_valid, result)
print(score)

s = 0
for i in range(len(result)):
	if result[i] == 1:
		s = s+1
print(s/len(result))

