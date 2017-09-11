# import matplotlib.pyplot as plt
import include.crossValidation as cv
import include.binary.binaryClassifier as bc
import include.multiclass.multiClassifier as mc
from include.utility import *
from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier


if __name__ == "__main__":

	### Get Data ###
	k = 8
	bag_size = 11
	data = cv.CV(k)
	X_train, Y_train, X_valid, Y_valid = data.iteration(0)
	Y_valid.dump('Y_valid.dat')

	### Binary Classify
	train_data_b = cv.CV(k, X_train, Y_train)
	b_classifier = bc.binaryClassifier(bag_size)
	b_classifier.train(train_data_b)
	result_b = b_classifier.predict(X_valid)
	Y_valid_b = convertUStoBinary(Y_valid)
	score_b = binaryEvaluation(result_b, Y_valid_b)
	print(score_b)
	result_b.dump('result_b.dat')

	### Multiclass Classify
	X_train_m, Y_train_m = extractNonUs(X_train, Y_train)
	train_data_m = cv.CV(k, X_train_m, Y_train_m)
	m_classifier = mc.multiClassifier(bag_size)
	m_classifier.train(train_data_m)
	result_m = m_classifier.predict(X_valid)
	score_m = multiclassEvaluation(result_m, Y_valid)
	print(score_m)
	

	### Calculate score
	for i in range(len(Y_valid)):
		if result_b[i] == 1:
			result_m[i][4] = result_m[i][3]
			result_m[i][3] = result_m[i][2]
			result_m[i][2] = result_m[i][1]
			result_m[i][1] = result_m[i][0]
			result_m[i][0] = 'US'
	score = multiclassEvaluation(result_m, Y_valid)
	
	result_b.dump('result_b.dat')
	result_m.dump('result_m.dat')
	print(score)