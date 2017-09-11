# import matplotlib.pyplot as plt
import include.crossValidation as cv
import include.multiclass.multiClassifier as mc
from include.utility import *
from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier


if __name__ == "__main__":

	k = 8 
	data = cv.CV(k)
	X_train, Y_train, X_valid, Y_valid = data.iteration(2)
	X_train, Y_train = extractNonUs(X_train, Y_train)
	X_valid, Y_valid = extractNonUs(X_valid, Y_valid)

	train_data = cv.CV(k, X_train, Y_train)


	bag_size = 11
	m_classifier = mc.multiClassifier(bag_size)
	m_classifier.train(train_data)
	result = m_classifier.predict(X_valid)
	
	# # check
	score = multiclassEvaluation(result, Y_valid)

	print(score)