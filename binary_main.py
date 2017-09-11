# import matplotlib.pyplot as plt
import include.crossValidation as cv
import include.binary.binaryClassifier as bc
from include.binary import binary_trainer as bt
from include.utility import *
from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier


if __name__ == "__main__":

	k = 8 
	data = cv.CV(k)
	X_train, Y_train, X_valid, Y_valid = data.iteration(2)

	train_data = cv.CV(k, X_train, Y_train)

	bag_size = 11
	b_classifier = bc.binaryClassifier(bag_size)
	b_classifier.train(train_data)

	Y_valid = np.ravel(pp.convertUStoBinary(Y_valid))
	X_valid, Y_valid = upSampling(X_valid, Y_valid)

	result = b_classifier.predict(X_valid)

	# # check
	score = binaryEvaluation(result, Y_valid)

	print(score)