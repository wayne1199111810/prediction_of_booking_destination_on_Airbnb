# import matplotlib.pyplot as plt
import preProcessing as pp
import crossValidation as cv
import binaryClassifier as bc
from sklearn import linear_model
import numpy as np
from utility import *


if __name__ == "__main__":

	#pp.writeToFile()
	#instance, label = pp.readFromFile()
	k = 2
	data = cv.CV(k)
	X_train, Y_train, X_valid, Y_valid = data.iteration(1)

	b_classifier = bc.binaryClassifier()
	b_classifier.train(data,k)
	result = b_classifier.predict(X_valid)

	# check

	score = binaryEvaluation(result, convertUStoBinary(Y_valid))

	print(score)

	#print(Y_test)
	#print(len(Y_test))
	# print(len(X_train))
	# print(len(Y_train))
