# import matplotlib.pyplot as plt
import preProcessing as pp
import crossValidation as cv
import binaryClassifier as bc
from sklearn import linear_model
import numpy as np
from evaluation import *

if __name__ == "__main__":
	#pp.writeToFile()
	#instance, label = pp.readFromFile()
	k = 4
	data = cv.CV(k)
	X_train, Y_train, X_test, Y_test = data.iteration(1)

	b_classifier = bc.binaryClassifier()
	b_classifier.train(data,k)
	result = b_classifier.test(X_test)

	# check
	num = len(Y_test)
	tmp = np.zeros((num,1))
	for i in range(num):
			if Y_test[i]=='US':
				tmp[i] = 1
			else:
				tmp[i] = 0
	Y_test = tmp

	score = binaryEvaluation(result, Y_test)

	print(score)

	#print(Y_test)
	#print(len(Y_test))
	# print(len(X_train))
	# print(len(Y_train))
