# import matplotlib.pyplot as plt
import crossValidation as cv
import binaryClassifier as bc
from utility import *

if __name__ == "__main__":

	# size_of_training = 1000;
	# createNewTrainingFileWithSize(size_of_training)

	k = 8
	data = cv.CV(k)

	b_classifier = bc.binaryClassifier()
	b_classifier.train(data,k)
	result = b_classifier.predict(X_valid)

	# check

	score = binaryEvaluation(result, pp.convertUStoBinary(Y_valid))

	print(score)

	#print(Y_test)
	#print(len(Y_test))
	# print(len(X_train))
	# print(len(Y_train))
