# import matplotlib.pyplot as plt
import include.crossValidation as cv
import include.binary.binaryClassifier as bc
from include.utility import *

if __name__ == "__main__":

	# size_of_training = 1000;
	# createNewTrainingFileWithSize(size_of_training)
	
	# destination, users = pp.readRawData() 
	# pp.writeToFile(users, destination, 'Data/user.dat', 'Data/destination.dat', cvs = False)


	k = 8 
	data = cv.CV(k)
	X_train, Y_train, X_valid, Y_valid = data.iteration(2)
	# print(X_valid)
	train_data = cv.CV(k, X_train, Y_train)

	bag_size = 21
	b_classifier = bc.binaryClassifier(bag_size)
	b_classifier.train(train_data)
	result = b_classifier.predict(X_valid)
	
	# check

	score = binaryEvaluation(result, pp.convertUStoBinary(Y_valid))

	print(score)

	#print(Y_test)
	#print(len(Y_test))
	# print(len(X_train))
	# print(len(Y_train))