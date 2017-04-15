import numpy as np
import preProcessing as pp

def binaryEvaluation(result, Y_test):
	# print(str(result.shape) + ', ' + str(Y_test.shape))
	assert result.shape == Y_test.shape
	num = len(result)
	score = 0
	for i in range(num):
		if result[i] == Y_test[i]:
			score = score + 1
	return score / num

def convertUStoBinary(country):
	num = len(country)
	res = np.zeros((num, 1))
	for i in range(num):
		if country[i] == 'US':
			res[i] = 1
		else:
			res[i] = 0
	return res

# Create a new file with the first size_of_training
# data in training data
def createNewTrainingFileWithSize(size_of_training):
	train_user = 'Data/users_train.dat'
	train_destination = 'Data/destination_train.dat'
	new_user = 'Data/users_' + str(size_of_training) + '.dat'
	new_destination = 'Data/destination_' + str(size_of_training) + '.dat'
	instance, label = pp.readFromFile(train_user, train_destination)
	pp.writeToFile(instance[0:size_of_training], label[0:size_of_training], new_user, new_destination)
