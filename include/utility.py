import numpy as np
import include.preProcessing as pp
import math

def binaryEvaluation(result, Y_test):
	# print(str(result.shape) + ', ' + str(Y_test.shape))
	assert result.shape == Y_test.shape
	num = len(result)
	score = 0
	for i in range(num):
		if result[i] == Y_test[i]:
			score = score + 1
	return score / num

# nDCG calculation
def multiclassEvaluation(result, Y_test):
	k = 5
	num = len(result)
	score = 0
	for i in range(num):
		for j in range(k):
			if result[i][j] == Y_test[i]:
				score = score + 1/(math.log(j+2,2))
				break
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

# Extract samples whose destination is non-US
def extractNonUs(data, label):
	dimension = len(data[0].tolist()[0])
	data_new = np.zeros((0,dimension))
	label_new = np.zeros((0,1))

	for i in range(len(data)):
		if label[i] != 'US':
			data_new = np.vstack((data_new, data[i]))
			label_new = np.vstack((label_new, label[i]))
	return data_new, label_new

def downSample(data, label, dominant):
	dimension = len(data[0].tolist()[0])
	idx_dm = np.where(label == dominant)[0]
	idx_ud = np.where(label != dominant)[0]

	data_dm = np.zeros((len(idx_dm), dimension))
	data_ud = np.zeros((len(idx_ud), dimension))
	label_dm = np.ones((len(idx_dm),1))
	label_ud = np.zeros((len(idx_ud),1))

	for i in range(len(idx_dm)):
		data_dm[i] = data[idx_dm[i]]

	for i in range(len(idx_ud)):
		data_ud[i] = data[idx_ud[i]]

	return data_dm, label_dm, data_ud, label_ud





