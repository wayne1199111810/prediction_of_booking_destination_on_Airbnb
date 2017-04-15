import numpy as np

def binaryEvaluation(result, Y_test):
	print(result.shape)
	print(Y_test.shape)
	assert result.shape == Y_test.shape
	num = len(result)
	score = 0
	for i in range(num):
		if result[i] == Y_test[i]:
			score = score + 1
	return score / num

#def multiEvaluation(result, Y_test):

def convertUStoBinary(country):
	num = len(country)
	res = np.zeros((num, 1))
	for i in range(num):
		if country[i] == 'US':
			res[i] = 1
		else:
			res[i] = 0
	return res