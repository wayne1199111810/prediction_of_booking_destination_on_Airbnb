def binaryEvaluation(result, Y_test):
	num = len(result)
	score = 0
	for i in range(num):
		if result[i]==Y_test[i]:
			score = score + 1
	return score/num

#def multiEvaluation(result, Y_test):


