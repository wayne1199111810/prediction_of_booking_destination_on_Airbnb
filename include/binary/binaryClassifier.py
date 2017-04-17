import numpy as np
from include.binary import logisticRegression_binary as lg
from include.binary import polyRegression_binary as pl
from include.binary import svmTrainers_binary as svm_b
from include.binary import NBTrainers_binary as nb
from sklearn import linear_model
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from include import preProcessing as pp
from include.Bag import *
from include.utility import *

class binaryClassifier:
	def __init__(self, bag_size = 9):
		self.bTrainers = []
		self.bag_size = bag_size

	def train(self, cv):
		score = 0
		for i in range(cv.k):
			X_train, Y_train, X_valid, Y_valid = cv.iteration(i)
			Y_valid = np.ravel(pp.convertUStoBinary(Y_valid))
			Y_train = np.ravel(pp.convertUStoBinary(Y_train))

			bTrainers = Bag.bagOfBinaryTrainers(X_train, Y_train, self.bag_size, cv.k * 4)
			result = Bag.predictFromBinaryTrainers(X_valid, bTrainers)
			accuracy = binaryEvaluation(np.ravel(result), Y_valid)
			if accuracy > score:
				self.bTrainers = bTrainers
			print(' Iteration ' + str(i) + ' accuracy:' + str(accuracy))
			# print(' Iteration ' + str(i))

	def predict(self, instance):
		result = Bag.predictFromBinaryTrainers(instance, self.bTrainers)
		return result