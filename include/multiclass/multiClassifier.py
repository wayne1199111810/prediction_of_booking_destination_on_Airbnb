import numpy as np
from include.multiclass import logisticRegression_multiclass as lg
from include.multiclass import svmTrainers_multiclass as svm
from sklearn.naive_bayes import GaussianNB
from include import preProcessing as pp
from include.Bag import *
from include.utility import *

class multiClassifier:
	def __init__(self, bag_size = 9):
		self.mTrainers = []
		self.bag_size = bag_size

	def train(self, cv):
		score = 0
		for i in range(cv.k):
			X_train, Y_train, X_valid, Y_valid = cv.iteration(i)
			# Y_valid = np.ravel(pp.convertUStoBinary(Y_valid))
			# Y_train = np.ravel(pp.convertUStoBinary(Y_train))

			mTrainers = Bag.bagOfMultiTrainers(X_train, Y_train, self.bag_size, cv.k * 2)
			result = Bag.predictFromMultiTrainers(X_valid, mTrainers)
			accuracy = multiclassEvaluation(result, Y_valid)
			if accuracy > score:
				self.mTrainers = mTrainers
			print(' Iteration ' + str(i) + ' accuracy:' + str(accuracy))
			# print(' Iteration ' + str(i))

	def predict(self, instance):
		result = Bag.predictFromMultiTrainers(instance, self.mTrainers)
		return result