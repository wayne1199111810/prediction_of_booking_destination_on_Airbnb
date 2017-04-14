import numpy as np
from evaluation import *
from sklearn import svm

class svmTrainers_binary:
	def __init__(self):
		self.score = []
		self.trainers = []

	def convertUStoBinary(self,country):
		num = len(country)
		res = np.zeros((num,1))
		for i in range(num):
			if country[i]=='US':
				res[i] = 1
			else:
				res[i] = 0
		return res

	def train(self,data,k):

		for i in range(k):
			X_train, Y_train, X_test, Y_test = data.iteration(i)
			Y_train = self.convertUStoBinary(Y_train)
			Y_test = self.convertUStoBinary(Y_test)
			Y_train = np.ravel(Y_train)

			trainer = svm.SVC(probability=True)
			trainer.fit(X_train, Y_train)
			result = trainer.predict(X_test)

			score = binaryEvaluation(result, Y_test)

			self.score.append(score)
			self.trainers.append(trainer)

	def getTrainer(self):
		trainer = self.trainers[ self.score.index(max(self.score)) ]	# get the trainer with highest score
		return trainer

	def text(data):
		trainer = self.getTrainer()
		result = trainer.predict(data)
		return result



