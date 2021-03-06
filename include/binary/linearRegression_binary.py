import numpy as np
from utility import *
from sklearn import linear_model

class linearRegression:
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
				res[i] = -1
		return res

	def train(self, data, k):

		for i in range(k):
			
			X_train, Y_train, X_test, Y_test = data.iteration(i)
			Y_train = self.convertUStoBinary(Y_train)
			Y_test = self.convertUStoBinary(Y_test)

			trainer = linear_model.LinearRegression()
			trainer.fit(X_test, Y_test)

			result = trainer.predict(X_test)

			score = binaryEvaluation(result, Y_test)

			self.score.append(score)
			self.trainers.append(trainer)

	def getTrainer(self):
		trainer = self.trainers[ self.score.index(max(self.score)) ]	# get the trainer with highest score
		return trainer

	def getScores(self):
		return self.score