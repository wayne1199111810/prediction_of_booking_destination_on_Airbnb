import numpy as np
from evaluation import *
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

class polyRegression_binary:
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

	def decideLabel(self,result):
		num = len(result)
		res = np.zeros((num,1))
		for i in range(num):
			if result[i]>=0.5:
				res[i] = 1
		return res


	def train(self,data,k,degree):

		for i in range(k):
			X_train, Y_train, X_test, Y_test = data.iteration(i)
			Y_train = self.convertUStoBinary(Y_train)
			Y_test = self.convertUStoBinary(Y_test)

			trainer = make_pipeline(PolynomialFeatures(degree), Ridge())
			trainer.fit(X_train, Y_train)
			result = trainer.predict(X_test)
			result = self.decideLabel(result)

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






