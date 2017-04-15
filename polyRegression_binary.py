import numpy as np
from utility import *
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import crossValidation

class polyRegression_binary:
	def __init__(self):
		self.score = []
		self.trainers = []

	def decideLabel(self, result):
		num = len(result)
		res = np.zeros((num))
		for i in range(num):
			if result[i] >= 0.5:
				res[i] = 1
		return np.reshape(res, (res.shape[0],))


	def train(self, instance, label, k ,degree = 2):
		cv = crossValidation.CV(k, instance, label)
		for i in range(k):
			degree = i
			X_train, Y_train, X_valid, Y_valid = cv.iteration(i)

			trainer = make_pipeline(PolynomialFeatures(degree), Ridge())
			trainer.fit(X_train, Y_train)
			result = trainer.predict(X_valid)
			result = self.decideLabel(result)

			score = binaryEvaluation(result, Y_valid)

			self.score.append(score)
			self.trainers.append(trainer)

	def getTrainer(self):
		trainer = self.trainers[ self.score.index(max(self.score)) ]	# get the trainer with highest score
		return trainer

	def predict(self, instance):
		return self.getTrainer().predict(data)