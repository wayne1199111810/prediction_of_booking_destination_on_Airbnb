import numpy as np
from utility import *
from sklearn import linear_model
import crossValidation

class logisticRegression_binary:
	def __init__(self):
		self.score = []
		self.trainers = []

	def train(self, instance, label, k, regulation_strength = 1e5):
		cv = crossValidation.CV(k, instance, label)
		for i in range(k):
			X_train, Y_train, X_valid, Y_valid = cv.iteration(i)
			trainer = linear_model.LogisticRegression(C=regulation_strength)
			trainer.fit(X_train, Y_train)
			result = trainer.predict(X_valid)

			score = binaryEvaluation(result, Y_valid)
			self.score.append(score)
			self.trainers.append(trainer)

	def getTrainer(self):
		trainer = self.trainers[ self.score.index(max(self.score)) ]	# get the trainer with highest score
		return trainer

	def predict(self, instance):
		return self.getTrainer().predict(instance)

