import numpy as np
from utility import *
from sklearn import linear_model
import crossValidation

class logisticRegression_binary:
	def __init__(self):
		self.score = []
		self.trainers = []

	def train(self, instance, label, k):
		cv = crossValidation.CV(k, instance, np.ravel(convertUStoBinary(label)))
		for i in range(k):
			# X_train, Y_train, X_valid, Y_valid = data.iteration(i)
			# Y_train = self.convertUStoBinary(Y_train)
			# Y_valid = self.convertUStoBinary(Y_valid)
			# Y_train = np.ravel(Y_train)


			X_train, Y_train, X_valid, Y_valid = cv.iteration(i)
			regulation_strength = 10**i
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

