import numpy as np
from utility import *
from sklearn import svm
import crossValidation

class svmTrainers_binary:
	def __init__(self):
		self.score = []
		self.trainers = []

	def train(self, instance, label, k):
		cv = crossValidation.CV(k, instance, label)
		for i in range(k):
			X_train, Y_train, X_valid, Y_valid = cv.iteration(i)

			trainer = svm.SVC(probability=True)
			trainer.fit(X_train, Y_train)
			result = trainer.predict(X_valid)

			score = binaryEvaluation(result, Y_valid)

			self.score.append(score)
			self.trainers.append(trainer)

	def getTrainer(self):
		trainer = self.trainers[ self.score.index(max(self.score)) ]	# get the trainer with highest score
		return trainer

	def predict(instance):
		return self.getTrainer().predict(instance)



