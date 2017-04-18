import numpy as np
from include.utility import *
from sklearn import svm
from include import crossValidation

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
			print(result)
			score = binaryEvaluation(result, Y_valid)

			self.score.append(score)
			self.trainers.append(trainer)

	def getTrainer(self):
		trainer = self.trainers[ self.score.index(max(self.score)) ]	# get the trainer with highest score
		return trainer

	def predict(instance):
		k = len(trainers)
		num = len(instance)
		res = np.zeros((num,1))
		for i in range(k):
			res = res + self.trainers[i].predict(instance) 
		for i in range(num):
			if res[i] >= k/2:
				res[i] = 1
			else:
				res[i] = 0
		return res
		#return self.getTrainer().predict(instance)



