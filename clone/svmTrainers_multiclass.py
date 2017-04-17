import numpy as np
from utility import *
from sklearn import svm
import crossValidation

class svmTrainers_multiclass:
	def __init__(self):
		self.score = []
		self.trainers = []

	def getTopProbResult(self, prob, classes):
		k = 5
		num = len(prob)
		result = np.empty([num, k], dtype=object)
		for i in range(num):
			row = prob[i].tolist()
			for j in range(k):
				result[i][j] = classes[row.index(max(row))]
				row[row.index(max(row))] = 0
		return result


	def train(self, instance, label, k):
		cv = crossValidation.CV(k, instance, label)
		for i in range(k):
			X_train, Y_train, X_valid, Y_valid = cv.iteration(i)

			trainer = svm.SVC(probability=True)
			trainer.fit(X_train, Y_train)

			#result = trainer.perdict(X_valid)
			prob = trainer.predict_proba(X_valid)

			result = self.getTopProbResult(prob, trainer.classes_)

			score = multiclassEvaluation(result, Y_valid)

			self.score.append(score)
			self.trainers.append(trainer)


	def getTrainer(self):
		trainer = self.trainers[ self.score.index(max(self.score)) ]	# get the trainer with highest score
		return trainer

	def predict(self,instance):
		prob = self.getTrainer().predict_proba(instance)
		result = self.getTopProbResult(prob, self.getTrainer().classes_)
		return result

	def getScore(self):
		return self.score