import numpy as np
import logisticRegression_binary as lg
import polyRegression_binary as pl
import svmTrainers_binary as svm_b
import NBTrainers_binary as nb


from sklearn import linear_model
from sklearn import svm
from sklearn.naive_bayes import GaussianNB

from utility import *

class binaryClassifier:
	def __init__(self):
		self.lg_trainer = linear_model.LogisticRegression()
		# self.pl_trainer = None
		self.svm_trainer = svm.SVC(probability=True)
		self.nb_trainer = GaussianNB()

	def vote(self, res1, res2, res3):
		num = len(res1)
		result = np.zeros((num,1))
		for i in range(num):
			if res1[i]+res2[i]+res3[i]>=1:
				result[i] = 1
		return result


	def train(self, cv, k):
		score = 0
		for i in range(k):
			X_train, Y_train, X_valid, Y_valid = cv.iteration(i)

			Y_valid = convertUStoBinary(Y_valid)

			lg_trainers = lg.logisticRegression_binary()
			lg_trainers.train(X_train, Y_train, k)

			svm_trainers = svm_b.svmTrainers_binary()
			svm_trainers.train(X_train, Y_train, k)

			nb_trainers = nb.NBTrainers_binary()
			nb_trainers.train(X_train, Y_train, k)

			res_lg = lg_trainers.getTrainer().predict(X_valid)
			res_svm = svm_trainers.getTrainer().predict(X_valid)
			res_nb = nb_trainers.getTrainer().predict(X_valid)
			result = self.vote(res_lg, res_svm, res_nb)

			accuracy = binaryEvaluation(np.ravel(result), np.ravel(Y_valid))

			if accuracy > score:
				score = accuracy
				self.lg_trainer = lg_trainers.getTrainer()
				self.svm_trainer = svm_trainers.getTrainer()
				self.nb_trainer = nb_trainers.getTrainer()
			print(' Iteration ' + str(i) + ' accuracy:' + str(accuracy))

	def predict(self, instance):
		res_lg = self.lg_trainer.predict(instance)
		#res_pl = self.pl_trainer.predict(instance)
		res_svm = self.svm_trainer.predict(instance)
		res_nb = self.nb_trainer.predict(instance)
		result = self.vote(res_lg, res_svm, res_nb)
		return result


