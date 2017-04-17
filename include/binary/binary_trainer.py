import numpy as np
from include.binary import logisticRegression_binary as lg_b
from include.binary import polyRegression_binary as pl
from include.binary import svmTrainers_binary as svm_b
from include.binary import NBTrainers_binary as nb_b
from sklearn import linear_model
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from include.utility import *

# TODO reassign constructor
class BinaryTrainer:
	def __init__(self):
		self.lg_trainer = linear_model.LogisticRegression()
		self.svm_trainer = svm.SVC(probability=True)
		self.nb_trainer = GaussianNB()

	def vote(self, res1, res2, res3):
		num = len(res1)
		result = np.zeros((num,1))
		for i in range(num):
			if res1[i]+res2[i]+res3[i]>=2:
				result[i, 0] = 1
		return result

	def train(self, instance, label, k):
		# logistic trainer
		lg_trainers = lg_b.logisticRegression_binary()
		lg_trainers.train(instance, label, k)
		self.lg_trainer = lg_trainers.getTrainer()

		# SVM trainer
		svm_trainers = svm_b.svmTrainers_binary()
		svm_trainers.train(instance, label, k)
		self.svm_trainer = svm_trainers.getTrainer()

		# NB trainer
		nb_trainers = nb_b.NBTrainers_binary()
		nb_trainers.train(instance, label, k)
		self.nb_trainer = nb_trainers.getTrainer()

	def predict(self, instance):
		res_lg = self.lg_trainer.predict(instance)
		res_svm = self.svm_trainer.predict(instance)
		res_nb = self.nb_trainer.predict(instance)
		result = self.vote(res_lg, res_svm, res_nb)
		return result

