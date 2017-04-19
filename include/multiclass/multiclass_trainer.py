import numpy as np
from include import crossValidation as cv
from include.multiclass import svmTrainers_multiclass as svm_m
from include.multiclass import logisticRegression_multiclass as log_m
import math
from sklearn import linear_model
from sklearn import svm
from include.utility import *

class multiclass_trainer:
	def __init__(self):
		self.lg_trainer = linear_model.LogisticRegression()
		self.svm_trainer = svm.SVC(probability=True)

	def getTopProbResult(self, prob1, classes1, prob2, classes2):
		k = 5
		classes1 = classes1.tolist()
		classes2 = classes2.tolist()
		num = len(prob1)
		dimension = len(classes1)
		classes = classes1
		result = np.empty([num,k], dtype=object)

		for i in range(num):
			row1 = prob1[i].tolist()
			row2 = prob2[i].tolist()
			row = row1
			for j in range(dimension):
				row[j] = row1[j] + row2[classes2.index(classes1[j])]
			for j in range(k):
				result[i][j] = classes[row.index(max(row))]
				row[row.index(max(row))] = 0
		return result


	def train(self, instance, label, k):
		# logistic trainer
		log_trainers = log_m.logisticRegression_multiclass()
		log_trainers.train(instance, label, k)
		self.lg_trainer = log_trainers.getTrainer()

		# SVM trainer
		svm_trainers = svm_m.svmTrainers_multiclass()
		svm_trainers.train(instance, label, k)
		self.svm_trainer = svm_trainers.getTrainer()

	def predict(self, instance):
		log_prob = self.lg_trainer.predict_proba(instance)
		svm_prob = self.svm_trainer.predict_proba(instance)
		
		return self.getTopProbResult(log_prob, self.lg_trainer.classes_, svm_prob, self.svm_trainer.classes_)

