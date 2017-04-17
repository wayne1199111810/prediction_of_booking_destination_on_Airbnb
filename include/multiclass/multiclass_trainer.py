import numpy as np
import crossValidation as cv
import logisticRegression_multiclass as lg 
import svmTrainers_multiclass as svm_m
import logisticRegression_multiclass as log_m
import math
from sklearn import linear_model
from sklearn import svm
from utility import *

class multiclass_trainer:
	def __init__(self):
		self.lg_trainer = linear_model.LogisticRegression()
		self.svm_trainer = svm.SVC(probability=True)

	def getTopProbResult(self, prob1, classes1, prob2, classes2):
		k = 5
		dimension = len(classes.tolist())
		num = len(prob1)
		result = np.empty([num, k], dtype=object)
		for i in range(num):
			row1 = prob1[i].tolist()
			row2 = prob2[i].tolist()
			row = row1 + row2
			for j in range(k):
				max_idx = row.index(max(row))
				if max_idx >= dimension:
					result[i][j] = classes2[max_idx-dimension]
					row2[max_idx-dimension] = 0
					row1[classes1.index(result[i][j])] = 0
				else:
					result[i][j] = classes1[max_idx]
					row1[max_dix] = 0
					row2[classes2.index[result[i][j]]] = 0
		return result


	def train(self, instance, label, k):
		# logistic trainer
		log_trainers = log_m.logisticRegression_multi()
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

