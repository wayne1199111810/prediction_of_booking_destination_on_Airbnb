import numpy as np
import logisticRegression_binary as lg
import polyRegression_binary as pl
import svmTrainers_binary as svm
import NBTrainers_binary as nb

class binaryClassifier:
	def __init__(self):
		self.lg_trainer = 0
		self.pl_trainer = 0
		self.svm_trainer = 0
		self.nb_trainer = 0

	def vote(self, res1, res2, res3):
		num = len(res1)
		result = np.zeros((num,1))
		for i in range(num):
			if res1[i]+res2[i]+res3[i]>=1:
				result[i] = 1
		return result


	def train(self,data,k):
		# logistic regression
		lg_trainers = lg.logisticRegression_binary()
		lg_trainers.train(data,k)
		self.lg_trainer = lg_trainers.getTrainer()
		
		# polynomial regression
		#pl_trainers = pl.polyRegression_binary()
		#pl_trainers.train(data,k,3)
		#self.pl_trainer = pl_trainers.getTrainer()
		
		# SVM
		svm_trainers = svm.svmTrainers_binary()
		svm_trainers.train(data,k)
		self.svm_trainer = svm_trainers.getTrainer()
		
		# NB
		nb_trainers = nb.NBTrainers_binary()
		nb_trainers.train(data,k)
		self.nb_trainer = nb_trainers.getTrainer()

	def test(self,data):
		res_lg = self.lg_trainer.predict(data)
		#res_pl = self.pl_trainer.predict(data)
		res_svm = self.svm_trainer.predict(data)
		res_nb = self.nb_trainer.predict(data)
		result = self.vote(res_lg, res_svm, res_nb)
		return result


	