import numpy as np
import logisticRegression_binary as lg
import polyRegression_binary as pl
import svmTrainers_binary as svm_b
import NBTrainers_binary as nb
from sklearn import linear_model
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
import preProcessing as pp

from utility import *

class binaryClassifier:
	class Trainer:
		def __init__(self, lg_t, svm_t, nb_t):
			self.lg_trainer = lg_t
			self.svm_trainer = svm_t
			self.nb_trainer = nb_t
			# self.poly_trainer = poly_t

	def __init__(self, bag_size = 9):
		self.trainers = []
		self.bag_size = bag_size

	def train(self, cv, k):
		score = 0
		for i in range(k):
			X_train, Y_train, X_valid, Y_valid = cv.iteration(i)
			Y_valid = pp.convertUStoBinary(Y_valid)
			Y_train = np.ravel(pp.convertUStoBinary(Y_train))
			trainers = self.bagOfTrainers(X_train, Y_train, k)
			result = self.predictFromTrainers(X_valid)
			accuracy = binaryEvaluation(np.ravel(result), np.ravel(Y_valid))
			if accuracy > score:
				self.trainers = trainers
			print(' Iteration ' + str(i) + ' accuracy:' + str(accuracy))

	def predict(self, instance):
		result = self.predictFromTrainers(instance)
		return result

	def vote(self, res1, res2, res3):
		num = len(res1)
		result = np.zeros((num,1))
		for i in range(num):
			if res1[i]+res2[i]+res3[i] >= 2:
				result[i , 0] = 1
		return result

	def bagOfTrainers(self, X_train, Y_train, k = 1):
		trainers = []
		for j in range(self.bag_size):
			x, y = subSampleFromBagging(X_train, Y_train)
			print(y.shape)
			lg_trainers = lg.logisticRegression_binary()
			lg_trainers.train(x, y, k)

			svm_trainers = svm_b.svmTrainers_binary()
			svm_trainers.train(x, y, k)

			nb_trainers = nb.NBTrainers_binary()
			nb_trainers.train(x, y, k)

			# poly_trainers = pl.polyRegression_binary()
			# poly_trainers.train(x, y, k)			

			res_lg = lg_trainers.getTrainer()
			res_svm = svm_trainers.getTrainer()
			res_nb = nb_trainers.getTrainer()

			trainers.append(binaryClassifier.Trainer(res_lg, res_svm, res_nb))
		return trainers

	def predictFromTrainers(self, X_valid):
		num_of_sample = X_valid.shape[0]
		set_of_result = np.zeros((num_of_sample, len(self.trainers)))
		for i in range(len(self.trainers)):
			res_lg = self.trainers[i].lg_trainer.predict(X_valid)
			res_svm = self.trainers[i].svm_trainer.predict(X_valid)
			res_nb = self.trainers[i].nb_trainer.predict(X_valid)
			result_of_one_trainers = self.vote(res_lg, res_svm, res_nb)
			set_of_result[:, i] = result_of_one_trainers.T
		result = self.binaryVoteFromBags(set_of_result)
		return result

	def binaryVoteFromBags(self, set_of_result):
		num_of_sample = set_of_result.shape[0]
		result = np.zeros((num_of_sample, 1))
		for i in range(num_of_sample):
			if sum(set_of_result[i, :]) > int(set_of_result.shape[1] / 2):
				result[i, 0] = 1
		return result