import numpy as np
from utility import *
from sklearn import linear_model

class logisticRegression_multi:
	def __init__(self):
		self.score = []
		self.trainers = []

	def getTopProb(self,prob,classes):
		top = 5
		num = len(prob)
		res = []
		for i in range(num):
			res_row = []
			for t in range(top):
				row = prob[i].tolist()
				topProbIdx = row.index(max(row))
				res_row.append(classes[topProbIdx])
				prob[i][topProbIdx] = 0
			res.append(res_row)
		return res

	def train(self,dtat,k):
		X_train, Y_train, X_test, Y_test = data.iteration(i)
		Y_train = np.ravel(Y_train)

		trainer = linear_model.LogisticRegression(C=1e5)
		trainer.fit(X_train, Y_train)
		prob = trainer.predict_prob(X_test)
		prob = getTopProb(prob,trainer.classes_)

		score = multiEvaluation(prob,Y_test)
