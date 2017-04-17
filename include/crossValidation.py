import numpy as np
import include.preProcessing as pp

x_train = "Data/users_1000.dat"
y_train = "Data/destination_1000.dat"

class CV:
	def __init__(self, k, instance = None, label = None):
		if instance is None and label is None:
			self.readFromFile(x_train, y_train)
		else:
			self.instance, self.label = instance, label
		self.setK(k)
			
	def iteration(self, nIters):
		assert(nIters >= 0 and nIters < self.k)
		row = self.label.shape[0]
		i = int(row / self.k)
		idx_train = self.idx[0: i * nIters]

		if nIters < self.k - 1:
			idx_valid = self.idx[i * nIters: i * (nIters + 1)]
			idx_train = np.append(idx_train, self.idx[i * (nIters + 1):])
		else:
			idx_valid = self.idx[i * nIters:]

		Y_valid = self.label[idx_valid]
		X_valid = self.instance[idx_valid]
		Y_train = self.label[idx_train]
		X_train = self.instance[idx_train]

		return X_train, Y_train, X_valid, Y_valid

	def setK(self, k):
		self.k = k
		self.idx = np.random.permutation(self.label.shape[0])

	def readFromFile(self, instance_file, label_file):
		self.instance, self.label = pp.readFromFile(instance_file, label_file)