import numpy as np
import preProcessing as pp

x_train = "Data/users.dat"
y_train = "Data/destination.dat"

class CV:
	def __init__(self, k, instance = None, label = None):
		self.k = k
		if instance is None and label is None:
			self.instance, self.label = pp.readFromFile(x_train, y_train)
			self.idx = np.random.permutation(self.label.shape[0])
		else:
			self.instance, self.label = instance, label
			self.idx = np.random.permutation(self.label.shape[0])
			
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