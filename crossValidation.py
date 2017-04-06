import numpy as np
import preProcessing as pp

class CV:
	def __init__(self, k):
		self.k = k
		self.instance, self.label = pp.readFromFile()
		self.idx = np.random.permutation(self.label.shape[0])

	def iteration(self, nIters):
		assert(nIters >= 0 and nIters < self.k)
		row = self.label.shape[0]
		i = int(row / self.k)
		idx_train = self.idx[0: i * nIters]

		if nIters < self.k - 1:
			idx_test = self.idx[i * nIters: i * (nIters + 1)]
			idx_train = np.append(idx_train, self.idx[i * (nIters + 1):])
		else:
			idx_test = self.idx[i * nIters:]

		Y_test = self.label[idx_test]
		X_test = self.instance[idx_test]
		Y_train = self.label[idx_train]
		X_train = self.instance[idx_train]

		return X_train, Y_train, X_test, Y_test