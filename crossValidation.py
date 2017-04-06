import numpy as np
import preProcessing as pp

class CV:
	def __init__(self, k):
		self.k = k
		self.instance, self.label = pp.readFromFile()
		self.idx = np.random.permutation(label.shape[0])

	def kfoldCV(self, nIters):
		assert(nIters >= 0 and nIters < self.k)

		row = self.label.shape[0]
		i = int(row / k)
		if nIters < self.k - 1:
			idx_test = self.idx[i * nIters: i * (nIters + 1)]
		else:
			idx_test = self.idx[i * nIters:]
		idx_train = self.idx[0: i * (nIters - 1)]
		idx_train = idx_train.append(self.idx[i * (nIters + 1):])

		Y_test = label[idx_test]
		X_test = instance[idx_test]
		Y_train = label[idx_train]
		X_train = instance[idx_train]

		return X_train, Y_train, X_test, Y_test