import numpy as np
import crossValidation as cv
import logisticRegression_multiclass as lg 
import svmTrainers_multiclass as svm_m
import logisticRegression_multiclass as log_m
import binary_trainer as bt
import math
from utility import *

if __name__ == "__main__":

	b_trainer = bt.binary_trainer()

	k = 3
	data = cv.CV(k)
	X_train, Y_train, X_valid, Y_valid = data.iteration(1)

	b_trainer.train(X_valid, Y_valid, k)

	res = b_trainer.predict(X_valid)


	print(res)







