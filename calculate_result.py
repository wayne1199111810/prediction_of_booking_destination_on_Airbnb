import numpy as np
from include.utility import *

Y_valid = np.load('Y_valid.dat')
result_b = np.load('result_b.dat')
result_m = np.load('result_m.dat')

print(len(Y_valid))
print(len(result_b))
print(len(result_m))



for i in range(len(Y_valid)):
	if result_b[i] == 1:
		result_m[i][4] = result_m[i][3]
		result_m[i][3] = result_m[i][2]
		result_m[i][2] = result_m[i][1]
		result_m[i][1] = result_m[i][0]
		result_m[i][0] = 'US'

score = multiclassEvaluation(result_m, Y_valid)

print(score)